"""Durable aggregate telemetry for resumable per-facet classification passes."""

from __future__ import annotations

import copy
import json
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rumen.models import (
    ClassificationCorrection,
    ClassifierMetricSnapshot,
    FeedItem,
    SkosConcept,
)
from rumen.taxonomy.hmc.fit import FacetFitReport

if TYPE_CHECKING:
    from rumen.classifier.cascade import FacetDecision

_BINS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _zero_facet_metrics() -> dict:
    return {
        "eligible_items": 0,
        "eligible_facet_decisions": 0,
        "tier0_resolved": 0,
        "tier1_evaluated": 0,
        "tier1_accepted": 0,
        "tier1_confidence_bins": {name: 0 for name in _BINS},
        "tier2_attempted": 0,
        "tier2_attempted_item_count": 0,
        "tier2_resolved": 0,
        "unclassified": 0,
    }


def _gold_metrics(report: FacetFitReport | None) -> dict:
    evaluation = report.gold_evaluation if report is not None else None
    if evaluation is None or not evaluation.evaluated:
        return {
            "available": False,
            "evaluated_count": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": None,
            "recall": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
        }
    positives = evaluation.true_positives + evaluation.false_positives
    gold = evaluation.true_positives + evaluation.false_negatives
    return {
        "available": True,
        "evaluated_count": evaluation.evaluated_count,
        "true_positives": evaluation.true_positives,
        "false_positives": evaluation.false_positives,
        "false_negatives": evaluation.false_negatives,
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        # The held-out set contains assigned positives, not true negatives. These are the
        # observable FP/FN rates for its predicted and gold positive populations.
        "false_positive_rate": evaluation.false_positives / positives if positives else 0.0,
        "false_negative_rate": evaluation.false_negatives / gold if gold else 0.0,
    }


def _report_metadata(report: FacetFitReport | None) -> dict | None:
    if report is None:
        return None
    return {
        "n_bootstrap": report.n_bootstrap,
        "n_after_gate": report.n_after_gate,
        "n_excluded": report.n_excluded,
        "tier1_eligible": report.tier1_eligible,
        "skipped": report.skipped,
        "gold_evaluation": _gold_metrics(report),
    }


def _manifest_metadata(classifier_path: str | None) -> dict | None:
    if not classifier_path:
        return None
    manifest_path = Path(classifier_path) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def start_or_resume_pass(
    session: Session,
    *,
    reports: dict[str, FacetFitReport] | None,
    tau_class: float,
    classifier_path: str | None,
) -> ClassifierMetricSnapshot:
    """Create one durable pass, or return the interrupted pass for exact continuation."""
    existing = session.scalars(
        select(ClassifierMetricSnapshot)
        .where(ClassifierMetricSnapshot.status == "in_progress")
        .order_by(ClassifierMetricSnapshot.started_at, ClassifierMetricSnapshot.id)
    ).first()
    if existing is not None:
        prior = existing.effective_config_json
        if prior.get("tau_class") != tau_class or prior.get("classifier_path") != classifier_path:
            raise ValueError(
                "cannot resume classification pass with changed tau_class or classifier_path; "
                "restore the captured classifier configuration first"
            )
        return existing

    reports = reports or {}
    active_facets = set(
        session.scalars(select(SkosConcept.facet).where(SkosConcept.status == "active"))
    )
    facets = sorted(active_facets | set(reports))
    high_water = session.scalar(
        select(func.max(FeedItem.id)).where(
            FeedItem.status == "active",
            FeedItem.facets_classified_at.is_(None),
        )
    )
    config_facets = {
        facet: {"tau_class": tau_class, "manifest": _report_metadata(reports.get(facet))}
        for facet in facets
    }
    snapshot = ClassifierMetricSnapshot(
        classification_pass_id=str(uuid.uuid4()),
        status="in_progress",
        eligibility_high_water_id=high_water or 0,
        processed_item_cursor=0,
        accumulator_json={"facets": {facet: _zero_facet_metrics() for facet in facets}},
        effective_config_json={
            "tau_class": tau_class,
            "classifier_path": classifier_path,
            "bundle_manifest": _manifest_metadata(classifier_path),
            "facets": config_facets,
        },
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def _bin(score: float) -> str:
    if score < 0.2:
        return _BINS[0]
    if score < 0.4:
        return _BINS[1]
    if score < 0.6:
        return _BINS[2]
    if score < 0.8:
        return _BINS[3]
    return _BINS[4]


def accumulate_decisions(
    snapshot: ClassifierMetricSnapshot, decisions: list[FacetDecision], *, cursor: int
) -> None:
    """Merge one uncommitted D-14 batch's decisions into its snapshot payload."""
    accumulator = copy.deepcopy(snapshot.accumulator_json)
    facets = accumulator.setdefault("facets", {})
    for decision in decisions:
        facet = facets.setdefault(decision.facet, _zero_facet_metrics())
        facet["eligible_items"] += 1
        facet["eligible_facet_decisions"] += 1
        if decision.tier0_resolved:
            facet["tier0_resolved"] += 1
        for score in decision.tier1_scores:
            facet["tier1_evaluated"] += 1
            facet["tier1_confidence_bins"][_bin(score)] += 1
        if decision.tier1_accepted:
            facet["tier1_accepted"] += 1
        if decision.tier2_attempted:
            facet["tier2_attempted"] += 1
            facet["tier2_attempted_item_count"] += 1
        if decision.tier2_resolved:
            facet["tier2_resolved"] += 1
        if decision.unclassified:
            facet["unclassified"] += 1
    snapshot.accumulator_json = accumulator
    snapshot.processed_item_cursor = cursor


def _review_metrics(
    session: Session, *, facet: str, started_at: datetime, completed_at: datetime
) -> dict:
    events = session.scalars(
        select(ClassificationCorrection).where(
            ClassificationCorrection.facet == facet,
            ClassificationCorrection.status.in_(("accepted", "superseded")),
            ClassificationCorrection.decided_at.is_not(None),
            ClassificationCorrection.decided_at >= started_at,
            ClassificationCorrection.decided_at <= completed_at,
        )
    ).all()
    valid: list[ClassificationCorrection] = []
    errors: list[str] = []
    for event in events:
        concept_ids = set(event.prior_concept_ids) | set(event.corrected_concept_ids)
        concepts = {
            concept.id: concept.facet
            for concept in session.scalars(
                select(SkosConcept).where(SkosConcept.id.in_(concept_ids))
            )
        }
        invalid = sorted(
            concept_id for concept_id in concept_ids if concepts.get(concept_id) != facet
        )
        if invalid:
            errors.append(f"correction {event.id} has concepts outside facet {facet}: {invalid}")
            continue
        valid.append(event)
    approvals = sum(event.action == "approval" for event in valid)
    corrections = [event for event in valid if event.action == "correction"]
    denominator = approvals + len(corrections)
    false_positives = sum(
        len(set(event.prior_concept_ids) - set(event.corrected_concept_ids))
        for event in corrections
    )
    false_negatives = sum(
        len(set(event.corrected_concept_ids) - set(event.prior_concept_ids))
        for event in corrections
    )
    return {
        "sample_type": "human-reviewed sample (not an unbiased gold-set metric)",
        "reviewed_events": denominator,
        "reviewed_approvals": approvals,
        "reviewed_corrections": len(corrections),
        "reviewed_correction_rate": len(corrections) / denominator if denominator else None,
        "reviewed_false_positives": false_positives,
        "reviewed_false_negatives": false_negatives,
        "validation_errors": errors,
    }


def finalize_pass(session: Session, snapshot: ClassifierMetricSnapshot) -> None:
    """Attach computed rates/review evidence and publish a completed snapshot."""
    now = _utcnow()
    accumulator = copy.deepcopy(snapshot.accumulator_json)
    for facet, values in accumulator.get("facets", {}).items():
        eligible_items = values["eligible_items"]
        eligible_decisions = values["eligible_facet_decisions"]
        values["item_escalation_numerator"] = values["tier2_attempted_item_count"]
        values["item_escalation_denominator"] = eligible_items
        values["item_escalation_rate"] = (
            values["tier2_attempted_item_count"] / eligible_items if eligible_items else 0.0
        )
        values["facet_decision_escalation_numerator"] = values["tier2_attempted"]
        values["facet_decision_escalation_denominator"] = eligible_decisions
        values["facet_decision_escalation_rate"] = (
            values["tier2_attempted"] / eligible_decisions if eligible_decisions else 0.0
        )
        values["reviewed"] = _review_metrics(
            session, facet=facet, started_at=snapshot.started_at, completed_at=now
        )
    snapshot.accumulator_json = accumulator
    snapshot.status = "complete"
    snapshot.completed_at = now
    snapshot.error_message = None


def fail_pass(session: Session, snapshot: ClassifierMetricSnapshot, exc: Exception) -> None:
    """Persist an interrupted pass for retry without publishing its partial telemetry."""
    snapshot.error_message = "".join(traceback.format_exception(exc))
    session.add(snapshot)
    session.commit()


def snapshot_payload(snapshot: ClassifierMetricSnapshot, *, threshold: float | None = None) -> dict:
    """Return the stable CLI representation, including the completed-snapshot alarm state."""
    payload = {
        "classification_pass_id": snapshot.classification_pass_id,
        "status": snapshot.status,
        "started_at": snapshot.started_at.isoformat(),
        "completed_at": snapshot.completed_at.isoformat() if snapshot.completed_at else None,
        "eligibility_high_water_id": snapshot.eligibility_high_water_id,
        "processed_item_cursor": snapshot.processed_item_cursor,
        "effective_config": snapshot.effective_config_json,
        "facets": snapshot.accumulator_json.get("facets", {}),
        "error_message": snapshot.error_message,
        "alarms": [],
    }
    if snapshot.status == "complete" and threshold is not None:
        for facet, metrics in payload["facets"].items():
            rate = metrics.get("facet_decision_escalation_rate", 0.0)
            if rate > threshold:
                payload["alarms"].append(
                    {
                        "type": "classifier_escalation_rate",
                        "facet": facet,
                        "rate": rate,
                        "threshold": threshold,
                    }
                )
    return payload


def snapshots_in_range(
    session: Session,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    threshold: float | None = None,
) -> list[dict]:
    statement = select(ClassifierMetricSnapshot).where(
        ClassifierMetricSnapshot.status == "complete"
    )
    if since is not None:
        statement = statement.where(ClassifierMetricSnapshot.started_at >= since)
    if until is not None:
        statement = statement.where(ClassifierMetricSnapshot.started_at <= until)
    snapshots = session.scalars(
        statement.order_by(ClassifierMetricSnapshot.started_at, ClassifierMetricSnapshot.id)
    ).all()
    return [snapshot_payload(snapshot, threshold=threshold) for snapshot in snapshots]
