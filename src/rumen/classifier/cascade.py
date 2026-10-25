"""Live per-facet Tier-0 → HMC → Tier-2 classification cascade."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rumen.classifier.tier0_rules import (
    Tier0Input,
    classify_tier0,
    tier0_expected_facets,
)
from rumen.classifier.tier1_embedding import DEFAULT_TAU_CLASS
from rumen.classifier.tier2_llm import classify_tier2
from rumen.db.engine import batched_write_mode
from rumen.dropped_items import log_drop
from rumen.llm import BudgetedLlmCommand, LlmCommand
from rumen.models import (
    ClassifierMetricSnapshot,
    FeedItem,
    ItemConcept,
    SkosConcept,
    SkosRelation,
)

if TYPE_CHECKING:
    # Annotation only: importing it at runtime closes a cycle through rumen.classifier,
    # so importing rumen.taxonomy.hmc.fit first would fail.
    from rumen.taxonomy.hmc.fit import FacetFitReport

logger = logging.getLogger(__name__)


class ClassificationError(RuntimeError):
    """A mutation failed, so the currently uncommitted D-14 batch was rolled back."""


@dataclass(slots=True)
class FacetAssignment:
    facet: str
    concept_confidences: dict[int, float]
    method: str


@dataclass(slots=True)
class FacetDecision:
    """One eligible per-facet cascade outcome, retained only for batch telemetry."""

    facet: str
    tier0_resolved: bool = False
    tier1_scores: tuple[float, ...] = ()
    tier1_accepted: bool = False
    tier2_attempted: bool = False
    tier2_resolved: bool = False
    tier2_budget_refused: bool = False
    unclassified: bool = False


@dataclass(slots=True)
class ClassificationResult:
    assignments: list[FacetAssignment]
    decisions: list[FacetDecision]
    category: str | None
    confidence: float | None
    method: str | None


def _text_of(item: FeedItem) -> str:
    return f"{item.title or ''}\n{item.content or ''}".strip()


def _active_concepts(session: Session) -> dict[str, list[SkosConcept]]:
    result: dict[str, list[SkosConcept]] = {}
    for concept in session.scalars(select(SkosConcept).where(SkosConcept.status == "active")):
        result.setdefault(concept.facet, []).append(concept)
    return result


def _parents(session: Session, facet: str, concepts: list[SkosConcept]) -> dict[int, set[int]]:
    ids = {concept.id for concept in concepts}
    result: dict[int, set[int]] = {}
    if not ids:
        return result
    for relation in session.scalars(
        select(SkosRelation).where(SkosRelation.relation_type.in_(("broader", "narrower")))
    ):
        if relation.source_concept_id not in ids and relation.target_concept_id not in ids:
            continue
        child, parent = (
            (relation.source_concept_id, relation.target_concept_id)
            if relation.relation_type == "broader"
            else (relation.target_concept_id, relation.source_concept_id)
        )
        if child not in ids:
            continue
        if parent not in ids:
            # A hierarchy edge to inactive/wrong-facet data means this target cannot have a
            # complete active parent path; make closure validation reject it.
            result.setdefault(child, set()).add(-1)
            continue
        result.setdefault(child, set()).add(parent)
    return result


def _closure(
    target_ids: set[int], parents: dict[int, set[int]], valid_ids: set[int]
) -> set[int] | None:
    """Return complete active parent closure; cycles or absent paths are invalid."""
    result: set[int] = set()
    visiting: set[int] = set()

    def visit(concept_id: int) -> bool:
        if concept_id not in valid_ids or concept_id in visiting:
            return False
        if concept_id in result:
            return True
        visiting.add(concept_id)
        if not all(visit(parent) for parent in parents.get(concept_id, set())):
            return False
        visiting.remove(concept_id)
        result.add(concept_id)
        return True

    return result if all(visit(target) for target in target_ids) else None


def _has_human_facet(session: Session, item: FeedItem, facet: str) -> bool:
    return (
        session.scalar(
            select(ItemConcept.id)
            .where(
                ItemConcept.item_table == "feed_items",
                ItemConcept.item_id == item.id,
                ItemConcept.facet == facet,
                ItemConcept.method.startswith("human:"),
            )
            .limit(1)
        )
        is not None
    )


def _topic_root(concept_confidences: dict[int, float], parents: dict[int, set[int]]) -> int | None:
    """Select a closure root by confidence, then stable numeric id."""
    roots = [concept_id for concept_id in concept_confidences if not parents.get(concept_id)]
    if not roots:
        return None
    return min(roots, key=lambda concept_id: (-concept_confidences[concept_id], concept_id))


def _topic_scalar(
    assignments: list[FacetAssignment],
    concepts: dict[int, SkosConcept],
    parents: dict[int, set[int]],
) -> tuple[str | None, float | None, str | None]:
    topic = next((assignment for assignment in assignments if assignment.facet == "topic"), None)
    if topic is None:
        return None, None, None
    root = _topic_root(topic.concept_confidences, parents)
    if root is None:
        return None, None, None
    return concepts[root].pref_label, topic.concept_confidences[root], topic.method


def classify_item(
    session: Session,
    item: FeedItem,
    *,
    reports: dict[str, FacetFitReport] | None = None,
    llm_command: LlmCommand | None = None,
    tau_class: float = DEFAULT_TAU_CLASS,
) -> ClassificationResult:
    """Evaluate each active facet independently. This function does not mutate the item."""
    reports = reports or {}
    by_facet = _active_concepts(session)
    tier0_input = Tier0Input(item.source, item.url, item.stable_id_kind, item.content)
    symbolic = classify_tier0(tier0_input)
    expected_tier0_facets = tier0_expected_facets(tier0_input)
    accepted: list[FacetAssignment] = []
    decisions: list[FacetDecision] = []
    all_concepts = {concept.id: concept for concepts in by_facet.values() for concept in concepts}
    topic_parents: dict[int, set[int]] = {}
    for facet in sorted(set(by_facet) | set(reports) | set(symbolic)):
        concepts = by_facet.get(facet, [])
        if not concepts or _has_human_facet(session, item, facet):
            continue
        concept_by_label = {concept.pref_label.lower(): concept for concept in concepts}
        parents = _parents(session, facet, concepts)
        if facet == "topic":
            topic_parents = parents
        targets = symbolic.get(facet, ())
        resolved = [
            concept_by_label[target.lower()]
            for target in targets
            if target.lower() in concept_by_label
        ]
        if targets and len(resolved) == len(targets):
            closure = _closure(
                {concept.id for concept in resolved}, parents, {concept.id for concept in concepts}
            )
            if closure is not None:
                accepted.append(
                    FacetAssignment(facet, {concept_id: 1.0 for concept_id in closure}, "rule")
                )
                decisions.append(FacetDecision(facet, tier0_resolved=True))
                continue
        report = reports.get(facet)
        # Compute raw_scores() once and reuse it for both the accept/reject gate and telemetry —
        # assign_for_tier1()/assign() would otherwise recompute it, doubling the embedding call.
        scores_map = (
            report.classifier.raw_scores(_text_of(item))
            if report is not None and report.tier1_eligible and report.classifier is not None
            else None
        )
        hmc = (
            report.classifier.assign_scores(scores_map, path_consistency_floor=tau_class)
            if scores_map is not None
            else {}
        )
        scores = tuple(scores_map.values()) if scores_map is not None else ()
        if hmc:
            closure = _closure(set(hmc), parents, {concept.id for concept in concepts})
            if closure is not None and closure.issubset(hmc):
                accepted.append(FacetAssignment(facet, hmc, "hmc"))
                decisions.append(FacetDecision(facet, tier1_scores=scores, tier1_accepted=True))
                continue
        if llm_command is None:
            decisions.append(FacetDecision(facet, tier1_scores=scores, unclassified=True))
            continue
        expected_tier2 = facet not in expected_tier0_facets
        budget_was_unavailable = isinstance(llm_command, BudgetedLlmCommand) and not (
            llm_command.tier2_available(expected=expected_tier2)
        )
        if budget_was_unavailable:
            decisions.append(
                FacetDecision(
                    facet,
                    tier1_scores=scores,
                    tier2_budget_refused=True,
                    unclassified=True,
                )
            )
            continue
        judged = classify_tier2(
            item.title or "",
            item.content,
            facet=facet,
            concepts=concepts,
            llm_command=(
                llm_command.for_tier2(expected=expected_tier2)
                if isinstance(llm_command, BudgetedLlmCommand)
                else llm_command
            ),
        )
        if judged:
            closure = _closure(set(judged), parents, {concept.id for concept in concepts})
            if closure is not None:
                confidence = min(judged.values())
                accepted.append(
                    FacetAssignment(
                        facet,
                        {concept_id: judged.get(concept_id, confidence) for concept_id in closure},
                        "llm",
                    )
                )
                decisions.append(
                    FacetDecision(
                        facet,
                        tier1_scores=scores,
                        tier2_attempted=True,
                        tier2_resolved=True,
                    )
                )
                continue
        decisions.append(
            FacetDecision(
                facet,
                tier1_scores=scores,
                tier2_attempted=True,
                tier2_budget_refused=budget_was_unavailable,
                unclassified=True,
            )
        )
    category, confidence, method = _topic_scalar(accepted, all_concepts, topic_parents)
    return ClassificationResult(accepted, decisions, category, confidence, method)


def _replace_facet(session: Session, item: FeedItem, assignment: FacetAssignment) -> None:
    if assignment.method.startswith(("bootstrap:", "human:")):
        raise ClassificationError("live HMC method must not use bootstrap: or human: provenance")
    if assignment.method not in {"rule", "hmc", "llm"}:
        raise ClassificationError(f"invalid live classification method {assignment.method!r}")
    if _has_human_facet(session, item, assignment.facet):
        return
    session.execute(
        delete(ItemConcept).where(
            ItemConcept.item_table == "feed_items",
            ItemConcept.item_id == item.id,
            ItemConcept.facet == assignment.facet,
            ~ItemConcept.method.startswith("human:"),
        )
    )
    session.add_all(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=concept_id,
            facet=assignment.facet,
            confidence=confidence,
            method=assignment.method,
        )
        for concept_id, confidence in assignment.concept_confidences.items()
    )


def classify_pending(
    session: Session,
    *,
    reports: dict[str, FacetFitReport] | None = None,
    llm_command: LlmCommand | None = None,
    tau_class: float = DEFAULT_TAU_CLASS,
    out_of_scope: frozenset[str] = frozenset(),
    commit_batch_size: int = 500,
    classify_write_max_attempts: int = 3,
    classify_write_retry_backoff_seconds: float = 1.0,
    metric_snapshot: ClassifierMetricSnapshot | None = None,
) -> dict[str, int]:
    """Classify eligible rows, atomically persisting each row within D-14 commit batches."""
    eligibility = [
        FeedItem.status == "active",
        FeedItem.facets_classified_at.is_(None),
    ]
    if metric_snapshot is not None:
        eligibility.extend(
            (
                FeedItem.id > metric_snapshot.processed_item_cursor,
                FeedItem.id <= metric_snapshot.eligibility_high_water_id,
            )
        )
    rows = session.scalars(
        select(FeedItem)
        .where(
            *eligibility,
        )
        .order_by(FeedItem.id)
    ).all()
    counts = {"rule": 0, "embedding": 0, "llm": 0, "unclassified": 0, "dropped": 0}
    since_commit = 0
    batch_decisions: list[FacetDecision] = []
    batch_cursor = metric_snapshot.processed_item_cursor if metric_snapshot is not None else 0
    with batched_write_mode(session):
        for row in rows:
            row_counts = counts.copy()
            for attempt_index in range(classify_write_max_attempts):
                try:
                    with session.begin_nested():
                        source_category = row.category not in (None, "unclassified")
                        result = classify_item(
                            session,
                            row,
                            reports=reports,
                            llm_command=llm_command,
                            tau_class=tau_class,
                        )
                        # A retry replaces the complete non-human projection, including facets that
                        # remain unresolved this time; human facets are authoritative and untouched.
                        session.execute(
                            delete(ItemConcept).where(
                                ItemConcept.item_table == "feed_items",
                                ItemConcept.item_id == row.id,
                                ~ItemConcept.method.startswith("human:"),
                            )
                        )
                        for assignment in result.assignments:
                            _replace_facet(session, row, assignment)
                        if source_category:
                            row.classification_confidence, row.classification_method = (
                                None,
                                "source",
                            )
                        elif result.category is None:
                            (
                                row.category,
                                row.classification_confidence,
                                row.classification_method,
                            ) = (
                                "unclassified",
                                None,
                                None,
                            )
                            counts["unclassified"] += 1
                        else:
                            (
                                row.category,
                                row.classification_confidence,
                                row.classification_method,
                            ) = (
                                result.category,
                                result.confidence,
                                result.method,
                            )
                            counts["embedding" if result.method == "hmc" else result.method] += 1
                            if result.category in out_of_scope:
                                log_drop(
                                    session,
                                    source=row.source,
                                    source_id=row.source_id,
                                    url=row.url,
                                    title=row.title,
                                    reason="classified_out_of_scope",
                                    drop_confidence=result.confidence or 0.0,
                                )
                                row.status = "dropped"
                                counts["dropped"] += 1
                        if not any(decision.tier2_budget_refused for decision in result.decisions):
                            row.facets_classified_at = datetime.now(UTC)
                        session.flush()
                    break
                except Exception as exc:
                    counts = row_counts.copy()
                    attempts = attempt_index + 1
                    if attempts == classify_write_max_attempts:
                        session.rollback()
                        raise ClassificationError(
                            "classification batch rolled back after write failure "
                            f"after {attempts} attempts"
                        ) from exc
                    backoff_seconds = classify_write_retry_backoff_seconds * (2**attempt_index)
                    logger.warning(
                        "classification write failed for item %s on attempt %s/%s; retrying in %s seconds: %s",
                        row.id,
                        attempts,
                        classify_write_max_attempts,
                        backoff_seconds,
                        exc,
                    )
                    time.sleep(backoff_seconds)
            since_commit += 1
            batch_decisions.extend(result.decisions)
            batch_cursor = row.id
            if since_commit >= commit_batch_size:
                if metric_snapshot is not None:
                    from rumen.classifier.metrics import accumulate_decisions

                    accumulate_decisions(metric_snapshot, batch_decisions, cursor=batch_cursor)
                    session.flush()
                session.commit()
                since_commit = 0
                batch_decisions = []
        if since_commit:
            if metric_snapshot is not None:
                from rumen.classifier.metrics import accumulate_decisions

                accumulate_decisions(metric_snapshot, batch_decisions, cursor=batch_cursor)
                session.flush()
            session.commit()
        elif not rows:
            session.commit()
    return counts
