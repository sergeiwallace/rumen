"""T-20 persisted per-facet classification telemetry."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from rumen.classifier.cascade import ClassificationError
from rumen.classifier.metrics import (
    fail_pass,
    finalize_pass,
    start_or_resume_pass,
)
from rumen.jobs import run_pipeline
from rumen.models import (
    ClassificationCorrection,
    ClassifierMetricSnapshot,
    FeedItem,
    SkosConcept,
)
from rumen.taxonomy.hmc.fit import FacetFitReport
from rumen.taxonomy.hmc.gold_eval import GoldEvaluation

_NOW = datetime(2026, 7, 20, tzinfo=UTC)


class _Hmc:
    def __init__(self, concept_id: int, score: float) -> None:
        self.concept_id = concept_id
        self.score = score

    def raw_scores(self, _text: str) -> dict[int, float]:
        return {self.concept_id: self.score}

    def assign(self, _text: str, *, path_consistency_floor: float) -> dict[int, float]:
        return {self.concept_id: self.score} if self.score >= path_consistency_floor else {}

    def assign_scores(
        self, _scores: dict[int, float], *, path_consistency_floor: float
    ) -> dict[int, float]:
        return {self.concept_id: self.score} if self.score >= path_consistency_floor else {}


class _Llm:
    def __init__(self, concept_id: int) -> None:
        self.concept_id = concept_id

    def run(self, _prompt: str) -> str:
        return f'{{"concept_ids": [{self.concept_id}], "confidence": 0.9}}'


def _concept(session, facet: str, label: str) -> SkosConcept:
    concept = SkosConcept(facet=facet, pref_label=label, scheme="test")
    session.add(concept)
    session.flush()
    return concept


def _item(session, source_id: str) -> FeedItem:
    item = FeedItem(
        source="test",
        source_id=source_id,
        url=f"https://example.test/{source_id}",
        title=f"title {source_id}",
        content=f"content {source_id}",
        published_at=_NOW,
        stable_id_kind="test",
        stable_id_value=source_id,
    )
    session.add(item)
    session.flush()
    return item


def _report(concept_id: int, score: float, *, evaluated: bool = True) -> FacetFitReport:
    gold = GoldEvaluation(
        "topic",
        0.8 if evaluated else None,
        0.5 if evaluated else None,
        4 if evaluated else 0,
        1 if evaluated else 0,
        4 if evaluated else 0,
        3 if evaluated else 0,
        evaluated,
    )
    return FacetFitReport("topic", _Hmc(concept_id, score), 8, 7, 1, gold, True, False)  # type: ignore[arg-type]


def test_given_hmc_and_tier2_decisions_when_pipeline_completes_then_persists_facet_metrics(
    session, monkeypatch
):
    topic = _concept(session, "topic", "tool")
    _item(session, "one")
    session.commit()
    monkeypatch.setattr("rumen.jobs.group_pending", lambda *_args, **_kwargs: {})

    result = run_pipeline(
        session,
        adapters=[],
        reports={"topic": _report(topic.id, 0.2)},
        llm_command=_Llm(topic.id),
        tau_class=0.75,
        classifier_path="fixtures/hmc-bundle",
        classifier_escalation_alarm_threshold=0.5,
        now=_NOW,
    )

    snapshot = session.query(ClassifierMetricSnapshot).one()
    facet = snapshot.accumulator_json["facets"]["topic"]
    assert result.ok is True
    assert snapshot.status == "complete"
    assert snapshot.processed_item_cursor == snapshot.eligibility_high_water_id
    assert facet["eligible_items"] == facet["eligible_facet_decisions"] == 1
    assert facet["tier1_evaluated"] == 1
    assert sum(facet["tier1_confidence_bins"].values()) == facet["tier1_evaluated"]
    assert facet["tier2_attempted"] == facet["tier2_resolved"] == 1
    assert facet["facet_decision_escalation_rate"] == 1.0
    assert facet["item_escalation_numerator"] == facet["item_escalation_denominator"] == 1
    assert facet["reviewed"]["reviewed_correction_rate"] is None
    assert facet["reviewed"]["sample_type"].startswith("human-reviewed sample")
    assert snapshot.effective_config_json["classifier_path"] == "fixtures/hmc-bundle"
    assert (
        snapshot.effective_config_json["facets"]["topic"]["manifest"]["gold_evaluation"][
            "false_negative_rate"
        ]
        == 0.5
    )
    assert result.classifier_alarms[0]["rate"] == 1.0


def test_given_source_categorized_bookmark_when_starting_pass_then_its_id_is_within_high_water(
    session,
):
    item = _item(session, "bookmark")
    item.source = "bookmarks"
    item.category = "source-curated"
    session.commit()

    snapshot = start_or_resume_pass(session, reports=None, tau_class=0.75, classifier_path=None)

    assert item.id <= snapshot.eligibility_high_water_id


def test_given_chained_classification_error_when_failing_pass_then_persists_underlying_cause(
    session,
):
    snapshot = start_or_resume_pass(session, reports=None, tau_class=0.75, classifier_path=None)
    try:
        try:
            raise RuntimeError("underlying database write cause")
        except RuntimeError as cause:
            raise ClassificationError(
                "classification batch rolled back after write failure"
            ) from cause
    except ClassificationError as error:
        fail_pass(session, snapshot, error)

    persisted = session.get(ClassifierMetricSnapshot, snapshot.id)
    assert persisted is not None
    assert "RuntimeError: underlying database write cause" in persisted.error_message
    assert "ClassificationError: classification batch rolled back after write failure" in (
        persisted.error_message
    )


def test_given_committed_batch_failure_when_retrying_then_resumes_same_snapshot_once(
    session, monkeypatch
):
    topic = _concept(session, "topic", "tool")
    first = _item(session, "one")
    second = _item(session, "two")
    session.commit()
    monkeypatch.setattr("rumen.jobs.group_pending", lambda *_args, **_kwargs: {})
    report = {"topic": _report(topic.id, 1.0)}
    from rumen.classifier import cascade

    original = cascade.classify_item
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("telemetry processing failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(cascade, "classify_item", fail_after_first)
    failed = run_pipeline(
        session,
        adapters=[],
        reports=report,
        commit_batch_size=1,
        classify_write_max_attempts=1,
        now=_NOW,
    )
    interrupted = session.query(ClassifierMetricSnapshot).one()
    assert failed.ok is False
    assert interrupted.status == "in_progress"
    assert interrupted.processed_item_cursor == first.id
    assert session.get(FeedItem, second.id).category is None

    monkeypatch.setattr(cascade, "classify_item", original)
    resumed = run_pipeline(session, adapters=[], reports=report, commit_batch_size=1, now=_NOW)
    snapshots = session.query(ClassifierMetricSnapshot).all()
    facet = snapshots[0].accumulator_json["facets"]["topic"]
    assert resumed.ok is True
    assert len(snapshots) == 1
    assert snapshots[0].status == "complete"
    assert snapshots[0].classification_pass_id == failed.classification_pass_id
    assert facet["eligible_items"] == facet["tier1_accepted"] == 2
    assert session.get(FeedItem, second.id).category == "tool"


def test_given_reviewed_and_invalid_events_when_finalizing_then_reports_only_valid_sample(session):
    topic = _concept(session, "topic", "tool")
    other = _concept(session, "content", "article")
    item = _item(session, "one")
    snapshot = start_or_resume_pass(
        session, reports={"topic": _report(topic.id, 1.0)}, tau_class=0.75, classifier_path=None
    )
    session.add_all(
        [
            ClassificationCorrection(
                item_table="feed_items",
                item_id=item.id,
                facet="topic",
                action="approval",
                prior_concept_ids=[topic.id],
                corrected_concept_ids=[topic.id],
                actor="reviewer",
                source="test",
                rationale="same",
                status="accepted",
                decided_at=datetime.now(UTC),
            ),
            ClassificationCorrection(
                item_table="feed_items",
                item_id=item.id,
                facet="topic",
                action="correction",
                prior_concept_ids=[other.id],
                corrected_concept_ids=[topic.id],
                actor="reviewer",
                source="test",
                rationale="invalid facet",
                status="accepted",
                decided_at=datetime.now(UTC),
            ),
        ]
    )
    session.flush()

    finalize_pass(session, snapshot)
    reviewed = snapshot.accumulator_json["facets"]["topic"]["reviewed"]
    assert reviewed["reviewed_events"] == 1
    assert reviewed["reviewed_correction_rate"] == 0.0
    assert reviewed["validation_errors"]


def test_given_completed_snapshot_when_cli_metrics_runs_then_returns_trend_and_alarm(
    session, tmp_path, capsys
):
    topic = _concept(session, "topic", "tool")
    snapshot = start_or_resume_pass(
        session, reports={"topic": _report(topic.id, 1.0)}, tau_class=0.75, classifier_path=None
    )
    accumulator = json.loads(json.dumps(snapshot.accumulator_json))
    accumulator["facets"]["topic"].update(
        {
            "eligible_items": 1,
            "eligible_facet_decisions": 1,
            "tier2_attempted": 1,
            "tier2_attempted_item_count": 1,
        }
    )
    snapshot.accumulator_json = accumulator
    finalize_pass(session, snapshot)
    session.commit()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sources": [], "classifier_escalation_alarm_threshold": 0.5}))

    from rumen.cli import run

    assert run(["classify", "metrics", "--config", str(config)], session) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["classification_pass_id"] for row in payload] == [snapshot.classification_pass_id]
    assert payload[0]["facets"]["topic"]["tier2_attempted"] == 1
    assert payload[0]["alarms"][0]["threshold"] == 0.5


@pytest.mark.parametrize("value", [None, -0.01, 1.01])
def test_given_invalid_alarm_threshold_when_loading_config_then_rejects(tmp_path, value):
    from rumen.config import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"sources": [], "classifier_escalation_alarm_threshold": value}))
    with pytest.raises(ValueError, match="classifier_escalation_alarm_threshold"):
        load_config(path)
