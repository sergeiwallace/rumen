"""T-25 live per-facet cascade regressions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from rumen.classifier.cascade import (
    ClassificationError,
    FacetAssignment,
    _parents,
    _topic_root,
    _topic_scalar,
    classify_item,
    classify_pending,
)
from rumen.classifier.tier0_rules import Tier0Input, tier0_expected_facets
from rumen.classifier.tier1_embedding import FacetHMCClassifier
from rumen.embeddings import HashingEmbeddingProvider
from rumen.llm import BudgetedLlmCommand, LlmUnavailable
from rumen.models import FeedItem, ItemConcept, SkosConcept, SkosRelation
from rumen.taxonomy.hmc.fit import FacetFitReport
from rumen.taxonomy.hmc.gold_eval import GoldEvaluation

_NOW = datetime(2026, 7, 20, tzinfo=UTC)


def _item(session, source="reddit", source_id="one"):
    item = FeedItem(
        source=source,
        source_id=source_id,
        url=f"https://example/{source_id}",
        title="release tool",
        published_at=_NOW,
    )
    session.add(item)
    session.flush()
    return item


def _concept(session, facet, label, concept_id=None):
    concept = (
        SkosConcept(id=concept_id, facet=facet, pref_label=label, scheme="test")
        if concept_id
        else SkosConcept(facet=facet, pref_label=label, scheme="test")
    )
    session.add(concept)
    session.flush()
    return concept


def _report(facet, concepts):
    classifier = FacetHMCClassifier(facet, HashingEmbeddingProvider()).fit(
        [("release tool", concept.id) for concept in concepts]
    )
    return FacetFitReport(
        facet, classifier, 2, 2, 0, GoldEvaluation(facet, 1.0, 1.0, 1, 0, 0, 1, True), True, False
    )


def test_given_tier0_signal_when_classify_pending_then_it_writes_resolved_facets_and_derived_category(
    session,
):
    tool = _concept(session, "topic", "tool")
    release = _concept(session, "content-type", "tool-release")
    source = _concept(session, "source", "github-releases")
    item = _item(session, source="github_releases")
    session.commit()

    counts = classify_pending(session)
    labels = session.query(ItemConcept).filter_by(item_table="feed_items", item_id=item.id).all()
    assert counts["rule"] == 1
    assert {(label.facet, label.concept_id, label.method) for label in labels} == {
        ("topic", tool.id, "rule"),
        ("content-type", release.id, "rule"),
        ("source", source.id, "rule"),
    }
    row = session.get(FeedItem, item.id)
    assert (row.category, row.classification_method, row.classification_confidence) == (
        "tool",
        "rule",
        1.0,
    )


def test_given_hmc_topic_when_classify_pending_then_it_writes_hmc_and_derives_category(
    session, monkeypatch
):
    root = _concept(session, "topic", "tool", 10)
    child = _concept(session, "topic", "cli", 11)
    session.add(
        SkosRelation(source_concept_id=child.id, target_concept_id=root.id, relation_type="broader")
    )
    item = _item(session)
    session.commit()
    report = _report("topic", [root, child])
    from rumen.classifier.tier1_embedding import EmbeddingClassifier

    monkeypatch.setattr(
        EmbeddingClassifier,
        "predict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retired flat model invoked")),
    )

    counts = classify_pending(session, reports={"topic": report}, tau_class=0.0)
    labels = (
        session.query(ItemConcept)
        .filter_by(item_table="feed_items", item_id=item.id, facet="topic")
        .all()
    )
    assert counts["embedding"] == 1
    assert {label.method for label in labels} == {"hmc"}
    assert session.get(FeedItem, item.id).classification_method == "hmc"


def test_given_one_low_confidence_facet_when_classify_pending_then_only_that_facet_reaches_tier2(
    session,
):
    topic = _concept(session, "topic", "tool")
    content = _concept(session, "content-type", "tool-release")
    item = _item(session)
    session.commit()
    llm = type(
        "Llm",
        (),
        {
            "calls": [],
            "run": lambda self, prompt: (
                self.calls.append(prompt) or f'{{"concept_ids": [{content.id}], "confidence": 0.9}}'
            ),
        },
    )()
    report = _report("topic", [topic])

    classify_pending(session, reports={"topic": report}, llm_command=llm, tau_class=0.0)
    assert len(llm.calls) == 1 and "content-type" in llm.calls[0]
    assert session.get(FeedItem, item.id).category == "tool"
    assert (
        session.query(ItemConcept)
        .filter_by(item_id=item.id, concept_id=content.id, method="llm")
        .one()
    )


def test_given_human_facet_when_live_classification_runs_then_it_is_unchanged(session):
    tool = _concept(session, "topic", "tool")
    human = _concept(session, "topic", "human-topic")
    item = _item(session, source="github_releases")
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=human.id,
            facet="topic",
            confidence=1.0,
            method="human:review",
        )
    )
    session.commit()

    classify_pending(session)
    labels = session.query(ItemConcept).filter_by(item_id=item.id, facet="topic").all()
    assert [(label.concept_id, label.method) for label in labels] == [(human.id, "human:review")]
    assert tool.id != human.id


def test_given_equivalent_live_and_human_topic_roots_when_derived_then_they_select_the_same_category(
    session,
) -> None:
    root = _concept(session, "topic", "Root", 10)
    child = _concept(session, "topic", "Child", 11)
    session.add(
        SkosRelation(source_concept_id=child.id, target_concept_id=root.id, relation_type="broader")
    )
    session.flush()
    parents = _parents(session, "topic", [root, child])

    live_category, _, _ = _topic_scalar(
        [FacetAssignment("topic", {root.id: 1.0, child.id: 1.0}, "rule")],
        {root.id: root, child.id: child},
        parents,
    )
    human_root = _topic_root({root.id: 1.0, child.id: 1.0}, parents)

    assert live_category == root.pref_label
    assert (
        human_root is not None
        and {root.id: root, child.id: child}[human_root].pref_label == live_category
    )


def test_given_live_assignment_write_failure_when_classify_pending_then_current_batch_is_atomic(
    session, monkeypatch
):
    _concept(session, "topic", "tool")
    first = _item(session, source="github_releases", source_id="first")
    _item(session, source="github_releases", source_id="second")
    session.commit()
    from rumen.classifier import cascade

    original = cascade._replace_facet
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("write failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(cascade, "_replace_facet", fail_second)
    with pytest.raises(ClassificationError):
        classify_pending(session, commit_batch_size=50, classify_write_max_attempts=1)
    assert session.get(FeedItem, first.id).category is None
    assert session.query(ItemConcept).count() == 0


def test_given_transient_row_write_failure_when_classifying_then_retries_and_persists_row(
    session, monkeypatch, caplog
):
    tool = _concept(session, "topic", "tool")
    item = _item(session, source="github_releases")
    session.commit()
    from rumen.classifier import cascade

    original = cascade._replace_facet
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(cascade, "_replace_facet", fail_once)
    with caplog.at_level(logging.WARNING, logger="rumen.classifier.cascade"):
        counts = classify_pending(
            session,
            classify_write_max_attempts=3,
            classify_write_retry_backoff_seconds=0,
        )

    row = session.get(FeedItem, item.id)
    assert calls == 2  # The failed facet write is retried for the same row.
    assert counts["rule"] == 1
    assert row is not None and row.category == "tool"
    assert session.query(ItemConcept).filter_by(item_id=item.id, concept_id=tool.id).count() == 1
    assert "transient write failure" in caplog.text


def test_given_exhausted_row_write_retries_when_classifying_then_chains_original_cause(
    session, monkeypatch
):
    _concept(session, "topic", "tool")
    _item(session, source="github_releases")
    session.commit()
    from rumen.classifier import cascade

    calls = 0

    def always_fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("persistent write failure")

    monkeypatch.setattr(cascade, "_replace_facet", always_fail)
    with pytest.raises(ClassificationError) as error:
        classify_pending(
            session,
            classify_write_max_attempts=3,
            classify_write_retry_backoff_seconds=0,
        )

    assert calls == 3
    assert (
        str(error.value) == "classification batch rolled back after write failure after 3 attempts"
    )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "persistent write failure"


def test_given_write_retry_disabled_when_classifying_then_fails_once_without_sleeping(
    session, monkeypatch
):
    _concept(session, "topic", "tool")
    _item(session, source="github_releases")
    session.commit()
    from rumen.classifier import cascade

    calls = 0

    def always_fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("single write failure")

    monkeypatch.setattr(cascade, "_replace_facet", always_fail)
    monkeypatch.setattr(cascade.time, "sleep", lambda _seconds: pytest.fail("unexpected retry"))
    with pytest.raises(ClassificationError) as error:
        classify_pending(session, classify_write_max_attempts=1)

    assert calls == 1
    assert (
        str(error.value) == "classification batch rolled back after write failure after 1 attempts"
    )
    assert isinstance(error.value.__cause__, RuntimeError)


def test_given_source_category_when_classify_pending_then_preserves_it_and_persists_facets(session):
    tool = _concept(session, "topic", "tool")
    item = _item(session, source="bookmarks")
    item.category = "source-curated"
    item.categories = ["source-curated"]
    item.domains = ["engineering"]
    session.commit()

    llm = type(
        "Llm",
        (),
        {"run": lambda self, prompt: f'{{"concept_ids": [{tool.id}], "confidence": 0.9}}'},
    )()
    counts = classify_pending(session, llm_command=llm, out_of_scope=frozenset({"source-curated"}))
    row = session.get(FeedItem, item.id)

    assert counts["dropped"] == 0
    assert (row.category, row.categories, row.domains) == (
        "source-curated",
        ["source-curated"],
        ["engineering"],
    )
    assert (row.classification_method, row.classification_confidence) == ("source", None)
    assert row.facets_classified_at is not None
    assert session.query(ItemConcept).filter_by(item_id=item.id, concept_id=tool.id).count() == 1


def test_given_completed_unclassified_pass_when_classifying_again_then_selects_no_rows(session):
    item = _item(session)
    session.commit()

    assert classify_pending(session)["unclassified"] == 1
    assert session.get(FeedItem, item.id).facets_classified_at is not None
    assert classify_pending(session) == {
        "rule": 0,
        "embedding": 0,
        "llm": 0,
        "unclassified": 0,
        "dropped": 0,
    }


def test_given_source_category_without_matching_facet_when_classifying_then_stamps_without_projection(
    session,
):
    item = _item(session, source="bookmarks")
    item.category = "source-curated"
    session.commit()

    classify_pending(session)
    row = session.get(FeedItem, item.id)
    assert row.category == "source-curated"
    assert row.facets_classified_at is not None
    assert session.query(ItemConcept).filter_by(item_id=item.id).count() == 0


def test_given_unexpected_route_budget_refusal_when_classifying_then_leaves_marker_null_for_next_run(
    session, monkeypatch
):
    _concept(session, "topic", "unmatched")
    item = _item(session, source="github_releases")
    session.commit()
    from rumen.classifier import cascade

    monkeypatch.setattr(cascade, "classify_tier0", lambda _item: {})
    llm = BudgetedLlmCommand(type("Llm", (), {"run": lambda self, prompt: "{}"})(), max_calls=0)

    classify_pending(session, llm_command=llm)

    assert session.get(FeedItem, item.id).facets_classified_at is None


def test_given_exhausted_unexpected_route_budget_when_classifying_item_then_skips_tier2_invocation(
    session, monkeypatch
):
    _concept(session, "topic", "unmatched")
    item = _item(session, source="github_releases")
    session.commit()
    from rumen.classifier import cascade

    monkeypatch.setattr(cascade, "classify_tier0", lambda _item: {})

    class InnerLlm:
        calls = 0

        def run(self, prompt):
            self.calls += 1
            return "{}"

    class CountingBudgetedLlmCommand(BudgetedLlmCommand):
        run_calls = 0

        def run(self, prompt):
            self.run_calls += 1
            return super().run(prompt)

    inner = InnerLlm()
    llm = CountingBudgetedLlmCommand(inner, max_calls=0)

    result = classify_item(session, item, llm_command=llm)

    assert llm.run_calls == 0
    assert inner.calls == 0
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.tier2_attempted is False
    assert decision.tier2_budget_refused is True
    assert decision.unclassified is True


def test_given_large_expected_tier2_backlog_when_classifying_then_budget_allows_every_item():
    class Llm:
        calls = 0

        def run(self, prompt: str) -> str:
            self.calls += 1
            return "ok"

    inner = Llm()
    budgeted = BudgetedLlmCommand(inner, max_calls=200)
    expected_route = budgeted.for_tier2(expected=True)

    for _ in range(862):
        assert expected_route.run("bookmark facet") == "ok"

    assert budgeted.calls == 862
    assert budgeted.unexpected_calls == 0
    assert inner.calls == 862


def test_given_expected_tier2_route_past_fixed_budget_when_classifying_then_no_item_is_stranded(
    session,
):
    topic = _concept(session, "topic", "bookmark-topic")
    for index in range(3):
        _item(session, source="bookmarks", source_id=f"bookmark-{index}")
    session.commit()

    class Llm:
        calls = 0

        def run(self, prompt: str) -> str:
            self.calls += 1
            return f'{{"concept_ids": [{topic.id}], "confidence": 0.9}}'

    inner = Llm()
    budgeted = BudgetedLlmCommand(inner, max_calls=2)

    counts = classify_pending(session, llm_command=budgeted)

    assert counts["llm"] == 3
    assert inner.calls == 3
    assert session.query(FeedItem).filter(FeedItem.facets_classified_at.is_(None)).count() == 0


def test_given_tier0_source_type_regression_when_classifying_then_budget_refuses_runaway():
    class Llm:
        calls = 0

        def run(self, prompt: str) -> str:
            self.calls += 1
            return "{}"

    inner = Llm()
    budgeted = BudgetedLlmCommand(inner, max_calls=200)
    unexpected_route = budgeted.for_tier2(expected=False)

    assert "topic" in tier0_expected_facets(Tier0Input("github_releases:owner/repo"))

    refused = 0
    for _ in range(1300):
        try:
            unexpected_route.run("github release topic")
        except LlmUnavailable:
            refused += 1

    assert budgeted.calls == budgeted.unexpected_calls == inner.calls == 200
    assert refused == 1100


def test_given_earlier_row_in_uncommitted_batch_when_later_row_retries_then_earlier_row_not_lost(
    session, monkeypatch
):
    """A per-row retry must only roll back its own SAVEPOINT, never the whole uncommitted
    batch -- an earlier row's already-successful write in the same commit_batch_size window
    must survive a later row's failed-then-retried attempt."""
    _concept(session, "topic", "tool")
    first = _item(session, source="github_releases", source_id="first")
    second = _item(session, source="npm_registry", source_id="second")
    session.commit()
    from rumen.classifier import cascade

    original = cascade._replace_facet
    calls = 0

    def fail_on_second_item_once(session_arg, row, assignment):
        nonlocal calls
        calls += 1
        if row.id == second.id and calls <= 2:
            # first item's _replace_facet call already succeeded (calls==1) before this fires
            raise RuntimeError("transient write failure on second item")
        return original(session_arg, row, assignment)

    monkeypatch.setattr(cascade, "_replace_facet", fail_on_second_item_once)
    # commit_batch_size large enough that both rows land in the SAME uncommitted batch
    classify_pending(
        session,
        commit_batch_size=50,
        classify_write_max_attempts=3,
        classify_write_retry_backoff_seconds=0,
    )

    session.commit()
    first_row = session.get(FeedItem, first.id)
    second_row = session.get(FeedItem, second.id)
    assert first_row.category == "tool", "FIRST ROW'S CLASSIFICATION WAS LOST"
    assert second_row.category == "tool", "SECOND ROW'S CLASSIFICATION WAS LOST"
    assert session.query(ItemConcept).filter_by(item_id=first.id).count() == 1
    assert session.query(ItemConcept).filter_by(item_id=second.id).count() == 1
