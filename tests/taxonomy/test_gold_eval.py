"""T-17 ACs 3–5: held-out human gold evaluation gates Tier-1 eligibility."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from rumen.classifier.tier1_embedding import FacetHMCClassifier
from rumen.embeddings import HashingEmbeddingProvider
from rumen.models import FeedItem, ItemConcept, SkosConcept
from rumen.taxonomy.hmc.cleanlab_gate import CleanlabGateResult
from rumen.taxonomy.hmc.fit import BootstrapExample, fit_facet
from rumen.taxonomy.hmc.gold_eval import (
    GoldExample,
    evaluate_gold_set,
    load_human_gold_examples,
)


def _keep_all(_labels: list[int], _probabilities: np.ndarray) -> CleanlabGateResult:
    return CleanlabGateResult((0, 1), ())


def _bootstrap() -> list[BootstrapExample]:
    return [
        BootstrapExample("bootstrap:one", "alpha parent", 1, 0.95),
        BootstrapExample("bootstrap:two", "beta child", 2, 0.95),
    ]


def test_gold_eval_when_held_out_human_labels_then_reports_facet_precision_and_recall() -> None:
    classifier = FacetHMCClassifier("topic", HashingEmbeddingProvider()).fit(
        [("alpha human gold", 1)]
    )
    # This deterministic fixture is human gold, deliberately separate from any bootstrap labels.
    gold = [GoldExample("alpha human gold", frozenset({1})), GoldExample("unseen", frozenset({2}))]

    evaluation = evaluate_gold_set(classifier, gold, path_consistency_floor=0.9)

    assert evaluation.evaluated is True
    assert evaluation.evaluated_count == 2
    assert evaluation.precision == 1.0
    assert evaluation.recall == 0.5


def test_fit_when_gold_precision_below_floor_then_all_assignments_fall_through_tier2() -> None:
    report = fit_facet(
        "topic",
        _bootstrap(),
        gold_examples=[GoldExample("alpha parent", frozenset({999}))],
        provider=HashingEmbeddingProvider(),
        gold_precision_floor=0.9,
        path_consistency_floor=0.0,
        gate=_keep_all,
    )

    assert report.gold_evaluation.precision == 0.0
    assert report.tier1_eligible is False
    assert report.assign_for_tier1("alpha parent", path_consistency_floor=0.0) == {}
    assert report.assign_for_tier1("beta child", path_consistency_floor=0.0) == {}


def test_fit_when_facet_has_no_gold_set_then_it_is_unevaluated_and_falls_through_tier2() -> None:
    report = fit_facet(
        "topic",
        _bootstrap(),
        provider=HashingEmbeddingProvider(),
        gate=_keep_all,
    )

    assert report.gold_evaluation.evaluated is False
    assert report.gold_evaluation.precision is None
    assert report.tier1_eligible is False
    assert report.assign_for_tier1("alpha parent", path_consistency_floor=0.0) == {}


def test_load_human_gold_examples_when_row_is_t14_rule_backfill_then_excluded(session) -> None:
    """T-14's legacy-crosswalk backfill writes method='rule' — a mechanical remap of the
    original noisy Flash label, not a human judgment. It must never be treated as gold, or
    gold-set evaluation collapses back into agreement-with-the-noisy-teacher (the exact
    failure T-17 exists to fix)."""
    session.add(SkosConcept(id=1, facet="topic", pref_label="Economy", scheme="fixture"))
    item = FeedItem(
        source="fixture",
        source_id="legacy-one",
        url="https://example.test/legacy-one",
        title="legacy backfilled item",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    session.add(item)
    session.flush()
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=1,
            facet="topic",
            confidence=1.0,
            method="rule",
        )
    )
    session.commit()

    assert load_human_gold_examples(session, facet="topic") == []
