"""T-17 ACs 2 and 6: hierarchy traversal/consistency and zero-row fit safety."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from rumen.classifier.tier1_embedding import FacetHMCClassifier
from rumen.embeddings import HashingEmbeddingProvider
from rumen.models import FeedItem, ItemConcept, SkosConcept, SkosRelation
from rumen.taxonomy.hmc.cleanlab_gate import CleanlabGateResult
from rumen.taxonomy.hmc.fit import (
    BootstrapExample,
    fit_facet,
    load_bootstrap_examples,
    load_facet_parents,
    save_report_bundle,
)


@pytest.mark.parametrize(
    "bad_facet", ["../outside", "absolute", "nested/facet", r"nested\facet", "..", ""]
)
@pytest.mark.parametrize("has_classifier", [True, False])
def test_given_unsafe_facet_when_saving_bundle_then_rejects_before_any_write(
    tmp_path, bad_facet, has_classifier
) -> None:
    if bad_facet == "absolute":
        bad_facet = str(tmp_path / "outside")
    report = fit_facet(
        "topic",
        [BootstrapExample("one", "alpha", 1, 0.9)],
        provider=HashingEmbeddingProvider(),
        gate=lambda _labels, _probabilities: CleanlabGateResult((0,), ()),
    )
    unsafe_report = replace(
        report, facet=bad_facet, classifier=report.classifier if has_classifier else None
    )
    bundle = tmp_path / "bundle"

    with pytest.raises(ValueError, match="invalid HMC bundle facet"):
        save_report_bundle({"topic": report, bad_facet: unsafe_report}, bundle)

    assert not bundle.exists()
    assert list(tmp_path.iterdir()) == []


def test_hmc_when_parent_is_below_path_floor_then_high_scoring_child_is_not_assigned(
    session,
) -> None:
    parent = SkosConcept(id=10, facet="topic", pref_label="Parent", scheme="fixture")
    child = SkosConcept(id=20, facet="topic", pref_label="Child", scheme="fixture")
    narrower_child = SkosConcept(id=30, facet="topic", pref_label="Narrower", scheme="fixture")
    session.add_all([parent, child, narrower_child])
    session.add(SkosRelation(source_concept_id=20, target_concept_id=10, relation_type="broader"))
    session.add(SkosRelation(source_concept_id=10, target_concept_id=30, relation_type="narrower"))
    session.commit()
    parents = load_facet_parents(session, facet="topic")
    classifier = FacetHMCClassifier("topic", HashingEmbeddingProvider(), parents=parents).fit(
        [("parent", 10), ("child", 20)]
    )

    assignments = classifier.assign_scores({10: 0.49, 20: 0.99}, path_consistency_floor=0.75)

    assert parents == {20: {10}, 30: {10}}
    assert assignments == {}


def test_bootstrap_loader_when_facet_contains_gold_and_bootstrap_rows_then_uses_only_bootstrap_rows(
    session,
) -> None:
    session.add_all(
        [
            SkosConcept(id=10, facet="topic", pref_label="Bootstrap", scheme="fixture"),
            SkosConcept(id=20, facet="topic", pref_label="Gold", scheme="fixture"),
        ]
    )
    item = FeedItem(
        source="fixture",
        source_id="one",
        url="https://example.test/one",
        title="bootstrap source text",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    session.add(item)
    session.flush()
    session.add_all(
        [
            ItemConcept(
                item_table="feed_items",
                item_id=item.id,
                concept_id=10,
                facet="topic",
                confidence=0.8,
                method="bootstrap:snorkel",
            ),
            ItemConcept(
                item_table="feed_items",
                item_id=item.id,
                concept_id=20,
                facet="topic",
                confidence=1.0,
                method="human:gold-fixture",
            ),
        ]
    )
    session.commit()

    examples = load_bootstrap_examples(session, facet="topic")

    assert [(example.concept_id, example.confidence) for example in examples] == [(10, 0.8)]


def test_hmc_when_cleanlab_flags_every_bootstrap_example_then_fit_is_skipped_and_logs_error(
    caplog,
) -> None:
    examples = [
        BootstrapExample("one", "alpha", 1, 0.9),
        BootstrapExample("two", "beta", 2, 0.9),
    ]

    def flag_all(_labels: list[int], _probabilities: np.ndarray) -> CleanlabGateResult:
        return CleanlabGateResult((), (0, 1))

    with caplog.at_level(logging.ERROR):
        report = fit_facet("topic", examples, provider=HashingEmbeddingProvider(), gate=flag_all)

    assert report.skipped is True
    assert report.classifier is None
    assert report.n_after_gate == 0
    assert "flagged all" in caplog.text


def test_given_live_hmc_method_when_loading_fit_examples_then_it_is_neither_bootstrap_nor_gold(
    session,
) -> None:
    concept = SkosConcept(id=99, facet="topic", pref_label="Live", scheme="fixture")
    item = FeedItem(
        source="fixture",
        source_id="live",
        url="https://example/live",
        title="live text",
        published_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    session.add_all([concept, item])
    session.flush()
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=concept.id,
            facet="topic",
            confidence=0.9,
            method="hmc",
        )
    )
    session.commit()

    assert load_bootstrap_examples(session, facet="topic") == []
    from rumen.taxonomy.hmc.gold_eval import load_human_gold_examples

    assert load_human_gold_examples(session, facet="topic") == []
