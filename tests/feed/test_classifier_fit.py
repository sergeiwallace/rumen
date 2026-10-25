"""T-25 HMC fit/load command and unfit fallback regressions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rumen.classifier.fit import fit_from_session, load_classifier, save_classifier
from rumen.embeddings import HashingEmbeddingProvider
from rumen.jobs import run_pipeline
from rumen.models import FeedItem, ItemConcept, SkosConcept

_NOW = datetime(2026, 7, 20, tzinfo=UTC)


def _bootstrap_item(session, source_id, concept):
    item = FeedItem(
        source="reddit",
        source_id=source_id,
        url=f"https://example/{source_id}",
        title="tool release package",
        published_at=_NOW,
    )
    session.add(item)
    session.flush()
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=concept.id,
            facet="topic",
            confidence=0.9,
            method="bootstrap:fixture",
        )
    )


def test_given_bootstrap_facets_when_fit_classifier_runs_then_it_persists_and_loads_the_hmc_bundle(
    session, tmp_path
):
    pytest.importorskip("cleanlab", reason="hmc-fit extra is not installed")
    concept = SkosConcept(facet="topic", pref_label="tool", scheme="test")
    session.add(concept)
    session.flush()
    _bootstrap_item(session, "one", concept)
    _bootstrap_item(session, "two", concept)
    session.commit()

    reports = fit_from_session(session, provider=HashingEmbeddingProvider())
    out = tmp_path / "hmc-bundle"
    save_classifier(reports, out)
    loaded = load_classifier(out, provider=HashingEmbeddingProvider())
    assert (out / "manifest.json").exists() and (out / "topic.npz").exists()
    assert loaded is not None and loaded["topic"].n_bootstrap == 2
    assert loaded["topic"].classifier is not None


def test_given_no_hmc_bundle_when_run_then_it_uses_the_unfit_facet_fallback(session):
    # A configured run with no reports does not synthesize a flat model; Tier-0 still works.
    concept = SkosConcept(facet="topic", pref_label="tool", scheme="test")
    session.add(concept)
    session.commit()
    result = run_pipeline(session, adapters=[], reports=None, now=_NOW)
    assert result.classified == {
        "rule": 0,
        "embedding": 0,
        "llm": 0,
        "unclassified": 0,
        "dropped": 0,
    }
    assert load_classifier("/definitely/missing/hmc-bundle") is None
