"""D-18: incremental-durability — a stage interrupted mid-run keeps its committed batches.

Each stage commits every ``commit_batch_size`` rows so a killed run (the real incident: a
~1hr grouping pass rolled back entirely) loses at most one batch, not everything. These tests
simulate an interruption by raising partway through and asserting the earlier batches survived.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rumen import grouping as grouping_pkg
from rumen.classifier import cascade
from rumen.classifier.cascade import classify_pending
from rumen.grouping import group_pending
from rumen.models import FeedItem, SkosConcept, TopicEntity

_BASE = datetime(2026, 7, 10, tzinfo=UTC)


def _add(session, source, source_id, url, *, title="", offset=0) -> FeedItem:
    item = FeedItem(
        source=source,
        source_id=source_id,
        url=url,
        title=title,
        published_at=_BASE + timedelta(hours=offset),
    )
    session.add(item)
    session.flush()
    return item


def test_classification_interrupted_midrun_keeps_committed_batches(session, monkeypatch) -> None:
    # 8 github items (Tier-0 → "tool", deterministic, no embedding/LLM).
    for n in range(8):
        _add(session, f"github_releases:owner/r{n}", f"g{n}", f"https://github.com/owner/r{n}")
    session.add_all(
        [
            SkosConcept(facet="topic", pref_label="tool", scheme="test"),
            SkosConcept(facet="content-type", pref_label="tool-release", scheme="test"),
            SkosConcept(facet="source", pref_label="github-releases", scheme="test"),
        ]
    )
    session.commit()

    real = cascade.classify_item
    seen = {"n": 0}

    def _boom(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 5:  # fail on the 5th item — batches of 2 already committed items 1-4
            raise RuntimeError("simulated crash mid-classification")
        return real(*args, **kwargs)

    monkeypatch.setattr(cascade, "classify_item", _boom)

    with pytest.raises(RuntimeError):
        classify_pending(session, commit_batch_size=2, classify_write_max_attempts=1)

    session.rollback()  # drop the failed (uncommitted) transaction
    classified = session.query(FeedItem).filter(FeedItem.category == "tool").count()
    assert classified == 4  # two committed batches survived the crash, not zero


def test_grouping_interrupted_midrun_keeps_committed_entities(session, monkeypatch) -> None:
    for n in range(8):
        _add(session, "reddit", f"u{n}", f"https://reddit.com/r/x/{n}", title=f"post {n}", offset=n)
    session.commit()

    real = grouping_pkg.recompute_entity
    seen = {"n": 0}

    def _boom(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 5:
            raise RuntimeError("simulated crash mid-grouping")
        return real(*args, **kwargs)

    monkeypatch.setattr(grouping_pkg, "recompute_entity", _boom)

    with pytest.raises(RuntimeError):
        group_pending(session, embed=False, commit_batch_size=2)

    session.rollback()
    assert session.query(TopicEntity).count() == 4  # committed batches survived
    # expire_on_commit is restored even on the exception path (finally block).
    assert session.expire_on_commit is True


def test_grouping_resumes_on_ungrouped_items_after_partial_run(session, monkeypatch) -> None:
    # After a partial grouping run, a re-run must pick up only the still-ungrouped pointers
    # (idempotent on topic_entity_id IS NULL) and finish the job.
    for n in range(8):
        _add(session, "reddit", f"v{n}", f"https://reddit.com/r/y/{n}", title=f"item {n}", offset=n)
    session.commit()

    real = grouping_pkg.recompute_entity
    seen = {"n": 0}

    def _boom(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 5:
            raise RuntimeError("crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(grouping_pkg, "recompute_entity", _boom)
    with pytest.raises(RuntimeError):
        group_pending(session, embed=False, commit_batch_size=2)
    session.rollback()
    assert session.query(FeedItem).filter(FeedItem.topic_entity_id.is_(None)).count() == 4

    monkeypatch.undo()  # "restart" the process — recompute_entity works again
    group_pending(session, embed=False, commit_batch_size=2)
    assert session.query(FeedItem).filter(FeedItem.topic_entity_id.is_(None)).count() == 0
    assert session.query(TopicEntity).count() == 8  # all eight now grouped, none lost or dup'd
