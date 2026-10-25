"""T-07: reversible merge_log + non-blocking review queue + unmerge (design D-8)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rumen.grouping import group_pending
from rumen.grouping.merge import record_non_duplicate, unmerge_item
from rumen.models import (
    FeedFeedback,
    FeedItem,
    MergeCandidate,
    MergeLog,
    TopicEntity,
)

_BASE = datetime(2026, 7, 10, tzinfo=UTC)


def _add(session, source, source_id, url, *, title="", content=None, offset=0) -> FeedItem:
    item = FeedItem(
        source=source,
        source_id=source_id,
        url=url,
        title=title,
        content=content,
        published_at=_BASE + timedelta(hours=offset),
    )
    session.add(item)
    session.flush()
    return item


def _make_two_pointer_entity(session) -> TopicEntity:
    _add(session, "github_stars", "s1", "https://github.com/owner/repo", title="repo", offset=0)
    _add(
        session,
        "bookmarks",
        "t1",
        "https://x.com/u/status/1",
        title="cool",
        content="https://github.com/owner/repo",
        offset=1,
    )
    session.commit()
    group_pending(session, embed=False)
    return session.query(TopicEntity).one()


def test_every_merge_writes_a_merge_log_row(session) -> None:
    entity = _make_two_pointer_entity(session)
    logs = session.query(MergeLog).filter_by(action="merge").all()
    assert len(logs) == 2  # one per attached pointer (seed + merge)
    assert all(log.match_method == "stable_id" for log in logs)
    assert all(log.topic_entity_id == entity.id for log in logs)
    assert all(log.evidence["stable_id_value"] == "owner/repo" for log in logs)


def test_unmerge_splits_into_fresh_entity_and_logs(session) -> None:
    entity = _make_two_pointer_entity(session)
    tweet = session.query(FeedItem).filter_by(source_id="t1").one()

    fresh = unmerge_item(session, tweet)
    session.commit()

    assert session.query(TopicEntity).count() == 2
    assert session.get(FeedItem, tweet.id).topic_entity_id == fresh.id
    assert session.query(MergeLog).filter_by(action="unmerge").count() == 1
    # both entities recompute their source_count
    assert session.get(TopicEntity, entity.id).source_count == 1
    assert fresh.source_count == 1


def test_unmerge_records_non_duplicate_and_keeps_feedback_on_survivor(session) -> None:
    entity = _make_two_pointer_entity(session)
    session.add(FeedFeedback(topic_entity_id=entity.id, rating=1))
    session.commit()

    tweet = session.query(FeedItem).filter_by(source_id="t1").one()
    unmerge_item(session, tweet)
    session.commit()

    # non_duplicate override recorded so the pair does not auto re-merge.
    cand = session.query(MergeCandidate).one()
    assert cand.status == "rejected"
    # feedback stays on the surviving (original) entity.
    fb = session.query(FeedFeedback).one()
    assert fb.topic_entity_id == entity.id


def test_rejected_candidate_blocks_remerge_in_residue(session) -> None:
    # Two near-identical no-ID entities that would band-merge, but pre-marked non_duplicate.
    _add(
        session,
        "reddit",
        "y1",
        "https://reddit.com/r/x/1",
        title="alpha beta gamma delta",
        offset=0,
    )
    _add(
        session,
        "reddit",
        "y2",
        "https://reddit.com/r/x/2",
        title="alpha beta gamma delta",
        offset=1,
    )
    session.commit()
    group_pending(session, embed=False)  # identity pass only → 2 entities
    a, b = session.query(TopicEntity).order_by(TopicEntity.id).all()
    record_non_duplicate(session, a, b, resolved_by="human")
    session.commit()

    # Re-run grouping with embeddings — the rejected pair must not auto-merge.
    group_pending(session, tau_high=0.9, tau_low=0.4)
    assert session.query(TopicEntity).count() == 2
