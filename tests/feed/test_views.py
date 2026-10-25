"""T-09: configurable views + presentation-layer filtering + MCP tools (D-6)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rumen.mcp_tools import get_feed_digest, get_feed_items, rate_feed_entity
from rumen.models import Digest, FeedFeedback, FeedItem, FeedView, TopicEntity
from rumen.views import UnknownSortError, UnknownViewError, filter_items, get_view

_NOW = datetime(2026, 7, 10, tzinfo=UTC)


def _seed(session) -> None:
    session.add(FeedView(name="AI/ML", filter_json={"category": ["tool", "paper"]}))
    session.add(FeedView(name="All", filter_json={}))
    for sid, cat, src in [
        ("t1", "tool", "github_releases"),
        ("p1", "paper", "hf_daily"),
        ("pol1", "politics", "bookmarks"),
        ("hum1", "humor", "bookmarks"),
    ]:
        session.add(
            FeedItem(
                source=src,
                source_id=sid,
                url=f"u/{sid}",
                title=sid,
                category=cat,
                published_at=_NOW,
            )
        )
    session.commit()


def test_view_filters_to_its_categories(session) -> None:
    _seed(session)
    cats = {i.category for i in filter_items(session, view_name="AI/ML")}
    assert cats == {"tool", "paper"}


def test_all_view_returns_full_corpus(session) -> None:
    _seed(session)
    cats = {i.category for i in filter_items(session, view_name="All")}
    assert cats == {"tool", "paper", "politics", "humor"}  # nothing hardcoded out


def test_ad_hoc_category_reaches_items_absent_from_default_view(session) -> None:
    _seed(session)
    # politics is not in the AI/ML view, but an ad-hoc filter still reaches it.
    items = filter_items(session, category="politics")
    assert {i.source_id for i in items} == {"pol1"}


def test_unknown_view_raises(session) -> None:
    _seed(session)
    with pytest.raises(UnknownViewError, match="no such view"):
        get_view(session, "does-not-exist")


def test_view_sort_json_overrides_default_order(session) -> None:
    _seed(session)
    session.add(
        FeedView(name="By category", filter_json={}, sort_json={"by": "category", "dir": "asc"})
    )
    session.commit()
    assert [item.category for item in filter_items(session, view_name="By category")] == [
        "humor",
        "paper",
        "politics",
        "tool",
    ]


def test_unknown_view_sort_raises_typed_error(session) -> None:
    session.add(FeedView(name="Bad sort", filter_json={}, sort_json={"by": "drop table"}))
    session.commit()
    with pytest.raises(UnknownSortError, match="unknown view sort field"):
        filter_items(session, view_name="Bad sort")


def test_get_feed_digest_returns_latest(session) -> None:
    assert get_feed_digest(session) is None
    session.add(
        Digest(
            period_start=_NOW,
            period_end=_NOW,
            content_json={"active_tool_updates": []},
            content_markdown="# d",
        )
    )
    session.commit()
    digest = get_feed_digest(session)
    assert digest["content"] == {"active_tool_updates": []}


def test_get_feed_items_honors_view_and_ad_hoc(session) -> None:
    _seed(session)
    assert {i["category"] for i in get_feed_items(session, view="AI/ML")} == {"tool", "paper"}
    assert {
        i["source_id"] if "source_id" in i else i["category"]
        for i in get_feed_items(session, category="humor")
    } == {"humor"}


def test_rate_feed_entity_writes_feedback(session) -> None:
    entity = TopicEntity(canonical_title="e", canonical_url="u", earliest_seen=_NOW, source_count=1)
    session.add(entity)
    session.commit()
    rate_feed_entity(session, entity.id, 1)
    fb = session.query(FeedFeedback).one()
    assert fb.topic_entity_id == entity.id
    assert fb.rating == 1


def test_rate_feed_entity_rejects_bad_rating(session) -> None:
    entity = TopicEntity(canonical_title="e", canonical_url="u", earliest_seen=_NOW, source_count=1)
    session.add(entity)
    session.commit()
    with pytest.raises(ValueError, match="rating must be"):
        rate_feed_entity(session, entity.id, 5)
