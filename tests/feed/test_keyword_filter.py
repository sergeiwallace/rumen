"""T-06: keyword filter — inclusion drop (keyword-scored), bookmark bypass, suppression."""

from __future__ import annotations

from datetime import UTC, datetime

from rumen.keyword_filter import apply_keyword_filter
from rumen.models import DroppedItem, FeedItem, FeedKeyword

_NOW = datetime(2026, 7, 10, tzinfo=UTC)


def _add_item(session, source: str, source_id: str, title: str, content: str = "") -> None:
    session.add(
        FeedItem(
            source=source,
            source_id=source_id,
            url=f"https://example.com/{source_id}",
            title=title,
            content=content,
            published_at=_NOW,
        )
    )


def _keywords(session, include=(), suppress=()) -> None:
    for kw in include:
        session.add(FeedKeyword(keyword=kw, keyword_type="include"))
    for kw in suppress:
        session.add(FeedKeyword(keyword=kw, keyword_type="suppress"))


def test_keyword_scored_zero_match_is_dropped(session) -> None:
    _keywords(session, include=["Claude"])
    _add_item(session, "github_releases", "g1", "random unrelated release")
    session.commit()

    apply_keyword_filter(session)
    item = session.query(FeedItem).filter_by(source_id="g1").one()
    assert item.status == "dropped"
    dropped = session.query(DroppedItem).filter_by(source_id="g1").one()
    assert dropped.drop_reason == "keyword_miss"


def test_match_stores_matched_keywords(session) -> None:
    _keywords(session, include=["Claude", "agent"])
    _add_item(session, "github_releases", "g2", "New Claude agent release")
    session.commit()

    apply_keyword_filter(session)
    item = session.query(FeedItem).filter_by(source_id="g2").one()
    assert set(item.matched_keywords) == {"Claude", "agent"}


def test_bookmark_zero_match_is_retained_and_annotated(session) -> None:
    _keywords(session, include=["Claude"])
    _add_item(session, "bookmarks", "b1", "some unrelated bookmark")
    _add_item(session, "bookmarks", "b2", "a Claude thing")
    session.commit()

    apply_keyword_filter(session)
    # Bookmark bypasses the inclusion drop even with zero matches.
    kept = session.query(FeedItem).filter_by(source_id="b1").one()
    assert kept.matched_keywords is None
    annotated = session.query(FeedItem).filter_by(source_id="b2").one()
    assert annotated.matched_keywords == ["Claude"]


def test_suppress_drops_any_source_even_with_inclusion_match(session) -> None:
    _keywords(session, include=["Claude"], suppress=["imaging"])
    _add_item(session, "github_releases", "g3", "Claude medical imaging tool")
    _add_item(session, "bookmarks", "b3", "Claude imaging bookmark")
    session.commit()

    apply_keyword_filter(session)
    assert session.query(FeedItem).filter_by(source_id="g3").one().status == "dropped"
    assert session.query(FeedItem).filter_by(source_id="b3").one().status == "dropped"
    reasons = {d.source_id: d.drop_reason for d in session.query(DroppedItem).all()}
    assert reasons["g3"] == "suppressed"
    assert reasons["b3"] == "suppressed"


def test_empty_inclusion_list_default_open_when_empty_keeps_items(session) -> None:
    # Documented default: open_when_empty=True — an empty inclusion list keeps
    # keyword-scored items (an unconfigured list must not drop the whole corpus).
    _add_item(session, "github_releases", "g4", "anything at all")
    session.commit()
    apply_keyword_filter(session)
    assert session.query(FeedItem).filter_by(source_id="g4").count() == 1


def test_empty_inclusion_list_with_open_when_empty_false_drops_keyword_scored(session) -> None:
    _add_item(session, "github_releases", "g5", "anything at all")
    session.commit()
    apply_keyword_filter(session, open_when_empty=False)
    assert session.query(FeedItem).filter_by(source_id="g5").one().status == "dropped"
    assert session.query(DroppedItem).filter_by(source_id="g5").one().drop_reason == "keyword_miss"


def test_keyword_dropped_item_is_not_logged_again_on_subsequent_runs(session) -> None:
    _keywords(session, include=["Claude"])
    _add_item(session, "github_releases", "g6", "random unrelated release")
    session.commit()

    apply_keyword_filter(session)
    apply_keyword_filter(session)

    assert session.query(FeedItem).filter_by(source_id="g6").one().status == "dropped"
    assert session.query(DroppedItem).filter_by(source_id="g6").count() == 1
