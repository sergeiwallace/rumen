from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from rumen.adapters.reddit_subreddits import (
    RedditFetchError,
    RedditSubredditAdapter,
    RedditSubredditSettings,
)
from rumen.ingest import run_ingestion

NOW = datetime(2026, 9, 9, 12, tzinfo=UTC)


class Gateway:
    def __init__(self, values):
        self.values, self.calls = values, []

    def listing(self, subreddit, mode, *, limit):
        self.calls.append((subreddit, mode, limit))
        return iter(self.values[(subreddit, mode)])

    def revalidate(self, fullnames):
        return iter(())


def post(name="t3_one", *, hours=1, self_post=False, **values):
    return SimpleNamespace(
        name=name,
        created_utc=(NOW - timedelta(hours=hours)).timestamp(),
        permalink="/r/Test/comments/a/title/",
        subreddit=SimpleNamespace(display_name="Test"),
        author=SimpleNamespace(name="author"),
        title="Title",
        selftext="Body",
        is_self=self_post,
        url="https://example.test",
        link_flair_text="flair",
        over_18=False,
        score=3,
        num_comments=4,
        **values,
    )


def comment(name="t1_one", **values):
    return SimpleNamespace(
        name=name,
        created_utc=(NOW - timedelta(hours=1)).timestamp(),
        permalink="/r/Test/comments/a/title/b/",
        subreddit=SimpleNamespace(display_name="Test"),
        author=None,
        body="[deleted]",
        link_id="t3_a",
        parent_id="t3_a",
        **values,
    )


def adapter(gateway, **values):
    settings = RedditSubredditSettings(
        "ai", ("Test",), ("submissions",), "ID", "SECRET", "agent", "approval", **values
    )
    return RedditSubredditAdapter(settings, gateway=gateway, now_fn=lambda: NOW)


def test_given_link_self_and_comment_when_fetched_then_maps_contract_fields() -> None:
    gateway = Gateway(
        {
            ("Test", "submissions"): [post(), post("t3_self", self_post=True)],
            ("Test", "comments"): [comment()],
        }
    )
    settings = RedditSubredditSettings(
        "ai", ("Test",), ("submissions", "comments"), "ID", "SECRET", "agent", "approval"
    )
    items = RedditSubredditAdapter(settings, gateway=gateway, now_fn=lambda: NOW).fetch_new(None)
    assert items[0].url == "https://example.test"
    assert items[1].url.startswith("https://www.reddit.com/")
    assert items[2].title == "Comment in r/Test" and items[2].content is None
    assert items[2].metadata["link_fullname"] == "t3_a"


def test_given_replay_duplicates_and_retention_when_fetched_then_dedups_and_bounds() -> None:
    gateway = Gateway(
        {
            ("Test", "submissions"): [
                post("t3_new", hours=2),
                post("t3_new", hours=3),
                post("t3_old", hours=49),
            ]
        }
    )
    items = adapter(gateway).fetch_new(NOW - timedelta(hours=1))
    assert [item.source_id for item in items] == ["t3_new"]
    assert gateway.calls == [("Test", "submissions", 100)]


def test_given_malformed_or_truncated_listing_when_fetched_then_fails_atomically(session) -> None:
    malformed = post(name="bad")
    with pytest.raises(RedditFetchError, match="malformed"):
        adapter(Gateway({("Test", "submissions"): [malformed]})).fetch_new(None)
    capped = adapter(
        Gateway({("Test", "submissions"): [post(str(i).join(("t3_", ""))) for i in range(1000)]})
    )
    with pytest.raises(RedditFetchError, match="truncated"):
        capped.fetch_new(None)
    failed = run_ingestion(session, adapter(Gateway({("Test", "submissions"): [malformed]})))
    assert (
        failed.status == "failed"
        and session.query(__import__("rumen.models", fromlist=["FeedItem"]).FeedItem).count() == 0
    )
