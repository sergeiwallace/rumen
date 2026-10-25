"""Reddit saved-cache adapter coverage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rumen.adapters.reddit import RedditSavedAdapter
from rumen.config import load_config


def _make_db(path: Path, *, synced_at: str = "2026-09-02T00:00:00+00:00") -> str:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE reddit_saved_items (
            fullname TEXT, kind TEXT, title TEXT, body TEXT, author TEXT, permalink TEXT,
            created_at TEXT, subreddit TEXT, synced_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO reddit_saved_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "t1_comment",
            "t1",
            "Comment in r/python",
            "Useful detail",
            "example-user-1",
            "/r/python/comments/post/comment/",
            "2026-09-01T12:00:00+00:00",
            "python",
            synced_at,
        ),
    )
    conn.commit()
    conn.close()
    return str(path)


def test_given_cached_comment_when_fetching_then_maps_to_raw_item(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "reddit.db")
    adapter = RedditSavedAdapter(db, now_fn=lambda: datetime(2026, 9, 2, 1, tzinfo=UTC))

    [item] = adapter.fetch_new(None)

    assert item.source == "reddit_saved"
    assert item.source_id == "t1_comment"
    assert item.url == "https://www.reddit.com/r/python/comments/post/comment/"
    assert item.title == "Comment in r/python"
    assert item.content == "Useful detail"
    assert item.author == "example-user-1"
    assert item.published_at == datetime(2026, 9, 1, 12, tzinfo=UTC)
    assert item.metadata == {"kind": "t1", "subreddit": "python"}


def test_given_stale_cache_when_fetching_then_records_staleness_warning(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "reddit.db", synced_at="2026-08-28T00:00:00+00:00")
    adapter = RedditSavedAdapter(
        db,
        staleness_threshold=timedelta(hours=48),
        now_fn=lambda: datetime(2026, 9, 2, tzinfo=UTC),
    )

    adapter.fetch_new(None)

    assert adapter.last_staleness_warning is not None
    assert "SYNC_STALE" in adapter.last_staleness_warning


def test_given_reddit_source_config_when_loading_then_builds_reddit_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "classifier_escalation_alarm_threshold": 0.25,
                "sources": [
                    {
                        "type": "reddit_saved",
                        "host": "main-desktop",
                        "db_path": "/operator/reddit-saved.db",
                        "staleness_hours": 24,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_HOST", "main-desktop")

    config = load_config(config_path)

    [adapter] = config.adapters
    assert isinstance(adapter, RedditSavedAdapter)
    assert adapter.db_path == "/operator/reddit-saved.db"
    assert adapter.staleness_threshold == timedelta(hours=24)
