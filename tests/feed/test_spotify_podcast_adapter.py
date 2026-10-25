"""Read-only adapter and factory coverage for Spotify podcast caches."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rumen.adapters.spotify_podcasts import (
    SpotifyPodcastAdapter,
    SpotifyPodcastCacheError,
)
from rumen.config import load_config
from rumen.ingest import last_high_water_mark, run_ingestion
from rumen.models import FeedItem
from rumen.podcasts.cache import (
    initialize,
    record_feed,
    store_ready_transcript,
    transaction,
)
from rumen.podcasts.transcripts import TranscriptSegment


def _ready_cache(path: Path, *, retrieved_at: str | None = None) -> None:
    conn = initialize(path)
    with transaction(conn):
        record_feed(
            conn, feed_url="https://publisher.example/feed", show_key="test show", body="<rss/>"
        )
        conn.execute(
            """INSERT INTO podcast_episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "episode-key",
                "https://publisher.example/feed",
                "guid",
                "https://publisher.example/episode",
                None,
                None,
                "Episode",
                "Test Show",
                "2026-08-01T00:00:00+00:00",
                None,
                "resolved",
                "test",
                "pending",
                "2026-08-01T00:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO spotify_podcast_listening_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "event",
                "spotify:episode:test",
                "Episode",
                "Test Show",
                "2026-08-02T00:00:00+00:00",
                1,
                None,
                None,
                "{}",
            ),
        )
        conn.execute(
            "INSERT INTO podcast_episode_aliases VALUES (?, ?, ?, ?, ?)",
            ("spotify:episode:test", "episode-key", "exact", "high", "test"),
        )
        store_ready_transcript(
            conn,
            "episode-key",
            source_class="publisher",
            source_url="https://publisher.example/text",
            mime_type="text/plain",
            language="en",
            plain_text="Transcript text",
            segments=[TranscriptSegment(None, None, None, "Transcript text")],
        )
        if retrieved_at:
            conn.execute("UPDATE podcast_transcripts SET retrieved_at = ?", (retrieved_at,))
    conn.close()


def _config(path: Path, sources: list[dict]) -> Path:
    path.write_text(json.dumps({"sources": sources, "classifier_escalation_alarm_threshold": 0.25}))
    return path


def test_given_ready_cache_when_fetching_with_any_cursor_then_returns_full_deterministic_items(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "podcasts.db"
    _ready_cache(cache)

    item = SpotifyPodcastAdapter(str(cache), "personal").fetch_new(datetime.now(UTC))[0]

    assert (item.source, item.source_id, item.content, item.author) == (
        "spotify_podcasts:personal",
        "episode-key",
        "Transcript text",
        "Test Show",
    )
    assert item.metadata["transcript"]["source_class"] == "publisher"
    assert item.metadata["spotify_episode_uris"] == ["spotify:episode:test"]


def test_given_unchanged_cache_when_ingested_twice_then_uses_existing_source_id_dedup(
    session, tmp_path: Path
) -> None:
    cache = tmp_path / "podcasts.db"
    _ready_cache(cache)
    adapter = SpotifyPodcastAdapter(str(cache), "personal")

    first = run_ingestion(session, adapter)
    second = run_ingestion(session, adapter)

    assert (first.status, second.status) == ("completed", "completed")
    assert session.query(FeedItem).filter_by(source=adapter.source_name()).count() == 1
    assert last_high_water_mark(session, adapter.source_name()) is not None


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"type": "spotify_podcasts", "profile": "bad profile", "db_path": "/x"}, "profile"),
        ({"type": "spotify_podcasts", "profile": "personal", "db_path": ""}, "db_path"),
        (
            {
                "type": "spotify_podcasts",
                "profile": "personal",
                "db_path": "/x",
                "staleness_hours": 0,
            },
            "staleness_hours",
        ),
        (
            {"type": "spotify_podcasts", "profile": "personal", "db_path": "/x", "unknown": 1},
            "unknown",
        ),
    ],
)
def test_given_invalid_spotify_source_when_loading_config_then_rejects_field(
    tmp_path: Path, source: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_config(tmp_path / "config.json", [source]))


def test_given_host_applicable_duplicate_profiles_when_loading_config_then_rejects(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AI_HOST", "test")
    sources = [
        {"type": "spotify_podcasts", "host": "test", "profile": "personal", "db_path": "/one"},
        {"type": "spotify_podcasts", "host": "test", "profile": "personal", "db_path": "/two"},
    ]
    with pytest.raises(ValueError, match="duplicate host-applicable"):
        load_config(_config(tmp_path / "config.json", sources))


def test_given_valid_host_source_when_loading_config_then_constructs_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AI_HOST", "test")
    cfg = load_config(
        _config(
            tmp_path / "config.json",
            [
                {
                    "type": "spotify_podcasts",
                    "host": "test",
                    "profile": "personal",
                    "db_path": "/private/cache.db",
                    "staleness_hours": 12,
                }
            ],
        )
    )
    [adapter] = cfg.adapters
    assert isinstance(adapter, SpotifyPodcastAdapter)
    assert (adapter.profile, adapter.db_path, adapter.staleness_threshold) == (
        "personal",
        "/private/cache.db",
        timedelta(hours=12),
    )


def test_given_absent_or_stale_cache_when_fetching_then_raises_connector_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(SpotifyPodcastCacheError, match="does not exist"):
        SpotifyPodcastAdapter(str(tmp_path / "absent.db"), "personal").fetch_new(None)
    cache = tmp_path / "old.db"
    _ready_cache(cache, retrieved_at="2000-01-01T00:00:00+00:00")
    with pytest.raises(SpotifyPodcastCacheError, match="stale"):
        SpotifyPodcastAdapter(
            str(cache), "personal", staleness_threshold=timedelta(days=1)
        ).fetch_new(None)
