"""YouTube state migration, refresh, and complete-snapshot reconciliation coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from rumen.adapters.youtube import YouTubeApiClient, YouTubePlaylistAdapter
from rumen.db.migrate import downgrade_to_base, upgrade_to_head
from rumen.models import FeedItem, YouTubeVideoState
from rumen.youtube_enrichment import (
    reconcile_youtube_discovery,
    run_youtube_maintenance,
)

_NOW = datetime(2026, 9, 3, tzinfo=UTC)


def _item(session, video_id="video-1"):
    item = FeedItem(
        source="youtube_playlist:playlist-a",
        source_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        title="old",
        content="old content",
        published_at=_NOW - timedelta(days=30),
    )
    session.add(item)
    session.flush()
    return item


def _adapter(monkeypatch, response):
    monkeypatch.setenv("YT_KEY", "not-for-output")
    client = YouTubeApiClient("YT_KEY", 100, transport=lambda _e, _p: (200, {}, response))
    return YouTubePlaylistAdapter(
        "playlist-a", "YT_KEY", 100, client=client, metadata_refresh_days=1
    )


def test_given_duplicate_youtube_state_when_persisting_then_database_rejects_it(session) -> None:
    item = _item(session)
    session.add_all(
        [
            YouTubeVideoState(feed_item_id=item.id, playlist_id="playlist-a", video_id="video-1"),
            YouTubeVideoState(feed_item_id=item.id, playlist_id="playlist-b", video_id="video-2"),
        ]
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_given_sqlite_migration_round_trip_when_downgrading_then_removes_only_youtube_state(
    sqlite_engine,
) -> None:
    upgrade_to_head(sqlite_engine)
    assert "youtube_video_states" in inspect(sqlite_engine).get_table_names()

    downgrade_to_base(sqlite_engine)
    assert "youtube_video_states" not in inspect(sqlite_engine).get_table_names()


def test_given_due_metadata_when_maintenance_runs_then_refreshes_and_requeues_machine_state(
    session, monkeypatch
) -> None:
    item = _item(session)
    item.category = "machine"
    item.classification_method = "hmc"
    item.embedding = b"machine"
    state = YouTubeVideoState(
        feed_item_id=item.id,
        playlist_id="playlist-a",
        video_id="video-1",
        metadata_refreshed_at=_NOW - timedelta(days=2),
    )
    session.add(state)
    session.commit()
    adapter = _adapter(
        monkeypatch,
        {
            "items": [
                {
                    "id": "video-1",
                    "snippet": {
                        "title": "new",
                        "description": "new content",
                        "channelTitle": "new channel",
                        "channelId": "channel",
                    },
                    "contentDetails": {"duration": "PT3M"},
                    "status": {"caption": "true"},
                }
            ]
        },
    )

    result = run_youtube_maintenance(session, adapter, now=_NOW)
    refreshed = session.get(FeedItem, item.id)

    assert result.metadata_refreshed == 1
    assert refreshed.title == "new"
    assert refreshed.content == "new content"
    assert refreshed.category is None
    assert refreshed.embedding is None
    assert session.get(YouTubeVideoState, state.id).metadata_refreshed_at == _NOW


def test_given_refresh_failure_when_maintenance_runs_then_preserves_item_and_records_sanitized_error(
    session, monkeypatch
) -> None:
    item = _item(session)
    state = YouTubeVideoState(feed_item_id=item.id, playlist_id="playlist-a", video_id="video-1")
    session.add(state)
    session.commit()
    monkeypatch.setenv("YT_KEY", "secret-canary")
    client = YouTubeApiClient(
        "YT_KEY", 100, transport=lambda _e, _p: (_ for _ in ()).throw(RuntimeError("secret-canary"))
    )
    adapter = YouTubePlaylistAdapter(
        "playlist-a", "YT_KEY", 100, client=client, metadata_refresh_days=1
    )

    result = run_youtube_maintenance(session, adapter, now=_NOW)

    assert result.metadata_failures == 1
    assert session.get(FeedItem, item.id).title == "old"
    assert (
        session.get(YouTubeVideoState, state.id).metadata_refresh_error == "youtube_refresh_failed"
    )


def test_given_api_data_expired_for_30_days_when_refresh_fails_then_clears_mutable_fields(
    session, monkeypatch
) -> None:
    item = _item(session)
    state = YouTubeVideoState(
        feed_item_id=item.id,
        playlist_id="playlist-a",
        video_id="video-1",
        metadata_refreshed_at=_NOW - timedelta(days=31),
    )
    session.add(state)
    session.commit()
    adapter = _adapter(monkeypatch, {"items": []})

    run_youtube_maintenance(session, adapter, now=_NOW)

    cleared = session.get(FeedItem, item.id)
    assert cleared.title == "YouTube video video-1"
    assert cleared.content is None
    assert cleared.author is None
    assert cleared.link_metadata == {"playlist_id": "playlist-a", "availability": "unavailable"}


def test_given_complete_snapshots_when_reconciling_then_archives_removals_and_restores_readded_pointer(
    session,
) -> None:
    item = _item(session)
    state = YouTubeVideoState(
        feed_item_id=item.id,
        playlist_id="playlist-a",
        video_id="video-1",
        membership_status="current",
    )
    session.add(state)
    session.commit()
    adapter = YouTubePlaylistAdapter("playlist-a", "YT_KEY", 100)
    adapter.last_complete_video_ids = set()

    removed = reconcile_youtube_discovery(session, adapter, now=_NOW)
    assert removed.removals == 1
    assert session.get(YouTubeVideoState, state.id).membership_status == "removed"
    assert session.get(FeedItem, item.id) is not None

    adapter.last_complete_video_ids = {"video-1"}
    reconcile_youtube_discovery(session, adapter, now=_NOW + timedelta(days=1))
    assert session.get(YouTubeVideoState, state.id).membership_status == "current"


def test_given_no_verified_complete_snapshot_when_reconciling_then_does_not_change_membership(
    session,
) -> None:
    item = _item(session)
    state = YouTubeVideoState(
        feed_item_id=item.id,
        playlist_id="playlist-a",
        video_id="video-1",
        membership_status="current",
    )
    session.add(state)
    session.commit()
    adapter = YouTubePlaylistAdapter("playlist-a", "YT_KEY", 100)

    reconcile_youtube_discovery(session, adapter, now=_NOW)

    assert session.get(YouTubeVideoState, state.id).membership_status == "current"
