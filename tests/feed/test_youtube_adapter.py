"""Fixture-driven official YouTube playlist discovery coverage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rumen.adapters.youtube import (
    YouTubeApiClient,
    YouTubeApiError,
    YouTubeBudgetError,
    YouTubePlaylistAdapter,
    YouTubeSourceError,
)
from rumen.config import load_config
from rumen.ingest import run_ingestion

_FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/youtube/playlist_pages.json").read_text()
)
_NOW = datetime(2026, 9, 3, tzinfo=UTC)


def _adapter(
    monkeypatch, *, budget: int = 10
) -> tuple[YouTubePlaylistAdapter, list[tuple[str, dict]]]:
    monkeypatch.setenv("YT_TEST_KEY", "secret-canary-do-not-print")
    calls: list[tuple[str, dict]] = []

    def transport(endpoint, params):
        calls.append((endpoint, dict(params)))
        if endpoint == "playlistItems":
            page = params.get("pageToken", "first")
            return 200, {}, _FIXTURE["pages"][page]
        requested = params["id"].split(",")
        return (
            200,
            {},
            {"items": [item for item in _FIXTURE["details"]["items"] if item["id"] in requested]},
        )

    client = YouTubeApiClient("YT_TEST_KEY", budget, transport=transport)
    return YouTubePlaylistAdapter(
        "playlist-a", "YT_TEST_KEY", budget, client=client, now=lambda: _NOW
    ), calls


def test_given_fixture_pages_when_fetching_then_scans_all_pages_joins_by_id_and_replays_boundary(
    monkeypatch,
) -> None:
    adapter, calls = _adapter(monkeypatch)
    boundary = datetime(2026, 9, 1, 12, tzinfo=UTC)

    items = adapter.fetch_new(boundary)

    assert [item.source_id for item in items] == ["video-1", "video-2", "video-3"]
    assert items[0].title == "Video one"
    assert items[2].metadata["availability"] == "unavailable"
    assert items[2].content is None
    assert [call[0] for call in calls] == ["playlistItems", "playlistItems", "videos"]
    assert calls[-1][1]["id"] == "video-1,video-2,video-3"


def test_given_replayed_youtube_boundary_when_ingesting_then_existing_pointer_is_not_duplicated(
    session, monkeypatch
) -> None:
    adapter, _ = _adapter(monkeypatch)
    first = run_ingestion(session, adapter)
    second = run_ingestion(session, adapter)

    assert first.items_passed_filter == 3
    assert second.items_passed_filter == 0
    assert second.high_water_mark == first.high_water_mark


def test_given_malformed_page_or_detail_when_fetching_then_raises_without_partial_results(
    monkeypatch,
) -> None:
    monkeypatch.setenv("YT_TEST_KEY", "secret-canary-do-not-print")
    client = YouTubeApiClient(
        "YT_TEST_KEY", 10, transport=lambda _endpoint, _params: (200, {}, {"items": "bad"})
    )
    adapter = YouTubePlaylistAdapter("playlist-a", "YT_TEST_KEY", 10, client=client)

    with pytest.raises(YouTubeSourceError, match="malformed"):
        adapter.fetch_new(None)
    assert adapter.last_complete_video_ids is None


def test_given_missing_credential_or_error_canary_when_fetching_then_error_is_secret_safe(
    monkeypatch,
) -> None:
    monkeypatch.delenv("YT_TEST_KEY", raising=False)
    client = YouTubeApiClient("YT_TEST_KEY", 10)

    with pytest.raises(YouTubeApiError) as error:
        client.playlist_items("playlist-a")

    assert "YT_TEST_KEY" in str(error.value)
    assert "secret-canary" not in str(error.value)


def test_given_retryable_response_when_requesting_then_honors_retry_after_without_sleeping(
    monkeypatch,
) -> None:
    monkeypatch.setenv("YT_TEST_KEY", "secret-canary-do-not-print")
    responses = iter([(429, {"Retry-After": "3"}, {"error": {}}), (200, {}, {"items": []})])
    delays: list[float] = []
    client = YouTubeApiClient(
        "YT_TEST_KEY", 10, transport=lambda _e, _p: next(responses), sleeper=delays.append
    )

    assert client.playlist_items("playlist-a") == {"items": []}
    assert delays == [3.0]
    assert client.consumed_units == 2


def test_given_budget_exhaustion_when_requesting_then_stops_before_transport_and_is_safe(
    monkeypatch,
) -> None:
    monkeypatch.setenv("YT_TEST_KEY", "secret-canary-do-not-print")
    client = YouTubeApiClient("YT_TEST_KEY", 0, transport=lambda _e, _p: pytest.fail("called"))

    with pytest.raises(YouTubeBudgetError) as error:
        client.playlist_items("playlist-a")

    assert "secret-canary" not in str(error.value)
    assert "consumed=0" in str(error.value)


def test_given_retry_when_budget_is_consumed_then_stops_before_the_second_attempt(
    monkeypatch,
) -> None:
    monkeypatch.setenv("YT_TEST_KEY", "secret-canary-do-not-print")
    calls = []
    client = YouTubeApiClient(
        "YT_TEST_KEY",
        1,
        transport=lambda _e, _p: calls.append(1) or (429, {}, {"error": {}}),
        sleeper=lambda _delay: None,
    )

    with pytest.raises(YouTubeBudgetError):
        client.playlist_items("playlist-a")

    assert calls == [1]


@pytest.mark.parametrize(
    "source",
    [
        {
            "type": "youtube_playlist",
            "playlist_id": "",
            "api_key_env": "YT_KEY",
            "max_api_units_per_run": 1,
        },
        {
            "type": "youtube_playlist",
            "playlist_id": "p",
            "api_key_env": "bad-name",
            "max_api_units_per_run": 1,
        },
        {
            "type": "youtube_playlist",
            "playlist_id": "p",
            "api_key_env": "YT_KEY",
            "max_api_units_per_run": True,
        },
        {
            "type": "youtube_playlist",
            "playlist_id": "p",
            "api_key_env": "YT_KEY",
            "max_api_units_per_run": 1,
            "transcript_mode": "off",
            "transcript_languages": ["en"],
        },
    ],
)
def test_given_invalid_youtube_config_when_loading_then_fails_before_requests(
    tmp_path, source
) -> None:
    path = tmp_path / "rumen.json"
    path.write_text(
        json.dumps({"sources": [source], "classifier_escalation_alarm_threshold": 0.25})
    )
    with pytest.raises(ValueError):
        load_config(path)


def test_given_valid_youtube_config_when_loading_then_builds_adapter_without_reading_secret(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("YT_KEY", raising=False)
    path = tmp_path / "rumen.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "type": "youtube_playlist",
                        "playlist_id": "p",
                        "api_key_env": "YT_KEY",
                        "max_api_units_per_run": 1,
                    }
                ],
                "classifier_escalation_alarm_threshold": 0.25,
            }
        )
    )

    [adapter] = load_config(path).adapters

    assert adapter.source_name() == "youtube_playlist:p"
    assert adapter.api_key_env == "YT_KEY"
