"""Fixture-backed publisher RSS identity and bounded HTTP behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rumen.podcasts.http import HttpError, HttpResponse, PublisherHttpClient
from rumen.podcasts.rss import episode_key, parse_feed, resolve_episode
from rumen.podcasts.sync import synchronize

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "spotify_podcasts"
_FEED = (_FIXTURES / "feed.xml").read_bytes()
_HISTORY = _FIXTURES / "history.json"
_VTT = (_FIXTURES / "transcript.vtt").read_bytes()


def test_given_feed_when_parsed_then_resolves_exact_or_explicit_alias_and_rejects_ambiguity() -> (
    None
):
    episodes = parse_feed(_FEED)
    exact = resolve_episode(
        "spotify:episode:one", "  episode   one ", "https://publisher.example/feed", episodes, {}
    )
    alias = resolve_episode(
        "spotify:episode:one",
        "different",
        "https://publisher.example/feed",
        episodes,
        {"spotify:episode:one": "guid-one"},
    )
    ambiguous = resolve_episode(
        "spotify:episode:two", "Ambiguous", "https://publisher.example/feed", episodes, {}
    )

    assert (exact.method, alias.method, ambiguous.reason) == (
        "exact-normalized-title",
        "explicit-alias",
        "zero-or-ambiguous-title-match",
    )
    assert episode_key("https://publisher.example/feed", exact.episode) is not None


def test_given_episode_without_publisher_identity_when_resolved_then_quarantines() -> None:
    feed = b"<rss><channel><title>S</title><item><title>No ID</title><pubDate>Fri, 01 Aug 2026 12:00:00 +0000</pubDate></item></channel></rss>"
    result = resolve_episode(
        "spotify:episode:x", "No ID", "https://publisher.example/feed", parse_feed(feed), {}
    )
    assert result.reason == "episode-has-no-canonical-identity"


def test_given_conditional_response_and_retry_when_fetching_then_preserves_headers_and_bounds_attempts() -> (
    None
):
    calls: list[dict[str, str]] = []
    responses = [HttpResponse(503, {}), HttpResponse(200, {"ETag": "new"}, b"ok")]
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda _url, headers, _timeout, _limit: calls.append(headers) or responses.pop(0),
        request_interval_seconds=0,
        sleep=lambda _seconds: None,
    )
    response = client.get(
        "https://publisher.example/feed", headers={"If-None-Match": "old"}, byte_ceiling=10
    )
    assert (response.body, client.retry_count, calls) == (b"ok", 1, [{"If-None-Match": "old"}] * 2)


def test_given_oversized_or_invalid_feed_when_fetching_then_does_not_replace_cache(
    tmp_path: Path,
) -> None:
    manifest = {
        "cache_path": str(tmp_path / "cache.db"),
        "exports": [str(_HISTORY)],
        "feeds": [{"show": "Test Show", "url": "https://publisher.example/feed"}],
        "limits": {"feed_byte_ceiling": 3},
    }
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda *_args: HttpResponse(200, {}, _FEED),
        request_interval_seconds=0,
    )
    result = synchronize(manifest, client=client)
    assert result["failure"] == 1


def test_given_mapped_show_when_syncing_then_requests_only_publisher_feed_and_never_event_data(
    tmp_path: Path,
) -> None:
    requested: list[str] = []
    manifest = {
        "cache_path": str(tmp_path / "cache.db"),
        "exports": [str(_HISTORY)],
        "feeds": [{"show": "Test Show", "url": "https://publisher.example/feed"}],
    }

    def transport(url, _headers, _timeout, _limit):
        requested.append(url)
        return HttpResponse(200, {}, _FEED if url.endswith("feed") else _VTT)

    result = synchronize(
        manifest,
        client=PublisherHttpClient(
            transport=transport, resolver=lambda _host: ["8.8.8.8"], request_interval_seconds=0
        ),
    )
    assert result["publisher_transcript"] == 1
    assert requested == ["https://publisher.example/feed", "https://publisher.example/one.vtt"]


def test_given_synthetic_ready_and_ambiguous_events_when_syncing_then_emits_one_and_quarantines_one(
    tmp_path: Path,
) -> None:
    history = json.loads(_HISTORY.read_text())
    history.append(
        {
            "ts": "2026-08-02T12:00:00Z",
            "spotify_episode_uri": "spotify:episode:ambiguous",
            "episode_name": "Ambiguous",
            "episode_show_name": "Test Show",
            "ms_played": 1,
        }
    )
    export = tmp_path / "history.json"
    export.write_text(json.dumps(history))
    manifest = {
        "cache_path": str(tmp_path / "cache.db"),
        "exports": [str(export)],
        "feeds": [{"show": "Test Show", "url": "https://publisher.example/feed"}],
    }
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda url, *_args: HttpResponse(
            200, {}, _FEED if url.endswith("feed") else _VTT
        ),
        request_interval_seconds=0,
    )

    result = synchronize(manifest, client=client)

    assert (result["publisher_transcript"], result["quarantined"]) == (1, 1)


def test_given_spotify_content_url_when_requested_then_rejects_policy_boundary() -> None:
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda *_args: pytest.fail("transport must not run"),
        request_interval_seconds=0,
    )
    with pytest.raises(HttpError, match="forbidden"):
        client.get("https://open.spotify.com/transcript", byte_ceiling=10)
