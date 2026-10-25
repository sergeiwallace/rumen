"""Optional transcript selection, state transitions, and sibling-pointer coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rumen.adapters.youtube import YouTubePlaylistAdapter
from rumen.grouping import group_pending
from rumen.models import FeedItem, YouTubeVideoState
from rumen.youtube_enrichment import (
    TranscriptBlockedError,
    TranscriptNotFoundError,
    TranscriptTrack,
    TranscriptTransientError,
    YouTubeTranscriptApiProvider,
    run_transcript_enrichment,
)

_NOW = datetime(2026, 9, 3, tzinfo=UTC)


class _Provider:
    name = "fixture-provider"
    version = "1"

    def __init__(self, result):
        self.result = result

    def list_tracks(self, _video_id):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _state(session):
    item = FeedItem(
        source="youtube_playlist:playlist-a",
        source_id="video-1",
        url="https://www.youtube.com/watch?v=video-1",
        title="metadata",
        content="description",
        published_at=_NOW,
    )
    session.add(item)
    session.flush()
    state = YouTubeVideoState(
        feed_item_id=item.id,
        playlist_id="playlist-a",
        video_id="video-1",
        transcript_status="pending",
    )
    session.add(state)
    session.commit()
    return item, state


def _adapter():
    return YouTubePlaylistAdapter(
        "playlist-a",
        "YT_KEY",
        10,
        transcript_mode="unofficial_captions",
        transcript_languages=["fr", "en"],
    )


def test_given_requested_tracks_when_enriching_then_prefers_manual_and_creates_one_groupable_sibling(
    session,
) -> None:
    item, state = _state(session)
    provider = _Provider(
        [
            TranscriptTrack("en", "generated", "generated text"),
            TranscriptTrack("en", "manual", "manual text"),
        ]
    )

    result = run_transcript_enrichment(session, _adapter(), provider, now=_NOW)
    transcript = session.query(FeedItem).filter_by(source="youtube_transcript:playlist-a").one()
    group_pending(session, embed=False)

    assert result.transcript_created == 1
    assert transcript.content == "manual text"
    assert transcript.link_metadata["provider"] == "fixture-provider"
    assert session.get(YouTubeVideoState, state.id).transcript_status == "succeeded"
    assert session.get(FeedItem, item.id).topic_entity_id == transcript.topic_entity_id
    assert session.query(FeedItem).count() == 2


def test_given_replayed_success_when_enriching_then_keeps_one_transcript_pointer(session) -> None:
    _state(session)
    provider = _Provider([TranscriptTrack("en", "manual", "text")])
    adapter = _adapter()

    run_transcript_enrichment(session, adapter, provider, now=_NOW)
    run_transcript_enrichment(session, adapter, provider, now=_NOW + timedelta(days=1))

    assert session.query(FeedItem).filter_by(source="youtube_transcript:playlist-a").count() == 1


def test_given_absent_transient_or_blocked_captions_when_enriching_then_records_isolated_state(
    session,
) -> None:
    for provider_error, expected in [
        (TranscriptNotFoundError("none"), "not_found"),
        (TranscriptTransientError("later"), "retry"),
        (TranscriptBlockedError("blocked"), "blocked"),
    ]:
        # A distinct source partition gives each loop a fresh lifecycle row.
        item = FeedItem(
            source=f"youtube_playlist:p-{expected}",
            source_id="video-1",
            url="https://www.youtube.com/watch?v=video-1",
            title="metadata",
            published_at=_NOW,
        )
        session.add(item)
        session.flush()
        state = YouTubeVideoState(
            feed_item_id=item.id,
            playlist_id=f"p-{expected}",
            video_id="video-1",
            transcript_status="pending",
        )
        session.add(state)
        session.commit()
        adapter = YouTubePlaylistAdapter(
            f"p-{expected}",
            "YT_KEY",
            10,
            transcript_mode="unofficial_captions",
            transcript_languages=["en"],
        )
        run_transcript_enrichment(session, adapter, _Provider(provider_error), now=_NOW)
        assert session.get(YouTubeVideoState, state.id).transcript_status == expected


@pytest.mark.parametrize("generated", [False, True])
def test_given_real_provider_transcript_when_wrapped_then_preserves_text_language_and_kind(
    monkeypatch,
    generated,
) -> None:
    api = pytest.importorskip("youtube_transcript_api")
    from requests import Response, Session
    from youtube_transcript_api._transcripts import Transcript, TranscriptList

    response = Response()
    response.status_code = 200
    response._content = (
        b'<transcript><text start="0" dur="1">first</text>'
        b'<text start="1" dur="1">second</text></transcript>'
    )
    monkeypatch.setattr(Session, "get", lambda *args, **kwargs: response)
    track = Transcript(
        Session(), "video-1", "https://captions.example/text", "English", "en", generated, []
    )
    transcript_list = TranscriptList(
        "video-1", {} if generated else {"en": track}, {"en": track} if generated else {}, []
    )
    monkeypatch.setattr(api.YouTubeTranscriptApi, "list", lambda self, video_id: transcript_list)

    tracks = list(YouTubeTranscriptApiProvider().list_tracks("video-1"))

    assert tracks == [
        TranscriptTrack(
            language="en", kind="generated" if generated else "manual", text="first second"
        )
    ]


@pytest.mark.parametrize(
    ("provider_error", "expected", "code"),
    [
        ("TranscriptsDisabled", TranscriptNotFoundError, "not_found"),
        ("NoTranscriptFound", TranscriptNotFoundError, "not_found"),
        ("RequestBlocked", TranscriptBlockedError, "blocked"),
        ("IpBlocked", TranscriptBlockedError, "blocked"),
        ("VideoUnavailable", TranscriptTransientError, "provider_transient"),
    ],
)
def test_given_real_provider_error_when_wrapped_then_preserves_error_classification(
    monkeypatch, provider_error, expected, code
) -> None:
    api = pytest.importorskip("youtube_transcript_api")
    error_class = getattr(api, provider_error)
    error = (
        error_class("video-1", ["en"], [])
        if provider_error == "NoTranscriptFound"
        else error_class("video-1")
    )

    def fail(self, video_id):
        raise error

    monkeypatch.setattr(api.YouTubeTranscriptApi, "list", fail)
    with pytest.raises(expected, match=f"^{code}$"):
        list(YouTubeTranscriptApiProvider().list_tracks("video-1"))
