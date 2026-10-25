"""Publisher transcript selection, format parsing, provenance, and policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rumen.podcasts.transcripts import (
    TranscriptCandidate,
    TranscriptError,
    normalize_transcript,
    select_candidate,
    validate_candidate,
)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "spotify_podcasts"


@pytest.mark.parametrize(
    ("mime", "body", "expected"),
    [
        ("text/plain", b"Example Speaker: hello\nworld\n", "hello\nworld"),
        ("text/vtt", (_FIXTURES / "transcript.vtt").read_bytes(), "Hello world\nSecond line"),
        ("application/x-subrip", (_FIXTURES / "transcript.srt").read_bytes(), "Hello\nWorld"),
    ],
)
def test_given_supported_transcript_formats_when_normalized_then_preserves_text_and_segments(
    mime: str, body: bytes, expected: str
) -> None:
    normalized = normalize_transcript(body, mime)
    assert normalized.plain_text == expected
    assert normalized.segments[0].speaker in {"Example Speaker", "Speaker One", None}
    assert len(normalized.segments) >= 1


def test_given_candidates_when_selecting_then_uses_language_then_publisher_then_feed_order() -> (
    None
):
    candidates = [
        TranscriptCandidate(
            "https://publisher.example/generated", "text/vtt", "en", generated=True
        ),
        TranscriptCandidate("https://publisher.example/publisher", "text/vtt", "en"),
        TranscriptCandidate("https://publisher.example/fr", "text/vtt", "fr"),
    ]
    assert select_candidate(candidates, "en").url == "https://publisher.example/publisher"
    assert select_candidate(candidates, "fr").url == "https://publisher.example/fr"


@pytest.mark.parametrize(
    "candidate",
    [
        TranscriptCandidate("ftp://publisher.example/x", "text/plain"),
        TranscriptCandidate("https://publisher.example/x", "application/json"),
        TranscriptCandidate("https://open.spotify.com/x", "text/plain"),
    ],
)
def test_given_disallowed_candidate_when_validated_then_fails_without_text(
    candidate: TranscriptCandidate,
) -> None:
    with pytest.raises(TranscriptError):
        validate_candidate(candidate)


def test_given_invalid_cue_when_normalized_then_fails_without_partial_ready_text() -> None:
    with pytest.raises(TranscriptError, match="cue"):
        normalize_transcript("WEBVTT\n\nnot a cue", "text/vtt")
