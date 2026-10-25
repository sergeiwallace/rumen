"""Publisher transcript selection and normalization shared by RSS and local ASR paths."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rumen.podcasts.http import HttpError, validate_publisher_url


class TranscriptError(ValueError):
    """A transcript candidate cannot safely produce normalized text."""


SUPPORTED_MIME_TYPES = frozenset(
    {"text/plain", "text/vtt", "application/x-subrip", "application/srt", "text/srt"}
)


@dataclass(frozen=True)
class TranscriptCandidate:
    url: str
    mime_type: str
    language: str | None = None
    generated: bool = False


@dataclass(frozen=True)
class TranscriptSegment:
    start_ms: int | None
    end_ms: int | None
    speaker: str | None
    text: str


@dataclass(frozen=True)
class NormalizedTranscript:
    plain_text: str
    segments: list[TranscriptSegment]


def select_candidate(
    candidates: list[TranscriptCandidate], preferred_language: str | None
) -> TranscriptCandidate | None:
    supported = [
        candidate
        for candidate in candidates
        if candidate.mime_type.casefold() in SUPPORTED_MIME_TYPES
    ]
    if not supported:
        return None
    preferred = preferred_language.casefold() if preferred_language else None
    return min(
        enumerate(supported),
        key=lambda pair: (
            0 if preferred and (pair[1].language or "").casefold() == preferred else 1,
            1 if pair[1].generated else 0,
            pair[0],
        ),
    )[1]


def validate_candidate(candidate: TranscriptCandidate) -> None:
    try:
        validate_publisher_url(candidate.url)
    except HttpError as exc:
        raise TranscriptError(str(exc)) from exc
    if candidate.mime_type.casefold() not in SUPPORTED_MIME_TYPES:
        raise TranscriptError(f"unsupported transcript MIME type {candidate.mime_type!r}")


_TIME = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:[.,](?P<ms>\d{1,3}))?")
_CUE = re.compile(
    r"^\s*(?P<start>\d{2}:\d{2}:\d{2}[.,]\d{1,3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[.,]\d{1,3})"
)


def _millis(value: str) -> int:
    match = _TIME.fullmatch(value)
    if match is None:
        raise TranscriptError("invalid transcript timecode")
    return (
        int(match["h"]) * 3_600_000
        + int(match["m"]) * 60_000
        + int(match["s"]) * 1_000
        + int((match["ms"] or "0").ljust(3, "0"))
    )


def _speaker_and_text(text: str) -> tuple[str | None, str]:
    vtt = re.fullmatch(r"<v\s+([^>]+)>(.*?)(?:</v>)?", text.strip(), flags=re.DOTALL)
    if vtt:
        return vtt.group(1).strip(), vtt.group(2).strip()
    named = re.fullmatch(r"([A-Za-z][\w .'-]{0,80}):\s*(.+)", text.strip(), flags=re.DOTALL)
    return (named.group(1), named.group(2).strip()) if named else (None, text.strip())


def _cue_segments(text: str, *, vtt: bool) -> list[TranscriptSegment]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if vtt:
        if not lines or not lines[0].lstrip("\ufeff").startswith("WEBVTT"):
            raise TranscriptError("invalid WebVTT header")
        lines = lines[1:]
    segments: list[TranscriptSegment] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        cue = _CUE.match(line)
        if cue is None and not vtt and index + 1 < len(lines):
            index += 1  # optional SRT numeric ordinal
            cue = _CUE.match(lines[index].strip())
        if cue is None:
            raise TranscriptError("invalid transcript cue")
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip():
            body.append(lines[index].strip())
            index += 1
        speaker, cue_text = _speaker_and_text(" ".join(body))
        if not cue_text:
            raise TranscriptError("empty transcript cue")
        segments.append(
            TranscriptSegment(_millis(cue["start"]), _millis(cue["end"]), speaker, cue_text)
        )
    if not segments:
        raise TranscriptError("transcript has no cues")
    return segments


def normalize_transcript(body: bytes | str, mime_type: str) -> NormalizedTranscript:
    try:
        text = body.decode("utf-8") if isinstance(body, bytes) else body
    except UnicodeDecodeError as exc:
        raise TranscriptError("transcript is not UTF-8") from exc
    normalized_mime = mime_type.casefold().split(";", 1)[0].strip()
    if normalized_mime == "text/plain":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise TranscriptError("plain transcript is empty")
        segments = [TranscriptSegment(None, None, *_speaker_and_text(line)) for line in lines]
    elif normalized_mime == "text/vtt":
        segments = _cue_segments(text, vtt=True)
    elif normalized_mime in {"application/x-subrip", "application/srt", "text/srt"}:
        segments = _cue_segments(text, vtt=False)
    else:
        raise TranscriptError(f"unsupported transcript MIME type {mime_type!r}")
    plain_text = "\n".join(segment.text for segment in segments).strip()
    if not plain_text:
        raise TranscriptError("transcript is empty")
    return NormalizedTranscript(plain_text, segments)
