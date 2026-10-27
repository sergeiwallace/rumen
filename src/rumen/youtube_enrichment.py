"""Database-aware YouTube metadata lifecycle and optional transcript enrichment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.adapters.youtube import YouTubePlaylistAdapter
from rumen.grouping.merge import requeue_source_refresh
from rumen.models import FeedItem, YouTubeVideoState


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class TranscriptTrack:
    language: str
    kind: str
    text: str


class TranscriptProvider(Protocol):
    """Small provider boundary: list tracks and return their validated text."""

    name: str
    version: str

    def list_tracks(self, video_id: str) -> Iterable[TranscriptTrack]: ...


class TranscriptNotFoundError(RuntimeError):
    pass


class TranscriptTransientError(RuntimeError):
    pass


class TranscriptBlockedError(RuntimeError):
    pass


class YouTubeTranscriptApiProvider:
    """Pinned optional ``youtube-transcript-api`` wrapper; imported only when enabled."""

    name = "youtube-transcript-api"
    version = "1.2.4"

    def list_tracks(self, video_id: str) -> Iterable[TranscriptTrack]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:  # pragma: no cover - exercised through package install
            raise TranscriptBlockedError("provider_unavailable") from exc
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            tracks = []
            for transcript in transcript_list:
                fetched = transcript.fetch()
                text = " ".join(part.text for part in fetched).strip()
                if not text:
                    raise TranscriptBlockedError("provider_contract")
                tracks.append(
                    TranscriptTrack(
                        language=transcript.language_code,
                        kind="generated" if transcript.is_generated else "manual",
                        text=text,
                    )
                )
            return tracks
        except (TranscriptNotFoundError, TranscriptTransientError, TranscriptBlockedError):
            raise
        except Exception as exc:  # package exception taxonomy is intentionally unstable
            name = type(exc).__name__.lower()
            if "notranscript" in name or "transcriptsdisabled" in name:
                raise TranscriptNotFoundError("not_found") from exc
            if "requestblocked" in name or "ipblocked" in name:
                raise TranscriptBlockedError("blocked") from exc
            raise TranscriptTransientError("provider_transient") from exc


@dataclass(slots=True)
class YouTubeMaintenanceResult:
    states_created: int = 0
    metadata_refreshed: int = 0
    metadata_failures: int = 0
    removals: int = 0
    transcript_created: int = 0
    transcript_not_found: int = 0
    transcript_retries: int = 0
    transcript_blocked: int = 0
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def reconcile_youtube_discovery(
    session: Session, adapter: YouTubePlaylistAdapter, *, now: datetime | None = None
) -> YouTubeMaintenanceResult:
    """Create state for persisted discovery pointers and reconcile only complete scans."""
    now = now or _utcnow()
    result = YouTubeMaintenanceResult()
    if adapter.last_complete_video_ids is None:
        return result
    source_items = list(
        session.scalars(select(FeedItem).where(FeedItem.source == adapter.source_name()))
    )
    existing = {
        state.video_id: state
        for state in session.scalars(
            select(YouTubeVideoState).where(YouTubeVideoState.playlist_id == adapter.playlist_id)
        )
    }
    for item in source_items:
        state = existing.get(item.source_id)
        if state is None:
            state = YouTubeVideoState(
                feed_item_id=item.id,
                playlist_id=adapter.playlist_id,
                video_id=item.source_id,
                membership_status="current"
                if item.source_id in adapter.last_complete_video_ids
                else "removed",
                last_seen_in_playlist_at=now
                if item.source_id in adapter.last_complete_video_ids
                else None,
                metadata_refreshed_at=now,
                transcript_status="pending"
                if adapter.transcript_mode == "unofficial_captions"
                else "disabled",
            )
            session.add(state)
            result.states_created += 1
        elif item.source_id in adapter.last_complete_video_ids:
            state.membership_status = "current"
            state.last_seen_in_playlist_at = now
    for video_id, state in existing.items():
        if video_id not in adapter.last_complete_video_ids and state.membership_status != "removed":
            state.membership_status = "removed"
            result.removals += 1
    session.commit()
    return result


def run_youtube_maintenance(
    session: Session, adapter: YouTubePlaylistAdapter, *, now: datetime | None = None
) -> YouTubeMaintenanceResult:
    """Refresh due provider metadata while leaving a failed batch independently retryable."""
    now = now or _utcnow()
    result = YouTubeMaintenanceResult()
    due_before = now - timedelta(days=adapter.metadata_refresh_days)
    states = list(
        session.scalars(
            select(YouTubeVideoState).where(
                YouTubeVideoState.playlist_id == adapter.playlist_id,
                (YouTubeVideoState.metadata_refreshed_at.is_(None))
                | (YouTubeVideoState.metadata_refreshed_at <= due_before),
            )
        )
    )
    for start in range(0, len(states), 50):
        batch = states[start : start + 50]
        try:
            envelope = adapter.client.video_details([state.video_id for state in batch])
            details = _details_by_id(envelope)
        except Exception:  # noqa: BLE001 - stored error code must be secret-safe and stable
            for state in batch:
                state.metadata_refresh_error = "youtube_refresh_failed"
                item = session.get(FeedItem, state.feed_item_id)
                if item is not None and _metadata_is_expired(state, now):
                    _clear_expired_provider_fields(session, item, state)
            result.metadata_failures += len(batch)
            session.commit()
            continue
        for state in batch:
            detail = details.get(state.video_id)
            if detail is None:
                state.metadata_refresh_error = "youtube_detail_unavailable"
                item = session.get(FeedItem, state.feed_item_id)
                if item is not None and _metadata_is_expired(state, now):
                    _clear_expired_provider_fields(session, item, state)
                continue
            item = session.get(FeedItem, state.feed_item_id)
            if item is None:
                continue
            changed = _apply_detail(item, detail, now)
            state.metadata_refreshed_at = now
            state.metadata_refresh_error = None
            if changed:
                requeue_source_refresh(session, item)
            result.metadata_refreshed += 1
        session.commit()
    return result


def _details_by_id(envelope: object) -> dict[str, dict]:
    if not isinstance(envelope, dict) or not isinstance(envelope.get("items"), list):
        raise ValueError("malformed detail envelope")
    details: dict[str, dict] = {}
    for detail in envelope["items"]:
        if not isinstance(detail, dict) or not isinstance(detail.get("id"), str):
            raise ValueError("malformed detail")
        if not all(
            isinstance(detail.get(key), dict) for key in ("snippet", "contentDetails", "status")
        ):
            raise ValueError("malformed detail")
        details[detail["id"]] = detail
    return details


def _apply_detail(item: FeedItem, detail: dict, now: datetime) -> bool:
    snippet, content, status = detail["snippet"], detail["contentDetails"], detail["status"]
    title = snippet.get("title")
    if not isinstance(title, str) or not title:
        raise ValueError("malformed title")
    description = snippet.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("malformed description")
    metadata = dict(item.link_metadata or {})
    metadata.update(
        {
            "duration": content.get("duration")
            if isinstance(content.get("duration"), str)
            else None,
            "channel_id": snippet.get("channelId")
            if isinstance(snippet.get("channelId"), str)
            else None,
            "caption_available": status.get("caption") == "true",
            "availability": "available",
            "youtube_api_refreshed_at": now.astimezone(UTC).isoformat(),
        }
    )
    if isinstance(snippet.get("publishedAt"), str):
        metadata["video_published_at"] = snippet["publishedAt"]
    changed = item.title != title or item.content != description
    item.title, item.content = title, description
    item.author = (
        snippet.get("channelTitle") if isinstance(snippet.get("channelTitle"), str) else None
    )
    item.link_metadata = metadata
    return changed


def _metadata_is_expired(state: YouTubeVideoState, now: datetime) -> bool:
    return (
        state.metadata_refreshed_at is not None
        and state.metadata_refreshed_at <= now - timedelta(days=30)
    )


def _clear_expired_provider_fields(
    session: Session, item: FeedItem, state: YouTubeVideoState
) -> None:
    """Retain only durable Rumen curation facts after the API-data retention window."""
    item.title = f"YouTube video {state.video_id}"
    item.content = None
    item.author = None
    item.link_metadata = {
        "playlist_id": state.playlist_id,
        "availability": "unavailable",
    }
    requeue_source_refresh(session, item)


def run_transcript_enrichment(
    session: Session,
    adapter: YouTubePlaylistAdapter,
    provider: TranscriptProvider,
    *,
    now: datetime | None = None,
    retry_limit: int = 3,
    recheck_days: int = 7,
) -> YouTubeMaintenanceResult:
    """Create a sibling pointer for each due selected caption without touching discovery."""
    now = now or _utcnow()
    result = YouTubeMaintenanceResult()
    if adapter.transcript_mode != "unofficial_captions":
        return result
    due = list(
        session.scalars(
            select(YouTubeVideoState).where(
                YouTubeVideoState.playlist_id == adapter.playlist_id,
                YouTubeVideoState.transcript_status.in_(("pending", "retry", "not_found")),
                (YouTubeVideoState.transcript_next_attempt_at.is_(None))
                | (YouTubeVideoState.transcript_next_attempt_at <= now),
            )
        )
    )
    for state in due:
        try:
            track = _select_track(
                provider.list_tracks(state.video_id),
                adapter.transcript_languages,
                adapter.prefer_manual_transcript,
            )
            if track is None:
                raise TranscriptNotFoundError("not_found")
        except TranscriptNotFoundError:
            state.transcript_status = "not_found"
            state.transcript_next_attempt_at = now + timedelta(days=recheck_days)
            state.last_error_code = "transcript_not_found"
            result.transcript_not_found += 1
        except TranscriptBlockedError as exc:
            state.transcript_status = "blocked"
            state.last_error_code = str(exc) or "transcript_blocked"
            result.transcript_blocked += 1
            result.warnings.append(f"youtube transcript blocked for {adapter.playlist_id}")
        except Exception:  # noqa: BLE001 - provider boundary
            state.transcript_attempts += 1
            state.last_error_code = "transcript_transient"
            if state.transcript_attempts >= retry_limit:
                state.transcript_status = "blocked"
                result.transcript_blocked += 1
            else:
                state.transcript_status = "retry"
                state.transcript_next_attempt_at = now + timedelta(
                    hours=2**state.transcript_attempts
                )
                result.transcript_retries += 1
        else:
            item = session.get(FeedItem, state.feed_item_id)
            source = f"youtube_transcript:{adapter.playlist_id}"
            transcript = session.scalar(
                select(FeedItem).where(
                    FeedItem.source == source, FeedItem.source_id == state.video_id
                )
            )
            if transcript is None:
                transcript = FeedItem(
                    source=source,
                    source_id=state.video_id,
                    url=item.url,
                    title=item.title,
                    content=track.text,
                    author=item.author,
                    published_at=item.published_at,
                    link_metadata={
                        "playlist_id": adapter.playlist_id,
                        "provider": provider.name,
                        "provider_version": provider.version,
                        "language": track.language,
                        "kind": track.kind,
                        "fetched_at": now.astimezone(UTC).isoformat(),
                    },
                )
                session.add(transcript)
                session.flush()
                result.transcript_created += 1
            state.transcript_status = "succeeded"
            state.transcript_feed_item_id = transcript.id
            state.transcript_provider = f"{provider.name}:{provider.version}"
            state.transcript_language = track.language
            state.transcript_kind = track.kind
            state.transcript_fetched_at = now
            state.transcript_next_attempt_at = None
            state.last_error_code = None
        session.commit()
    return result


def _select_track(
    tracks: Iterable[TranscriptTrack], languages: list[str], prefer_manual: bool
) -> TranscriptTrack | None:
    available = list(tracks)
    for language in languages:
        matches = [
            track for track in available if track.language == language and track.text.strip()
        ]
        if matches:
            if prefer_manual:
                return next((track for track in matches if track.kind == "manual"), matches[0])
            return matches[0]
    return None
