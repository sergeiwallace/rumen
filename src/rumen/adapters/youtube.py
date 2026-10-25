"""Official YouTube Data API v3 playlist discovery adapter.

The transport is deliberately small and injectable.  It is the only place that
ever sees an API-key value; all errors crossing this boundary are stable,
secret-safe messages suitable for ``FeedRun.error_message``.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from rumen.adapters.base import RawItem
from rumen.net_safety import UnsafeTarget, open_public_request

_API_ROOT = "https://www.googleapis.com/youtube/v3"
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class YouTubeSourceError(RuntimeError):
    """A sanitized source-level YouTube error."""


class YouTubeApiError(YouTubeSourceError):
    """A sanitized official API error."""


class YouTubeBudgetError(YouTubeApiError):
    """The local request-unit ceiling would be exceeded."""


Transport = Callable[[str, Mapping[str, str]], tuple[int, Mapping[str, str], object]]


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise YouTubeSourceError(f"youtube malformed {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise YouTubeSourceError(f"youtube malformed {field}") from exc
    if parsed.tzinfo is None:
        raise YouTubeSourceError(f"youtube malformed {field}")
    return parsed.astimezone(UTC)


def _string(value: object, field: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value):
        raise YouTubeSourceError(f"youtube malformed {field}")
    return value


def _response_json(body: object) -> dict:
    if isinstance(body, dict):
        return body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if not isinstance(body, str):
        raise YouTubeApiError("youtube malformed response")
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise YouTubeApiError("youtube malformed response") from exc
    if not isinstance(result, dict):
        raise YouTubeApiError("youtube malformed response")
    return result


def _default_transport(
    endpoint: str, params: Mapping[str, str]
) -> tuple[int, Mapping[str, str], object]:
    request_url = f"{_API_ROOT}/{endpoint}?{urlencode(params)}"
    try:
        with open_public_request(request_url, timeout=30) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read()
    except (URLError, OSError, UnsafeTarget) as exc:
        raise YouTubeApiError("youtube transport failure") from exc


@dataclass(slots=True)
class YouTubeApiClient:
    """Narrow official API client with a local quota ceiling and bounded retries."""

    api_key_env: str
    max_api_units_per_run: int
    transport: Transport = _default_transport
    sleeper: Callable[[float], None] = time.sleep
    jitter: Callable[[], float] = random.random
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    consumed_units: int = 0

    def _request(self, endpoint: str, params: Mapping[str, str]) -> dict:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise YouTubeApiError(
                f"youtube credential environment variable is unset: {self.api_key_env}"
            )
        request_params = {**params, "key": api_key}
        for attempt in range(4):
            if self.consumed_units + 1 > self.max_api_units_per_run:
                raise YouTubeBudgetError(
                    "youtube API unit budget exceeded "
                    f"(consumed={self.consumed_units}, required=1, "
                    f"configured={self.max_api_units_per_run})"
                )
            self.consumed_units += 1
            try:
                status, headers, body = self.transport(endpoint, request_params)
            except YouTubeApiError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport seam must never leak details
                raise YouTubeApiError("youtube transport failure") from exc
            response = _response_json(body)
            reason = _api_reason(response)
            retryable = status in _RETRYABLE_STATUSES or (
                status == 403 and reason == "rateLimitExceeded"
            )
            if 200 <= status < 300:
                return response
            if retryable and attempt < 3:
                self.sleeper(_retry_delay(headers, attempt, self.jitter, self.now))
                continue
            if status == 403 and reason == "quotaExceeded":
                raise YouTubeApiError("youtube API quota exceeded")
            if status in {401, 403}:
                raise YouTubeApiError("youtube API authentication or permission failure")
            if retryable:
                raise YouTubeApiError(f"youtube {endpoint} retry exhausted")
            raise YouTubeApiError(f"youtube {endpoint} request failed ({status})")
        raise AssertionError("unreachable")

    def playlist_items(self, playlist_id: str, page_token: str | None = None) -> dict:
        params = {
            "part": "snippet,contentDetails,status",
            "playlistId": playlist_id,
            "maxResults": "50",
        }
        if page_token is not None:
            params["pageToken"] = page_token
        return self._request("playlistItems", params)

    def video_details(self, video_ids: list[str]) -> dict:
        return self._request(
            "videos",
            {
                "part": "snippet,contentDetails,status",
                "id": ",".join(video_ids),
                "maxResults": "50",
            },
        )


def _api_reason(response: Mapping[str, object]) -> str | None:
    error = response.get("error")
    if not isinstance(error, dict):
        return None
    errors = error.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        reason = errors[0].get("reason")
        return reason if isinstance(reason, str) else None
    reason = error.get("status")
    return reason if isinstance(reason, str) else None


def _retry_delay(
    headers: Mapping[str, str],
    attempt: int,
    jitter: Callable[[], float],
    now: Callable[[], datetime],
) -> float:
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, min(30.0, float(retry_after)))
        except ValueError:
            try:
                return max(
                    0.0, min(30.0, (parsedate_to_datetime(retry_after) - now()).total_seconds())
                )
            except (TypeError, ValueError):
                pass
    return min(30.0, (2**attempt) * max(0.0, min(1.0, jitter())))


@dataclass(slots=True)
class _Membership:
    playlist_item_id: str
    video_id: str
    added_at: datetime
    fallback_title: str | None


class YouTubePlaylistAdapter:
    """A complete, atomic scan of one configured public playlist."""

    def __init__(
        self,
        playlist_id: str,
        api_key_env: str,
        max_api_units_per_run: int,
        *,
        metadata_refresh_days: int = 28,
        transcript_mode: str = "off",
        transcript_languages: list[str] | None = None,
        prefer_manual_transcript: bool = True,
        client: YouTubeApiClient | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.playlist_id = playlist_id
        self.api_key_env = api_key_env
        self.max_api_units_per_run = max_api_units_per_run
        self.metadata_refresh_days = metadata_refresh_days
        self.transcript_mode = transcript_mode
        self.transcript_languages = transcript_languages or []
        self.prefer_manual_transcript = prefer_manual_transcript
        self.client = client or YouTubeApiClient(api_key_env, max_api_units_per_run)
        self.now = now
        self.last_complete_video_ids: set[str] | None = None

    def source_name(self) -> str:
        return f"youtube_playlist:{self.playlist_id}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        self.client.consumed_units = 0
        memberships = self._scan_membership()
        details = self._fetch_details([membership.video_id for membership in memberships])
        items = [
            self._to_raw(membership, details.get(membership.video_id)) for membership in memberships
        ]
        self.last_complete_video_ids = {membership.video_id for membership in memberships}
        return [item for item in items if since is None or item.published_at >= since]

    def _scan_membership(self) -> list[_Membership]:
        memberships: list[_Membership] = []
        seen_tokens: set[str] = set()
        token: str | None = None
        while True:
            envelope = self.client.playlist_items(self.playlist_id, token)
            items = envelope.get("items")
            if not isinstance(items, list):
                raise YouTubeSourceError("youtube malformed playlistItems envelope")
            for item in items:
                if not isinstance(item, dict):
                    raise YouTubeSourceError("youtube malformed playlist item")
                snippet = item.get("snippet")
                content = item.get("contentDetails")
                if not isinstance(snippet, dict) or not isinstance(content, dict):
                    raise YouTubeSourceError("youtube malformed playlist item")
                memberships.append(
                    _Membership(
                        playlist_item_id=_string(item.get("id"), "playlist item id", required=True),
                        video_id=_string(content.get("videoId"), "video id", required=True),
                        added_at=_parse_time(
                            snippet.get("publishedAt"), "playlist added timestamp"
                        ),
                        fallback_title=_string(
                            snippet.get("title"), "playlist title", required=True
                        ),
                    )
                )
            token = envelope.get("nextPageToken")
            if token is None:
                break
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise YouTubeSourceError("youtube malformed playlist page token")
            seen_tokens.add(token)
        return memberships

    def _fetch_details(self, video_ids: list[str]) -> dict[str, dict]:
        records: dict[str, dict] = {}
        for start in range(0, len(set(video_ids)), 50):
            batch = list(dict.fromkeys(video_ids))[start : start + 50]
            if not batch:
                continue
            envelope = self.client.video_details(batch)
            items = envelope.get("items")
            if not isinstance(items, list):
                raise YouTubeSourceError("youtube malformed videos envelope")
            for item in items:
                if not isinstance(item, dict):
                    raise YouTubeSourceError("youtube malformed video detail")
                video_id = _string(item.get("id"), "video detail id", required=True)
                if video_id not in batch or video_id in records:
                    raise YouTubeSourceError("youtube malformed video detail")
                if (
                    not isinstance(item.get("snippet"), dict)
                    or not isinstance(item.get("contentDetails"), dict)
                    or not isinstance(item.get("status"), dict)
                ):
                    raise YouTubeSourceError("youtube malformed video detail")
                records[video_id] = item
        return records

    def _to_raw(self, membership: _Membership, detail: dict | None) -> RawItem:
        metadata: dict[str, object] = {
            "playlist_id": self.playlist_id,
            "playlist_item_id": membership.playlist_item_id,
            "youtube_api_refreshed_at": self.now().astimezone(UTC).isoformat(),
        }
        if detail is None:
            metadata["availability"] = "unavailable"
            return RawItem(
                source=self.source_name(),
                source_id=membership.video_id,
                url=f"https://www.youtube.com/watch?v={membership.video_id}",
                title=membership.fallback_title,
                published_at=membership.added_at,
                metadata=metadata,
            )
        snippet, content, status = detail["snippet"], detail["contentDetails"], detail["status"]
        title = _string(snippet.get("title"), "video title") or membership.fallback_title
        if not title:
            raise YouTubeSourceError("youtube malformed video title")
        video_published = snippet.get("publishedAt")
        if video_published is not None:
            metadata["video_published_at"] = _parse_time(
                video_published, "video published timestamp"
            ).isoformat()
        metadata.update(
            {
                "duration": _string(content.get("duration"), "duration"),
                "channel_id": _string(snippet.get("channelId"), "channel id"),
                "caption_available": _string(status.get("caption"), "caption status") == "true",
                "availability": "available",
            }
        )
        return RawItem(
            source=self.source_name(),
            source_id=membership.video_id,
            url=f"https://www.youtube.com/watch?v={membership.video_id}",
            title=title,
            published_at=membership.added_at,
            content=_string(snippet.get("description"), "description"),
            author=_string(snippet.get("channelTitle"), "channel title"),
            metadata=metadata,
        )
