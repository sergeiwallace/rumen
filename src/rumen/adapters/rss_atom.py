"""Bounded, replay-safe RSS 2.0 and Atom 1.0 ingestion adapter."""

from __future__ import annotations

import http.client
import io
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urljoin, urlsplit

from rumen.adapters.base import FetchResult, RawItem, StatefulFetchError
from rumen.net_safety import (
    UnsafeTarget,
    open_public_https,
    public_addresses,
    validate_public_https_url,
)

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_MAX_ENTRIES = 1000
_MAX_TEXT_LENGTH = 16_000
_MAX_METADATA_LENGTH = 4_096
_TIMEOUT_SECONDS = 30
_USER_AGENT = "rumen/rss-atom"


def require_feedparser() -> None:
    """Keep the parser dependency optional until an RSS source is configured."""
    try:
        import feedparser  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("RSS/Atom support requires `pip install rumen[feeds]`") from exc


@dataclass(slots=True)
class FeedResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    final_url: str


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "object", "embed", "iframe"}:
            self._ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "object", "embed", "iframe"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)


def _plain_text(value: object, limit: int = _MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        return ""
    parser = _PlainText()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())[:limit]


def _header(headers: dict[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower() and isinstance(value, str):
            return value[:_MAX_METADATA_LENGTH]
    return None


def _validate_public_https_url(url: str, resolver: Callable[[str], list[str]]) -> None:
    try:
        validate_public_https_url(url, resolver)
    except UnsafeTarget as exc:
        raise StatefulFetchError(str(exc), {"reason": exc.reason}) from exc


def _default_transport(
    url: str, headers: dict[str, str], timeout: float, resolver: Callable[[str], list[str]]
) -> FeedResponse:
    if not any(name.lower() == "user-agent" for name in headers):
        headers = {"User-Agent": _USER_AGENT, **headers}
    try:
        with open_public_https(url, headers, timeout, resolver=resolver) as (response, final_url):
            response_headers = dict(response.headers.items())
            body = _read_limited(response, _MAX_RESPONSE_BYTES) if response.status < 400 else b""
            body = _decode_limited(body, _header(response_headers, "Content-Encoding"))
            return FeedResponse(response.status, response_headers, body, final_url)
    except UnsafeTarget as exc:
        raise StatefulFetchError(str(exc), {"reason": exc.reason}) from exc
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise StatefulFetchError("feed transport failed", {"reason": "transport_failure"}) from exc


def _read_limited(response: object, limit: int) -> bytes:
    read = getattr(response, "read")
    body = read(limit + 1)
    if len(body) > limit:
        raise StatefulFetchError(
            "feed response exceeds byte limit", {"reason": "response_too_large"}
        )
    return body


def _decode_limited(body: bytes, content_encoding: str | None) -> bytes:
    """Decode supported compressed responses without returning an oversized body."""
    if content_encoding in (None, "identity"):
        return body
    if content_encoding == "gzip":
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif content_encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        raise StatefulFetchError(
            "unsupported feed content encoding", {"reason": "decompression_failure"}
        )
    try:
        decoded = decompressor.decompress(body, _MAX_RESPONSE_BYTES + 1)
        if len(decoded) > _MAX_RESPONSE_BYTES or decompressor.unconsumed_tail:
            raise StatefulFetchError(
                "feed response exceeds byte limit", {"reason": "response_too_large"}
            )
        decoded += decompressor.flush(_MAX_RESPONSE_BYTES + 1 - len(decoded))
    except zlib.error as exc:
        raise StatefulFetchError(
            "feed decompression failed", {"reason": "decompression_failure"}
        ) from exc
    if len(decoded) > _MAX_RESPONSE_BYTES:
        raise StatefulFetchError(
            "feed response exceeds byte limit", {"reason": "response_too_large"}
        )
    return decoded


class RssAtomAdapter:
    """One independently stateful, explicitly configured public feed."""

    def __init__(
        self,
        profile: str,
        feed_url: str,
        min_poll_interval_seconds: int = 900,
        *,
        transport: Callable[[str, dict[str, str], float], FeedResponse] | None = None,
        resolver: Callable[[str], list[str]] = public_addresses,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        require_feedparser()
        self.profile = profile
        self.feed_url = feed_url
        self.min_poll_interval_seconds = min_poll_interval_seconds
        self._resolver = resolver
        self._transport = transport
        self._clock = clock

    def source_name(self) -> str:
        return f"rss_atom:{self.profile}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        return self.fetch_new_with_state(since, None).items

    def fetch_new_with_state(self, since: datetime | None, state: dict | None) -> FetchResult:
        del since  # Feed windows are replayed; DB identity deduplication is the cursor.
        if self._transport is not None:
            _validate_public_https_url(self.feed_url, self._resolver)
        previous = self._validate_state(state)
        now = self._clock().astimezone(UTC)
        if previous and self._not_due(previous, now):
            return FetchResult([], previous, {"reason": "not_due"})

        headers: dict[str, str] = {}
        if previous and previous.get("configured_url") == self.feed_url:
            if etag := previous.get("etag"):
                headers["If-None-Match"] = etag
            if last_modified := previous.get("last_modified"):
                headers["If-Modified-Since"] = last_modified
        response = (
            self._transport(self.feed_url, headers, _TIMEOUT_SECONDS)
            if self._transport is not None
            else _default_transport(self.feed_url, headers, _TIMEOUT_SECONDS, self._resolver)
        )
        if not isinstance(response, FeedResponse):
            raise StatefulFetchError(
                "feed transport returned an invalid response", {"reason": "transport_failure"}
            )
        if self._transport is not None:
            _validate_public_https_url(response.final_url, self._resolver)
        if len(response.body) > _MAX_RESPONSE_BYTES:
            raise StatefulFetchError(
                "feed response exceeds byte limit", {"reason": "response_too_large"}
            )
        next_state = self._next_state(previous, response, now)
        if response.status == 304:
            return FetchResult([], next_state, {"reason": "not_modified"})
        if response.status != 200:
            diagnostics = {"reason": f"http_{response.status}"}
            if response.status in {429, 503} and (
                retry := _retry_after(_header(response.headers, "Retry-After"))
            ):
                diagnostics["retry_after_seconds"] = retry
            raise StatefulFetchError("feed request was not successful", diagnostics)
        items, diagnostics = self._parse(response.body, response.final_url)
        next_state["last_changed_at"] = now.isoformat()
        return FetchResult(items, next_state, diagnostics)

    def _validate_state(self, state: dict | None) -> dict | None:
        if state is None:
            return None
        if not isinstance(state, dict) or state.get("version") != 1:
            raise StatefulFetchError(
                "unsupported RSS/Atom source state", {"reason": "invalid_state"}
            )
        for key in (
            "etag",
            "last_modified",
            "configured_url",
            "final_url",
            "checked_at",
            "last_changed_at",
        ):
            if key in state and (
                not isinstance(state[key], str) or len(state[key]) > _MAX_METADATA_LENGTH
            ):
                raise StatefulFetchError(
                    "invalid RSS/Atom source state", {"reason": "invalid_state"}
                )
        return dict(state)

    def _not_due(self, state: dict, now: datetime) -> bool:
        checked_at = state.get("checked_at")
        if not checked_at:
            return False
        try:
            checked = datetime.fromisoformat(checked_at).astimezone(UTC)
        except ValueError as exc:
            raise StatefulFetchError(
                "invalid RSS/Atom source state", {"reason": "invalid_state"}
            ) from exc
        return now - checked < timedelta(seconds=self.min_poll_interval_seconds)

    def _next_state(self, previous: dict | None, response: FeedResponse, now: datetime) -> dict:
        state = {
            "version": 1,
            "configured_url": self.feed_url,
            "final_url": response.final_url,
            "checked_at": now.isoformat(),
        }
        for key, header_name in (("etag", "ETag"), ("last_modified", "Last-Modified")):
            value = _header(response.headers, header_name)
            if value is not None:
                state[key] = value
            elif previous and key in previous:
                state[key] = previous[key]
        if previous and previous.get("last_changed_at"):
            state["last_changed_at"] = previous["last_changed_at"]
        return state

    def _parse(self, body: bytes, final_url: str) -> tuple[list[RawItem], dict]:
        require_feedparser()
        import feedparser

        parsed = feedparser.parse(
            io.BytesIO(body), response_headers={"content-location": final_url}
        )
        if parsed.get("version") not in {"rss20", "atom10"}:
            raise StatefulFetchError("unrecognized RSS/Atom feed", {"reason": "unrecognized_feed"})
        diagnostics: dict = {"skipped": {}}
        if parsed.get("bozo"):
            exc = parsed.get("bozo_exception")
            if type(exc).__name__ not in {"CharacterEncodingOverride", "NonXMLContentType"}:
                raise StatefulFetchError(
                    "feed parser rejected document", {"reason": "parser_error"}
                )
            diagnostics["recoverable_parser_warning"] = type(exc).__name__
        entries = parsed.get("entries", [])
        if not isinstance(entries, list) or len(entries) > _MAX_ENTRIES:
            raise StatefulFetchError("feed entry limit exceeded", {"reason": "entry_limit"})
        items: list[RawItem] = []
        seen: dict[str, RawItem] = {}
        for entry in entries:
            item, reason = self._map_entry(entry, final_url, diagnostics)
            if item is None:
                skipped = diagnostics["skipped"]
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            original = seen.get(item.source_id)
            if original is not None:
                if (original.url, original.published_at, original.content) != (
                    item.url,
                    item.published_at,
                    item.content,
                ):
                    raise StatefulFetchError(
                        "ambiguous duplicate feed identity", {"reason": "identity_ambiguous"}
                    )
                continue
            seen[item.source_id] = item
            items.append(item)
        if entries and not items:
            raise StatefulFetchError("feed has no valid entries", {"reason": "no_valid_entries"})
        return items, diagnostics

    def _map_entry(
        self, entry: object, final_url: str, diagnostics: dict
    ) -> tuple[RawItem | None, str]:
        if not isinstance(entry, dict):
            return None, "invalid_entry"
        identity = _plain_text(entry.get("guid") or entry.get("id"))
        url = self._entry_url(entry, final_url)
        if not identity:
            identity = url or ""
        if not identity:
            return None, "missing_identity"
        if not url:
            return None, "missing_url"
        title = _plain_text(entry.get("title"))
        content = self._content(entry)
        if not title:
            title = content
        if not title:
            return None, "missing_title"
        published_at, original_published, original_updated = self._timestamp(entry)
        if published_at is None:
            return None, "missing_timestamp"
        categories: list[str] = []
        for tag in entry.get("tags", []):
            if (
                isinstance(tag, dict)
                and (term := _plain_text(tag.get("term") or tag.get("label"), _MAX_METADATA_LENGTH))
                and term not in categories
            ):
                categories.append(term)
        metadata = {
            "configured_feed_url": self.feed_url,
            "final_feed_url": final_url,
            "original_published": original_published,
            "original_updated": original_updated,
            "identity_field": "guid_or_id" if entry.get("guid") or entry.get("id") else "url",
            "enclosures": self._enclosures(entry, final_url),
        }
        if diagnostics.get("recoverable_parser_warning"):
            metadata["parser_diagnostic"] = diagnostics["recoverable_parser_warning"]
        author = (
            entry.get("author_detail", {}).get("name")
            if isinstance(entry.get("author_detail"), dict)
            else entry.get("author")
        )
        return RawItem(
            source=self.source_name(),
            source_id=identity[:_MAX_METADATA_LENGTH],
            url=url,
            title=title,
            content=content or None,
            author=_plain_text(author) or None,
            published_at=published_at,
            categories=categories or None,
            metadata=metadata,
        ), ""

    @staticmethod
    def _entry_url(entry: dict, final_url: str) -> str | None:
        links = entry.get("links", [])
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and link.get("rel", "alternate") == "alternate":
                    href = link.get("href")
                    if isinstance(href, str):
                        resolved = urljoin(final_url, href)
                        if len(resolved) <= _MAX_METADATA_LENGTH and urlsplit(resolved).scheme in {
                            "http",
                            "https",
                        }:
                            return resolved
        href = entry.get("link")
        if isinstance(href, str):
            resolved = urljoin(final_url, href)
            if len(resolved) <= _MAX_METADATA_LENGTH and urlsplit(resolved).scheme in {
                "http",
                "https",
            }:
                return resolved
        return None

    @staticmethod
    def _content(entry: dict) -> str:
        values = entry.get("content")
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict) and (text := _plain_text(value.get("value"))):
                    return text
        return _plain_text(entry.get("summary") or entry.get("description"))

    @staticmethod
    def _timestamp(entry: dict) -> tuple[datetime | None, str | None, str | None]:
        original_published = (
            entry.get("published") if isinstance(entry.get("published"), str) else None
        )
        original_updated = entry.get("updated") if isinstance(entry.get("updated"), str) else None
        for parsed_key in ("published_parsed", "updated_parsed"):
            value = entry.get(parsed_key)
            if value:
                return datetime(*value[:6], tzinfo=UTC), original_published, original_updated
        return None, original_published, original_updated

    @staticmethod
    def _enclosures(entry: dict, final_url: str) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for enclosure in entry.get("enclosures", []):
            if not isinstance(enclosure, dict):
                continue
            descriptor = {
                key: str(value)[:_MAX_METADATA_LENGTH]
                for key, value in enclosure.items()
                if key in {"href", "type", "length"} and value is not None
            }
            if "href" in descriptor:
                descriptor["href"] = urljoin(final_url, descriptor["href"])
            if descriptor:
                result.append(descriptor)
        return result[:20]


def _retry_after(value: str | None) -> int | None:
    if value is None:
        return None
    if value.isdigit():
        return min(int(value), 86_400)
    try:
        seconds = int(
            (parsedate_to_datetime(value).astimezone(UTC) - datetime.now(UTC)).total_seconds()
        )
    except (TypeError, ValueError):
        return None
    return min(max(seconds, 0), 86_400)
