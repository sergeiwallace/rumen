"""Small, injected HTTP boundary for publisher resources only."""

from __future__ import annotations

import http.client
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from rumen.net_safety import (
    Resolver,
    UnsafeTarget,
    open_public_https,
    public_addresses,
    public_https_parts,
    validate_public_https_url,
)


class HttpError(RuntimeError):
    """A bounded publisher request failed."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes = b""


Transport = Callable[[str, dict[str, str], float, int], HttpResponse]


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def validate_publisher_url(url: str) -> None:
    try:
        parts = public_https_parts(url)
    except UnsafeTarget as exc:
        raise HttpError("publisher URL must be HTTPS without credentials or fragment") from exc
    if parts.hostname and "spotify" in parts.hostname.casefold():
        raise HttpError("Spotify content URLs are forbidden")


def _https_transport(
    url: str, headers: dict[str, str], timeout: float, ceiling: int, resolver: Resolver
) -> HttpResponse:
    try:
        with open_public_https(
            url, headers, timeout, resolver=resolver, url_policy=validate_publisher_url
        ) as (response, _final_url):
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > ceiling:
                raise HttpError(f"publisher response exceeds {ceiling} byte ceiling")
            body = response.read(ceiling + 1)
            if len(body) > ceiling:
                raise HttpError(f"publisher response exceeds {ceiling} byte ceiling")
            return HttpResponse(response.status, dict(response.headers.items()), body)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise HttpError("publisher request transport failure") from exc


class PublisherHttpClient:
    """Sequential, paced, conditional requests with a fully fakeable transport."""

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        resolver: Resolver = public_addresses,
        timeout_seconds: float = 30,
        max_attempts: int = 3,
        request_interval_seconds: float = 1,
        max_retry_after_seconds: float = 300,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0 or max_attempts < 1 or request_interval_seconds < 0:
            raise ValueError("HTTP limits must be positive (interval may be zero)")
        self.transport = transport
        self._resolver = resolver
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.request_interval_seconds = request_interval_seconds
        self.max_retry_after_seconds = max_retry_after_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request: dict[str, float] = {}
        self.retry_count = 0

    def get(
        self, url: str, *, headers: dict[str, str] | None = None, byte_ceiling: int
    ) -> HttpResponse:
        validate_publisher_url(url)
        if self.transport is not None:
            try:
                validate_public_https_url(url, self._resolver)
            except UnsafeTarget as exc:
                raise HttpError("unsafe publisher URL") from exc
        if byte_ceiling <= 0:
            raise ValueError("byte_ceiling must be positive")
        host = urlsplit(url).netloc.casefold()
        for attempt in range(1, self.max_attempts + 1):
            elapsed = self._monotonic() - self._last_request.get(host, float("-inf"))
            if elapsed < self.request_interval_seconds:
                self._sleep(self.request_interval_seconds - elapsed)
            self._last_request[host] = self._monotonic()
            try:
                response = (
                    self.transport(url, headers or {}, self.timeout_seconds, byte_ceiling)
                    if self.transport is not None
                    else _https_transport(
                        url, headers or {}, self.timeout_seconds, byte_ceiling, self._resolver
                    )
                )
            except HttpError:
                response = None
            if response is not None and response.status < 400:
                if len(response.body) > byte_ceiling:
                    raise HttpError(f"publisher response exceeds {byte_ceiling} byte ceiling")
                return response
            retryable = response is None or response.status == 429 or response.status >= 500
            if not retryable or attempt == self.max_attempts:
                code = "transport" if response is None else str(response.status)
                raise HttpError(f"publisher request failed ({code})")
            self.retry_count += 1
            retry_after = response.headers.get("Retry-After") if response else None
            try:
                delay = (
                    float(retry_after) if retry_after is not None else min(2 ** (attempt - 1), 30)
                )
            except ValueError:
                delay = min(2 ** (attempt - 1), 30)
            self._sleep(min(max(delay, 0), self.max_retry_after_seconds))
        raise AssertionError("unreachable")
