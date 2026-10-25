"""Public HTTPS requests pinned to validated addresses, without proxy routing."""

from __future__ import annotations

import http.client
import io
import ipaddress
import socket
import ssl
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import SplitResult, urljoin, urlsplit
from urllib.request import Request

from rumen import __version__

Resolver = Callable[[str], list[str]]


class UnsafeTarget(ValueError):
    """A URL or redirect cannot safely be fetched."""

    def __init__(self, message: str, reason: str = "unsafe_target") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PublicTarget:
    host: str
    port: int
    path: str
    addresses: tuple[str, ...]


def public_addresses(host: str) -> list[str]:
    return list(
        dict.fromkeys(row[4][0] for row in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM))
    )


def public_https_parts(url: str) -> SplitResult:
    """Reject ambiguous URLs before parsing can discard controls or credentials."""
    try:
        if any(ord(char) <= 32 or ord(char) == 127 for char in url):
            raise ValueError("URL contains whitespace or controls")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "%" in parsed.hostname
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError("URL must be public HTTPS without credentials or fragment")
        return parsed
    except ValueError as exc:
        raise UnsafeTarget("URL must be HTTPS without credentials or fragment") from exc


def validate_public_https_url(url: str, resolver: Resolver = public_addresses) -> PublicTarget:
    """Resolve once and retain only numeric public addresses for the connection."""
    parsed = public_https_parts(url)
    try:
        try:
            addresses = [str(ipaddress.ip_address(parsed.hostname))]
        except ValueError:
            addresses = resolver(parsed.hostname)
        if not addresses:
            raise ValueError("host has no addresses")
        numeric = []
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global or ip.is_multicast or ip.is_reserved or "%" in address:
                raise ValueError("host has a non-public address")
            numeric.append(str(ip))
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        return PublicTarget(parsed.hostname, parsed.port or 443, path, tuple(numeric))
    except (OSError, ValueError) as exc:
        raise UnsafeTarget("URL must resolve only to public HTTPS addresses") from exc


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, target: PublicTarget, timeout: float) -> None:
        # HTTPSConnection keeps the original host for SNI and certificate checks.
        super().__init__(
            target.host, target.port, timeout=timeout, context=ssl.create_default_context()
        )
        self._addresses = target.addresses
        self._create_connection = self._connect_validated

    def _connect_validated(self, address, timeout, source_address=None):
        # socket.create_connection would resolve again, reopening the DNS race.
        for index, ip in enumerate(self._addresses):
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                sock.settimeout(timeout)
                sock.connect((ip, self.port))
                return sock
            except OSError:
                sock.close()
                if index == len(self._addresses) - 1:
                    raise
        raise OSError("no validated address")


@contextmanager
def open_public_https(
    url: str,
    headers: dict[str, str],
    timeout: float,
    *,
    resolver: Resolver = public_addresses,
    url_policy: Callable[[str], None] | None = None,
    method: str = "GET",
    body: bytes | None = None,
) -> Iterator[tuple[http.client.HTTPResponse, str]]:
    """Validate each redirect before opening a direct, hostname-verified connection."""
    headers = dict(headers)
    if not any(name.lower() == "user-agent" for name in headers):
        headers["User-Agent"] = f"rumen/{__version__}"
    current = url
    for _ in range(6):
        if url_policy is not None:
            url_policy(current)
        target = validate_public_https_url(current, resolver)
        connection = _PinnedHTTPSConnection(target, timeout)
        try:
            connection.request(method, target.path, body=body, headers=headers)
            response = connection.getresponse()
            try:
                if response.status not in {301, 302, 303, 307, 308}:
                    yield response, current
                    return
                location = response.getheader("Location")
                if not location:
                    raise UnsafeTarget("redirect lacks location", "redirect_failure")
                current = urljoin(current, location)
            finally:
                response.close()
        finally:
            connection.close()
    raise UnsafeTarget("request exceeded redirect limit", "redirect_limit")


@contextmanager
def open_public_request(request: str | Request, *, timeout: float = 30):
    """Service requests with pinned connections and same-origin redirects.

    Keep credentials and POST bodies on their configured origin. Non-GET
    redirects are refused rather than replaying a potentially sensitive body.
    """
    if isinstance(request, str):
        request = Request(request)
    origin = public_https_parts(request.full_url).netloc

    def check_redirect(url: str) -> None:
        if public_https_parts(url).netloc != origin:
            raise UnsafeTarget("cross-origin service redirects are disabled")
        if url != request.full_url and request.get_method() != "GET":
            raise UnsafeTarget("non-GET service redirects are disabled")

    with open_public_https(
        request.full_url,
        dict(request.header_items()),
        timeout,
        url_policy=check_redirect,
        method=request.get_method(),
        body=request.data,
    ) as (response, _):
        if response.status >= 400:
            raise HTTPError(
                "",
                response.status,
                "service request failed",
                response.headers,
                io.BytesIO(response.read()),
            )
        yield response
