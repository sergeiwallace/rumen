"""URL canonicalization funnel (design Stage 4 / D-8).

Deterministic, in order: scheme/host normalization → tracking-param stripping →
shortener/redirect resolution (injected) → publisher canonical hint (injected) →
AMP/mobile/print folding. Two pointers to the same resource collapse to one
``canonical_url``, the fallback grouping key when no stable ID exists.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = frozenset(
    {"gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src", "s", "amp"}
)
_DEFAULT_PORTS = {"http": "80", "https": "443"}

Resolver = Callable[[str], str]
CanonicalHint = Callable[[str], str | None]


def _strip_tracking(query: str) -> str:
    kept = [
        (k, v)
        for k, v in parse_qsl(query, keep_blank_values=True)
        if not (k.lower().startswith(_TRACKING_PREFIXES) or k.lower() in _TRACKING_KEYS)
    ]
    return urlencode(kept)


def _fold_amp_mobile(host: str, path: str) -> tuple[str, str]:
    host = host.removeprefix("m.")
    host = host.removeprefix("amp.")
    # Drop an /amp/ segment or a trailing /amp.
    parts = [p for p in path.split("/") if p]
    parts = [p for p in parts if p.lower() != "amp"]
    path = "/" + "/".join(parts) if parts else "/"
    return host, path


def _normalize_path(path: str) -> str:
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path or "/"


def canonicalize_url(
    url: str,
    *,
    resolve_redirect: Resolver | None = None,
    canonical_hint: CanonicalHint | None = None,
) -> str:
    """Return the canonical form of ``url`` (see module docstring for the funnel order)."""
    # (iii) shortener/redirect resolution first — operate on the terminal URL.
    if resolve_redirect is not None:
        url = resolve_redirect(url)
    # (iv) publisher-declared canonical is a strong hint, not a directive.
    if canonical_hint is not None:
        hinted = canonical_hint(url)
        if hinted:
            url = hinted

    parts = urlsplit(url)
    scheme = "https"  # (i) force https
    host = parts.hostname or ""
    host = host.removeprefix("www.")
    host, path = _fold_amp_mobile(host, parts.path)  # (v) AMP/mobile folding

    # Drop default ports (non-default ports are preserved).
    port = parts.port
    netloc = host
    if port is not None and str(port) != _DEFAULT_PORTS.get(parts.scheme or "https"):
        netloc = f"{host}:{port}"

    path = _normalize_path(path)
    query = _strip_tracking(parts.query)  # (ii) tracking-param stripping
    return urlunsplit((scheme, netloc, path, query, ""))  # fragment dropped
