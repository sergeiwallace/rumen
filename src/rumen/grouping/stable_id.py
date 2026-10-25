"""Work-level stable-ID extraction + alias resolution (design Stage 4 / D-8).

Extracts a version-stripped, work-level identity from a URL — the confident merge key
(precision > canonical URL). Kinds: ``github_repo``, ``arxiv``, ``doi``, ``youtube``,
``isbn``. The ``stable_id_alias`` map is applied first (GitHub renames, preprint↔published
DOI). PDF SHA-256 needs the file body and is handled at content time, not from a URL.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.models import StableIdAlias

_URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:a-z0-9]+)", re.IGNORECASE)
_ISBN_RE = re.compile(r"(97[89]\d{10})")


def extract_stable_id(url: str | None) -> tuple[str | None, str | None]:
    """Return ``(kind, value)`` for the URL's work-level identity, or ``(None, None)``."""
    if not url:
        return None, None
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    host = host.removeprefix("www.")
    path = parts.path

    if host in {"github.com", "raw.githubusercontent.com"}:
        segments = [s for s in path.split("/") if s]
        if len(segments) >= 2:
            owner, repo = segments[0], segments[1]
            repo = repo.removesuffix(".git")
            return "github_repo", f"{owner}/{repo}"

    if "arxiv.org" in host:
        match = _ARXIV_RE.search(path)
        if match:
            return "arxiv", match.group(1)  # version-stripped

    if host in {"youtube.com", "m.youtube.com"}:
        vid = parse_qs(parts.query).get("v", [None])[0]
        if vid:
            return "youtube", vid
    if host == "youtu.be":
        vid = path.strip("/").split("/")[0]
        if vid:
            return "youtube", vid

    doi = _DOI_RE.search(url)
    if doi:
        return "doi", doi.group(1).lower()

    isbn = _ISBN_RE.search(url)
    if isbn:
        return "isbn", isbn.group(1)

    return None, None


def stable_id_from_content(
    text: str | None, link_metadata: dict | None = None
) -> tuple[str | None, str | None]:
    """Scan free text for the first URL that yields a stable ID (a tweet/HN post linking
    a repo/paper carries its identity in the linked URL, not its own URL)."""
    urls: list[str] = []
    if link_metadata:
        for key in ("github_urls", "links"):
            links = link_metadata.get(key, [])
            if isinstance(links, list):
                urls.extend(str(link) for link in links)
    urls.extend(_URL_RE.findall(text or ""))
    for candidate in urls:
        kind, value = extract_stable_id(candidate)
        if value:
            return kind, value
    return None, None


def resolve_alias(
    session: Session, kind: str | None, value: str | None
) -> tuple[str | None, str | None]:
    """Map ``(kind, value)`` through the ``stable_id_alias`` table to its survivor id."""
    if kind is None or value is None:
        return kind, value
    to_value = session.scalars(
        select(StableIdAlias.to_value).where(
            StableIdAlias.kind == kind, StableIdAlias.from_value == value
        )
    ).first()
    return (kind, to_value) if to_value else (kind, value)


AliasMap = dict[tuple[str, str], str]


def load_alias_map(session: Session) -> AliasMap:
    """Load the whole ``stable_id_alias`` table once, keyed ``(kind, from_value) → to_value``.

    Grouping resolves every pointer's identity against this in-memory map instead of one
    query per pointer — the difference between a single round-trip and thousands when
    grouping against a remote Postgres.
    """
    rows = session.execute(
        select(StableIdAlias.kind, StableIdAlias.from_value, StableIdAlias.to_value)
    ).all()
    return {(kind, from_value): to_value for kind, from_value, to_value in rows}


def apply_alias(
    alias_map: AliasMap, kind: str | None, value: str | None
) -> tuple[str | None, str | None]:
    """In-memory equivalent of ``resolve_alias`` using a preloaded ``load_alias_map`` result."""
    if kind is None or value is None:
        return kind, value
    return kind, alias_map.get((kind, value), value)
