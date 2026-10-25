"""Narrow, plain-data BibTeX boundary for local Paperpile exports.

The parser is intentionally dependency-free while the optional ``paperpile`` extra is
being vetted.  It handles the exported BibTeX subset and rejects ambiguity rather than
letting parser-native objects enter the cache.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime


class BibtexParseError(ValueError):
    """Malformed/ambiguous input with bounded, content-free evidence."""


@dataclass(frozen=True, slots=True)
class PaperRecord:
    citation_key: str
    title: str
    authors: list[str]
    published_at: datetime
    date_precision: str
    identifiers: dict[str, str]
    abstract: str | None = None
    url: str | None = None


_ENTRY = re.compile(r"@(?!comment|preamble|string)([\w-]+)\s*\{\s*([^,\s]+)\s*,", re.I)
_IDENTIFIER_FIELDS = {"doi": "doi", "pmid": "pmid", "arxiv": "arxiv", "isbn": "isbn"}


def _clean(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).replace("{", "").replace("}", "").split())


def _body_at(text: str, offset: int) -> tuple[str, int]:
    depth, quoted, escaped, start = 1, False, False, offset
    for position in range(offset, len(text)):
        char = text[position]
        if quoted:
            if char == '"' and not escaped:
                quoted = False
            escaped = char == "\\" and not escaped
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:position], position + 1
    raise BibtexParseError("malformed BibTeX entry near record index")


def _fields(body: str, macros: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 0
    while index < len(body):
        while index < len(body) and body[index] in "\t\n\r ,":
            index += 1
        if index >= len(body):
            break
        match = re.match(r"([\w-]+)\s*=\s*", body[index:])
        if not match:
            raise BibtexParseError("malformed field near record index")
        name = match.group(1).lower()
        index += match.end()
        values: list[str] = []
        while True:
            if index >= len(body):
                raise BibtexParseError("unterminated field near record index")
            if body[index] == "{":
                value, index = _body_at(body, index + 1)
            elif body[index] == '"':
                end = index + 1
                while end < len(body) and body[end] != '"':
                    end += 1
                if end == len(body):
                    raise BibtexParseError("unterminated quoted field near record index")
                value, index = body[index + 1 : end], end + 1
            else:
                token = re.match(r"[^,#\s]+", body[index:])
                if not token:
                    raise BibtexParseError("invalid field value near record index")
                value, index = token.group(0), index + token.end()
                value = macros.get(value.lower(), value)
            values.append(value)
            while index < len(body) and body[index].isspace():
                index += 1
            if index >= len(body) or body[index] != "#":
                break
            index += 1
            while index < len(body) and body[index].isspace():
                index += 1
        result[name] = _clean("".join(values))
    return result


def _date(fields: dict[str, str], observed_at: datetime) -> tuple[datetime, str]:
    value = fields.get("date")
    if value and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return datetime.fromisoformat(value).replace(tzinfo=UTC), "date"
    year = fields.get("year") or (value[:4] if value and re.match(r"\d{4}", value) else None)
    if year and re.fullmatch(r"\d{4}", year):
        return datetime(int(year), 1, 1, tzinfo=UTC), "year"
    return observed_at, "observed"


def parse_bibtex(text: str, *, observed_at: datetime) -> list[PaperRecord]:
    """Parse a complete export, rejecting duplicate keys and strong-id conflicts."""
    observed_at = observed_at.astimezone(UTC)
    entries: list[PaperRecord] = []
    macros: dict[str, str] = {}
    for macro in re.finditer(r'@string\s*\{\s*([\w-]+)\s*=\s*[\{"]([^\}"]+)[\}"]\s*\}', text, re.I):
        macros[macro.group(1).lower()] = _clean(macro.group(2))
    seen_keys: set[str] = set()
    seen_identifiers: dict[tuple[str, str], str] = {}
    cursor = 0
    while match := _ENTRY.search(text, cursor):
        key = _clean(match.group(2))
        body, cursor = _body_at(text, match.end())
        if key in seen_keys:
            raise BibtexParseError("duplicate citation key at record index")
        fields = _fields(body, macros)
        title = _clean(fields.get("title", ""))
        if not title:
            raise BibtexParseError("missing title at record index")
        identifiers = {
            target: _clean(fields[source]).lower() if target == "doi" else _clean(fields[source])
            for source, target in _IDENTIFIER_FIELDS.items()
            if _clean(fields.get(source, ""))
        }
        for kind, value in identifiers.items():
            prior = seen_identifiers.setdefault((kind, value), key)
            if prior != key:
                raise BibtexParseError("strong identifier maps to multiple records")
        published_at, precision = _date(fields, observed_at)
        entries.append(
            PaperRecord(
                citation_key=key,
                title=title,
                authors=[
                    _clean(author)
                    for author in fields.get("author", "").split(" and ")
                    if _clean(author)
                ],
                published_at=published_at,
                date_precision=precision,
                identifiers=identifiers,
                abstract=_clean(fields["abstract"]) if fields.get("abstract") else None,
                url=fields.get("url"),
            )
        )
        seen_keys.add(key)
    if not entries or text[cursor:].strip():
        raise BibtexParseError("malformed or empty BibTeX input")
    return entries
