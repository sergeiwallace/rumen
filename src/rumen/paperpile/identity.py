"""Precision-first exact aliases for Paperpile snapshot records."""

from __future__ import annotations

from dataclasses import dataclass

from rumen.paperpile.parsers import PaperRecord


@dataclass(frozen=True, slots=True)
class IdentityResolution:
    work_id: str | None
    aliases: set[tuple[str, str]]
    quarantine_reason: str | None = None


def aliases_for(record: PaperRecord, hashes: list[str] | None = None) -> set[tuple[str, str]]:
    aliases = {("citation_key", record.citation_key)}
    aliases.update((kind, value) for kind, value in record.identifiers.items())
    aliases.update(("pdf_sha256", value) for value in hashes or [])
    return aliases


def resolve_exact(
    aliases: set[tuple[str, str]], known: dict[tuple[str, str], str]
) -> IdentityResolution:
    matches = {known[alias] for alias in aliases if alias in known}
    if len(matches) > 1:
        return IdentityResolution(None, aliases, "conflicting exact aliases")
    return IdentityResolution(next(iter(matches), None), aliases)
