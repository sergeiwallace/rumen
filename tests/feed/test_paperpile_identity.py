from datetime import UTC, datetime

from rumen.paperpile.identity import aliases_for, resolve_exact
from rumen.paperpile.parsers import PaperRecord


def _record(doi: str = "10.1/a") -> PaperRecord:
    return PaperRecord("cite", "Paper", [], datetime(2026, 1, 1, tzinfo=UTC), "year", {"doi": doi})


def test_identity_when_exact_alias_gained_then_reuses_work_and_adds_alias() -> None:
    aliases = aliases_for(_record(), ["a" * 64])
    result = resolve_exact(aliases, {("doi", "10.1/a"): "work-1"})
    assert result.work_id == "work-1"
    assert ("pdf_sha256", "a" * 64) in result.aliases


def test_identity_when_exact_aliases_conflict_then_quarantines_without_merge() -> None:
    result = resolve_exact(
        aliases_for(_record()), {("doi", "10.1/a"): "one", ("citation_key", "cite"): "two"}
    )
    assert result.work_id is None
    assert result.quarantine_reason == "conflicting exact aliases"
