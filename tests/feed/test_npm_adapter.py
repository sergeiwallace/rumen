"""T-03: NpmRegistryAdapter — parses a captured registry fixture (AD-3, Claude Code)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rumen.adapters.npm_registry import NpmRegistryAdapter

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "npm_claude_code.json"


def _adapter() -> NpmRegistryAdapter:
    payload = _FIXTURE.read_text()
    return NpmRegistryAdapter(
        "@anthropic-ai/claude-code", "Claude Code", fetcher=lambda _url: payload
    )


def test_fetch_new_when_since_none_then_one_item_per_version() -> None:
    items = _adapter().fetch_new(None)
    assert {i.source_id for i in items} == {"1.2.0", "1.2.3"}
    latest = next(i for i in items if i.source_id == "1.2.3")
    assert latest.source == "npm_registry:@anthropic-ai/claude-code"
    assert latest.title == "Claude Code 1.2.3"
    assert latest.published_at == datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    assert latest.metadata["package"] == "@anthropic-ai/claude-code"


def test_fetch_new_excludes_created_and_modified_keys() -> None:
    items = _adapter().fetch_new(None)
    assert "created" not in {i.source_id for i in items}
    assert "modified" not in {i.source_id for i in items}


def test_fetch_new_when_no_new_versions_then_empty() -> None:
    since = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    assert _adapter().fetch_new(since) == []


def test_fetch_new_when_since_between_versions_then_only_newer() -> None:
    since = datetime(2026, 7, 5, tzinfo=UTC)
    items = _adapter().fetch_new(since)
    assert [i.source_id for i in items] == ["1.2.3"]
