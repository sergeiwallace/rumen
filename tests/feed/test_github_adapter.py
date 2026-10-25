"""GitHubReleasesAdapter parses a synthetic Atom fixture, filters by ``since``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rumen.adapters.github_releases import GitHubReleasesAdapter

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "github_releases.atom"


def _adapter() -> GitHubReleasesAdapter:
    atom = _FIXTURE.read_text()
    return GitHubReleasesAdapter("example-user-1/example-sdk", fetcher=lambda _url: atom)


def test_fetch_new_when_since_none_then_returns_all_entries_as_rawitems() -> None:
    items = _adapter().fetch_new(None)
    assert len(items) == 2
    latest = items[0]
    assert latest.source == "github_releases:example-user-1/example-sdk"
    assert latest.source_id == "v0.30.0"
    assert latest.url.endswith("/releases/tag/v0.30.0")
    assert latest.author == "example-user-1"
    assert "streaming" in latest.content
    assert latest.published_at == datetime(2026, 7, 10, 12, 0, tzinfo=UTC)


def test_fetch_new_when_since_set_then_returns_only_newer_entries() -> None:
    since = datetime(2026, 7, 6, tzinfo=UTC)
    items = _adapter().fetch_new(since)
    assert [i.source_id for i in items] == ["v0.30.0"]


def test_fetch_new_when_since_at_latest_then_returns_empty() -> None:
    since = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    assert _adapter().fetch_new(since) == []


def test_source_name_is_repo_qualified() -> None:
    # Per-repo identity (cursor + dedup namespace), not the bare type.
    assert _adapter().source_name() == "github_releases:example-user-1/example-sdk"
