"""T-03: ingestion orchestration — high-water-mark, idempotency, failure isolation.

Runs against both dialects via the ``session`` fixture."""

from __future__ import annotations

import json
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from rumen.adapters.base import FetchResult, RawItem, StatefulFetchError
from rumen.adapters.chrome_bookmarks import ChromeBookmarksAdapter
from rumen.adapters.github_releases import GitHubReleasesAdapter
from rumen.adapters.github_repository import GitHubRepositoryAdapter
from rumen.adapters.github_stars import GitHubStarredAdapter
from rumen.ingest import last_completed_context, last_high_water_mark, run_ingestion
from rumen.models import FeedItem, FeedRun

_ATOM = (Path(__file__).parents[1] / "fixtures" / "github_releases.atom").read_text()


def _gh_adapter() -> GitHubReleasesAdapter:
    return GitHubReleasesAdapter("anthropics/anthropic-sdk-python", fetcher=lambda _url: _ATOM)


class _RaisingAdapter:
    def source_name(self) -> str:
        return "github_releases:anthropics/anthropic-sdk-python"

    def fetch_new(self, since):
        raise RuntimeError("upstream 500")


class _StaticAdapter:
    def __init__(self, items: list[RawItem]) -> None:
        self._items = items

    def source_name(self) -> str:
        return "static"

    def fetch_new(self, since):
        return list(self._items)


class _StatefulAdapter(_StaticAdapter):
    def __init__(self, items: list[RawItem], state: dict, diagnostics: dict) -> None:
        super().__init__(items)
        self.state = state
        self.diagnostics = diagnostics
        self.received_state: dict | None = None

    def fetch_new_with_state(self, since, state):
        self.received_state = state
        return FetchResult(list(self._items), self.state, self.diagnostics)


class _InvalidStatefulAdapter(_StaticAdapter):
    def fetch_new_with_state(self, since, state):
        return {"items": []}


class _FailingStatefulAdapter(_StaticAdapter):
    def fetch_new_with_state(self, since, state):
        raise StatefulFetchError("safe transport failure", {"reason": "transport_failure"})


def _atom_one(tag: str, updated: str) -> str:
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        f"<title>{tag}</title><updated>{updated}</updated>"
        f'<link href="https://x/{tag}"/><content>notes</content>'
        "</entry></feed>"
    )


def test_same_type_different_repos_have_independent_cursors_and_dedup(session) -> None:
    # Two GitHub-releases sources must NOT share a high-water-mark cursor or a dedup
    # namespace: each tracked repo has its own release timeline, and identical tags across
    # repos (e.g. both ship "v1.0.0") are distinct items. Regression for the source-identity
    # collision (source_name was the type, not the repo) that silently dropped later repos'
    # releases and mis-reported ingest counts.
    repo_a = GitHubReleasesAdapter(
        "owner/a", fetcher=lambda _u: _atom_one("v1.0.0", "2026-07-10T00:00:00+00:00")
    )
    # repo_b ships the SAME tag, OLDER than repo_a's release (so a shared cursor would filter
    # it out, and a shared dedup namespace would treat it as a duplicate).
    repo_b = GitHubReleasesAdapter(
        "owner/b", fetcher=lambda _u: _atom_one("v1.0.0", "2026-07-01T00:00:00+00:00")
    )
    run_ingestion(session, repo_a)
    run_ingestion(session, repo_b)

    total = session.scalar(select(func.count()).select_from(FeedItem))
    assert total == 2, "repo_b's same-tag/older release was dropped or missed"
    assert repo_a.source_name() != repo_b.source_name()
    hwm_a = last_high_water_mark(session, repo_a.source_name())
    hwm_b = last_high_water_mark(session, repo_b.source_name())
    assert hwm_a is not None and hwm_b is not None and hwm_a != hwm_b


def test_run_ingestion_writes_completed_run_with_max_high_water_mark(session) -> None:
    run = run_ingestion(session, _gh_adapter())
    assert run.status == "completed"
    assert run.items_fetched == 2
    assert run.items_passed_filter == 2
    # high-water mark = max published_at consumed (the v0.30.0 timestamp).
    assert datetime.fromisoformat(run.high_water_mark) == datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    assert session.query(FeedItem).count() == 2


def test_rerun_with_stored_high_water_mark_is_idempotent(session) -> None:
    run_ingestion(session, _gh_adapter())
    second = run_ingestion(session, _gh_adapter())
    assert second.status == "completed"
    assert second.items_fetched == 0  # adapter filtered on the stored cursor
    assert second.items_passed_filter == 0
    assert session.query(FeedItem).count() == 2  # no duplicates


def test_duplicate_source_id_is_not_reinserted(session) -> None:
    # A static adapter that always returns the same item, run twice, inserts once.
    item = RawItem(
        source="static",
        source_id="x1",
        url="https://example.com/x1",
        title="X",
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    run_ingestion(session, _StaticAdapter([item]), since=None)
    run_ingestion(session, _StaticAdapter([item]), since=None)
    assert session.query(FeedItem).filter_by(source="static", source_id="x1").count() == 1


def test_stateful_adapter_when_completed_then_persists_state_diagnostics_and_replays(
    session,
) -> None:
    old = RawItem(
        source="static",
        source_id="old",
        url="https://example.com/old",
        title="Old",
        published_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    first = _StatefulAdapter([old], {"version": 1, "etag": "one"}, {"skipped": {}})
    run_ingestion(session, first)
    prior_hwm = last_high_water_mark(session, "static")

    late = RawItem(
        source="static",
        source_id="late",
        url="https://example.com/late",
        title="Late",
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    second = _StatefulAdapter([old, late], {"version": 1, "etag": "two"}, {"skipped": {}})
    run = run_ingestion(session, second)

    assert run.status == "completed"
    assert run.source_state == {"version": 1, "etag": "two"}
    assert run.diagnostics == {"skipped": {}}
    assert second.received_state == {"version": 1, "etag": "one"}
    assert session.query(FeedItem).filter_by(source="static").count() == 2
    assert last_high_water_mark(session, "static") == prior_hwm
    assert last_completed_context(session, "static") == (prior_hwm, {"version": 1, "etag": "two"})


def test_stateful_adapter_when_invalid_or_failed_then_does_not_finalize_state(session) -> None:
    item = RawItem(
        source="static",
        source_id="one",
        url="https://example.com/one",
        title="One",
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    run_ingestion(session, _StatefulAdapter([item], {"version": 1}, {"ok": True}))
    invalid = run_ingestion(session, _InvalidStatefulAdapter([]))
    failed = run_ingestion(session, _FailingStatefulAdapter([]))

    assert invalid.status == "failed"
    assert invalid.source_state is None
    assert failed.status == "failed"
    assert failed.diagnostics == {"reason": "transport_failure"}
    assert last_completed_context(session, "static") == (
        datetime(2026, 7, 1, tzinfo=UTC),
        {"version": 1},
    )


def test_failure_path_records_failed_run_and_does_not_advance_high_water_mark(session) -> None:
    # Seed one completed run so we can prove the cursor is not advanced by a failure.
    first = run_ingestion(session, _gh_adapter())
    good_hwm = first.high_water_mark

    failed = run_ingestion(session, _RaisingAdapter())
    assert failed.status == "failed"
    assert "upstream 500" in failed.error_message
    assert failed.high_water_mark is None

    # The next since resolves from the last COMPLETED run, ignoring the failure.
    assert last_high_water_mark(
        session, "github_releases:anthropics/anthropic-sdk-python"
    ) == datetime.fromisoformat(good_hwm)


def test_failure_does_not_raise_so_other_sources_continue(session) -> None:
    # run_ingestion captures the adapter error rather than propagating it.
    run = run_ingestion(session, _RaisingAdapter())
    assert isinstance(run, FeedRun)
    assert run.status == "failed"


def _github_star_fetcher(request) -> str:
    page = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["page"][0]
    if page != "1":
        return "[]"
    return json.dumps(
        [
            {
                "starred_at": "2026-07-12T00:00:00Z",
                "repo": {
                    "id": 1,
                    "name": "starred",
                    "full_name": "owner/starred",
                    "html_url": "https://github.com/owner/starred",
                    "description": "Starred repository.",
                    "owner": {"login": "owner"},
                    "topics": [],
                    "language": "Python",
                    "stargazers_count": 1,
                },
            }
        ]
    )


def _repository_fetcher(_request) -> str:
    return json.dumps(
        {
            "id": 2,
            "name": "tool",
            "full_name": "owner/tool",
            "html_url": "https://github.com/owner/tool",
            "description": "Tracked tool.",
            "owner": {"login": "owner"},
            "topics": ["ai"],
            "language": "Python",
            "stargazers_count": 2,
            "pushed_at": "2026-07-13T00:00:00Z",
        }
    )


def _chrome_adapter(tmp_path: Path) -> ChromeBookmarksAdapter:
    bookmarks_path = tmp_path / "Bookmarks"
    bookmarks_path.write_text(
        json.dumps(
            {
                "roots": {
                    "other": {
                        "children": [
                            {
                                "type": "url",
                                "guid": "11111111-1111-4111-8111-111111111111",
                                "name": "Chrome item",
                                "url": "https://example.com/chrome",
                                "date_added": "13400064000000000",
                            }
                        ]
                    }
                }
            }
        )
    )
    return ChromeBookmarksAdapter(str(bookmarks_path), "Default")


def test_new_connectors_when_rerun_with_stored_cursors_then_preserve_items_without_duplicates(
    session, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "test-token")
    adapters = [
        GitHubStarredAdapter("TEST_GITHUB_TOKEN", fetcher=_github_star_fetcher),
        _chrome_adapter(tmp_path),
        GitHubRepositoryAdapter("owner/tool", "TEST_GITHUB_TOKEN", fetcher=_repository_fetcher),
    ]

    first = [run_ingestion(session, adapter) for adapter in adapters]
    second = [run_ingestion(session, adapter) for adapter in adapters]

    assert [run.status for run in first] == ["completed", "completed", "completed"]
    assert [run.items_fetched for run in second] == [0, 0, 0]
    assert session.query(FeedItem).count() == 3


def test_new_connectors_when_one_fails_then_other_sources_continue_and_cursor_is_preserved(
    session, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "test-token")
    starred = GitHubStarredAdapter("TEST_GITHUB_TOKEN", fetcher=_github_star_fetcher)
    completed = run_ingestion(session, starred)
    failed = run_ingestion(
        session,
        GitHubStarredAdapter(
            "TEST_GITHUB_TOKEN", fetcher=lambda _request: (_ for _ in ()).throw(OSError("down"))
        ),
    )
    chrome = run_ingestion(session, _chrome_adapter(tmp_path))
    repository = run_ingestion(
        session,
        GitHubRepositoryAdapter("owner/tool", "TEST_GITHUB_TOKEN", fetcher=_repository_fetcher),
    )

    assert failed.status == "failed"
    assert last_high_water_mark(session, starred.source_name()) == datetime.fromisoformat(
        completed.high_water_mark
    )
    assert chrome.status == repository.status == "completed"
