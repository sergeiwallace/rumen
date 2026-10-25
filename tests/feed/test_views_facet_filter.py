"""T-19: saved views support facet and tag clauses, including stale-filter safety."""

from __future__ import annotations

from datetime import UTC, datetime

from rumen.cli import run
from rumen.models import FeedItem, FeedView, ItemConcept, SkosConcept
from rumen.views import filter_items

_NOW = datetime(2026, 7, 19, tzinfo=UTC)


def _item(session, source_id: str, *, tags: list[str] | None = None) -> FeedItem:
    item = FeedItem(
        source="fixture",
        source_id=source_id,
        url=f"https://example.test/{source_id}",
        title=source_id,
        published_at=_NOW,
        tags=tags,
    )
    session.add(item)
    session.flush()
    return item


def _concept(session, label: str, *, status: str = "active") -> SkosConcept:
    concept = SkosConcept(facet="content-type", pref_label=label, scheme="fixture", status=status)
    session.add(concept)
    session.flush()
    return concept


def test_saved_view_facets_return_only_items_with_matching_assignment(session) -> None:
    matching = _item(session, "matching")
    other = _item(session, "other")
    wanted = _concept(session, "Wanted")
    unwanted = _concept(session, "Unwanted")
    session.add_all(
        [
            ItemConcept(
                item_table="feed_items",
                item_id=matching.id,
                concept_id=wanted.id,
                facet="content-type",
                confidence=1.0,
                method="fixture",
            ),
            ItemConcept(
                item_table="feed_items",
                item_id=other.id,
                concept_id=unwanted.id,
                facet="content-type",
                confidence=1.0,
                method="fixture",
            ),
        ]
    )
    session.add(FeedView(name="Wanted", filter_json={"facets": {"content-type": [wanted.id]}}))
    session.commit()

    assert [item.source_id for item in filter_items(session, view_name="Wanted")] == ["matching"]


def test_saved_view_tags_return_items_with_any_listed_tag(session) -> None:
    release = _item(session, "release", tags=["release", "open-source"])
    _item(session, "paper", tags=["paper"])
    _item(session, "untagged")
    session.add(FeedView(name="Tagged", filter_json={"tags": ["release", "announcement"]}))
    session.commit()

    assert [item.id for item in filter_items(session, view_name="Tagged")] == [release.id]


def test_stale_facet_filter_returns_no_matches_and_status_warns(session, capsys) -> None:
    item = _item(session, "deprecated-assignment")
    stale = _concept(session, "Pruned", status="deprecated")
    session.add_all(
        [
            ItemConcept(
                item_table="feed_items",
                item_id=item.id,
                concept_id=stale.id,
                facet="content-type",
                confidence=1.0,
                method="fixture",
            ),
            FeedView(
                name="Stale",
                filter_json={"facets": {"content-type": [stale.id]}},
            ),
        ]
    )
    session.commit()

    assert filter_items(session, view_name="Stale") == []
    assert run(["feed", "status"], session) == 0
    assert "view:Stale  [stale filter: concept ids" in capsys.readouterr().out
