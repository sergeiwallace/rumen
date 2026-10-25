"""Configurable views + presentation-layer filtering (design Stage 7 / T-09).

Filtering is **presentation-layer, not ingest-time** (D-6): the full corpus stays queryable;
a view is a saved filter+sort preset over ``feed_items``. The ``All`` view (or no filter)
returns everything — nothing is hardcoded out.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Text, exists, func, select
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.orm import Session

from rumen.models import FeedItem, FeedView, ItemConcept, SkosConcept, SkosRelation


class UnknownViewError(ValueError):
    """Raised when a named view does not exist."""


class UnknownSortError(ValueError):
    """Raised when a saved view asks for a non-sortable field."""


def _resolved_concept_ids(session: Session, concept_id: int) -> set[int]:
    """Return a referenced concept and its active ``useFor`` destination, if any.

    Consolidation retains the deprecated source assignment, so both ends of a valid
    alias chain remain accepted by a saved filter.  A missing concept, a deprecated
    concept without an active destination, or an alias cycle is deliberately empty.
    """

    def resolve(current_id: int, visited: set[int]) -> set[int]:
        if current_id in visited:
            return set()
        concept = session.get(SkosConcept, current_id)
        if concept is None:
            return set()
        if concept.status == "active":
            return {current_id}
        target_ids = session.scalars(
            select(SkosRelation.target_concept_id)
            .where(
                SkosRelation.source_concept_id == current_id,
                SkosRelation.relation_type == "useFor",
            )
            .order_by(SkosRelation.target_concept_id)
        ).all()
        for target_id in target_ids:
            resolved_ids = resolve(target_id, visited | {current_id})
            if resolved_ids:
                return {current_id, *resolved_ids}
        return set()

    return resolve(concept_id, set())


def stale_facet_concept_ids(session: Session, filter_json: dict | None) -> set[int]:
    """Return facet ids that cannot resolve to an active concept through ``useFor``."""
    facets = (filter_json or {}).get("facets", {})
    if not isinstance(facets, dict):
        return set()
    return {
        concept_id
        for concept_ids in facets.values()
        if isinstance(concept_ids, list)
        for concept_id in concept_ids
        if isinstance(concept_id, int) and not _resolved_concept_ids(session, concept_id)
    }


def _tag_overlap_clause(session: Session, tags: list[str]):
    """A portable JSON-array overlap predicate for SQLite and PostgreSQL."""
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        return FeedItem.tags.op("?|")(array(tags, type_=Text))
    tag_values = func.json_each(FeedItem.tags).table_valued("value").alias("tag_values")
    return exists(select(1).select_from(tag_values).where(tag_values.c.value.in_(tags)))


def get_view(session: Session, name: str) -> FeedView:
    view = session.scalars(select(FeedView).where(FeedView.name == name)).first()
    if view is None:
        raise UnknownViewError(f"no such view: {name!r}")
    return view


def filter_items(
    session: Session,
    *,
    view_name: str | None = None,
    category: str | None = None,
    source: str | None = None,
    since: datetime | None = None,
    limit: int | None = None,
) -> list[FeedItem]:
    """Return items matching a named view or ad-hoc filters (presentation-layer).

    A view's ``filter_json`` (``{category:[...], source:[...], since:..., facets:{...},
    tags:[...]}``) is applied when ``view_name`` is given; otherwise the ad-hoc
    ``category``/``source``/``since`` apply. An empty filter (the ``All`` view) returns the
    full corpus.
    """
    categories: list[str] | None = None
    sources: list[str] | None = None
    facets: dict[str, list[int]] = {}
    tags: list[str] | None = None

    sort_json: dict | None = None
    if view_name is not None:
        view = get_view(session, view_name)
        fj = view.filter_json or {}
        sort_json = view.sort_json
        categories = fj.get("category")
        sources = fj.get("source")
        facets = fj.get("facets", {}) if isinstance(fj.get("facets", {}), dict) else {}
        tags = fj.get("tags") if isinstance(fj.get("tags"), list) else None
        if fj.get("since"):
            since = since or datetime.fromisoformat(fj["since"])
    if category is not None:
        categories = [category]
    if source is not None:
        sources = [source]

    stmt = select(FeedItem)
    if categories:
        stmt = stmt.where(FeedItem.category.in_(categories))
    if sources:
        stmt = stmt.where(FeedItem.source.in_(sources))
    if since is not None:
        stmt = stmt.where(FeedItem.published_at >= since)
    for facet, concept_ids in facets.items():
        accepted_ids: set[int] = set()
        if isinstance(concept_ids, list):
            for concept_id in concept_ids:
                if isinstance(concept_id, int):
                    accepted_ids.update(_resolved_concept_ids(session, concept_id))
        stmt = stmt.where(
            exists(
                select(1).where(
                    ItemConcept.item_table == "feed_items",
                    ItemConcept.item_id == FeedItem.id,
                    ItemConcept.facet == facet,
                    ItemConcept.concept_id.in_(accepted_ids),
                )
            )
        )
    if tags is not None:
        stmt = stmt.where(
            _tag_overlap_clause(session, [tag for tag in tags if isinstance(tag, str)])
        )
    sortable = {
        "published_at": FeedItem.published_at,
        "category": FeedItem.category,
        "classification_confidence": FeedItem.classification_confidence,
        "source": FeedItem.source,
    }
    if sort_json:
        by = sort_json.get("by")
        column = sortable.get(by)
        if column is None:
            raise UnknownSortError(f"unknown view sort field: {by!r}")
        stmt = stmt.order_by(
            column.asc() if sort_json.get("dir", "desc") == "asc" else column.desc()
        )
    else:
        stmt = stmt.order_by(FeedItem.published_at.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))
