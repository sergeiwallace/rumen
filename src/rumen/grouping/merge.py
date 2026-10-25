"""Entity attach / create / survivorship / merge_log / unmerge (design Stage 4 / D-8)."""

from __future__ import annotations

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rumen.embeddings import decode_embedding, encode_embedding
from rumen.models import FeedItem, ItemConcept, MergeCandidate, MergeLog, TopicEntity

# Survivorship authority: primary source > crawled article > social pointer.
_SOURCE_AUTHORITY = {
    "github_releases": 3,
    "npm_registry": 3,
    "hf_daily": 3,
    "article": 2,
    "bookmarks": 1,
    "reddit": 1,
    "hn": 1,
    "chrome": 1,
    "github_stars": 1,
}


def _authority(source: str) -> int:
    return _SOURCE_AUTHORITY.get(source.split(":", 1)[0], 1)


def _entity_items(session: Session, entity_id: int) -> list[FeedItem]:
    return list(session.scalars(select(FeedItem).where(FeedItem.topic_entity_id == entity_id)))


def recompute_entity(
    session: Session,
    entity: TopicEntity,
    *,
    items: list[FeedItem] | None = None,
    project_facets: bool = True,
    human_topic_authoritative: bool | None = None,
) -> None:
    """Recompute survivorship, centroid, and source_count from constituent pointers.

    ``items`` lets a caller that already holds the entity's constituents (e.g. the batched
    identity pass) skip the per-entity re-query — a hot path when grouping thousands of
    pointers against a remote Postgres. When ``None`` the constituents are loaded from the DB.

    ``project_facets=False`` skips the ``item_concepts`` facet-survivorship write (still
    updates every other field). A caller that may invoke this more than once for the same
    entity within one uncommitted transaction (the batched identity pass, when 2+ items merge
    into the same entity in one commit_batch_size window) must pass ``False`` and project
    facets itself exactly once, after its flush — this function's DELETE-then-INSERT pattern
    only sees already-*flushed* rows (grouping runs with autoflush off, D-14), so calling it
    twice unflushed for the same entity/facet double-inserts and violates the unique
    constraint; calling it before the entity itself is flushed fails item_concepts' NOT NULL
    item_id the same way.
    """
    items = _entity_items(session, entity.id) if items is None else items
    if not items:
        return
    # Survivorship: highest authority, ties → earliest published.
    survivor = min(items, key=lambda i: (-_authority(i.source), i.published_at))
    entity.canonical_title = survivor.title
    entity.canonical_url = survivor.canonical_url or survivor.url

    # Resolved identity = any constituent that carries a stable id.
    with_id = next((i for i in items if i.stable_id_value), None)
    if with_id is not None:
        entity.stable_id_kind = with_id.stable_id_kind
        entity.stable_id_value = with_id.stable_id_value

    if human_topic_authoritative is None:
        human_topic_authoritative = _has_human_topic_scalar(session, entity)
    if not human_topic_authoritative:
        entity.category = _survivor_category(items)
    if project_facets:
        _project_facet_survivorship(session, entity, items)
    entity.earliest_seen = min(i.published_at for i in items)
    entity.source_count = len(items)

    vectors = [decode_embedding(i.embedding) for i in items if i.embedding is not None]
    if vectors:
        entity.embedding = encode_embedding(np.mean(np.vstack(vectors), axis=0))
    else:
        entity.embedding = None


def _project_facet_survivorship(
    session: Session, entity: TopicEntity, items: list[FeedItem]
) -> None:
    """Project each constituent's multi-label facets by union, unless humans own the facet."""
    # The live classifier only creates a topic projection together with a derived scalar
    # category. Avoid an extra item_concepts round-trip for raw, still-unclassified pointers
    # in the grouping hot path.
    if not any(item.category and item.category != "unclassified" for item in items):
        return
    labels = session.scalars(
        select(ItemConcept).where(
            ItemConcept.item_table == "feed_items",
            ItemConcept.item_id.in_([item.id for item in items]),
        )
    ).all()
    by_facet: dict[str, dict[int, float]] = {}
    for label in labels:
        if label.method.startswith("human:") or label.method in {"rule", "hmc", "llm"}:
            by_facet.setdefault(label.facet, {})[label.concept_id] = max(
                label.confidence, by_facet.get(label.facet, {}).get(label.concept_id, 0.0)
            )
    existing = session.scalars(
        select(ItemConcept).where(
            ItemConcept.item_table == "topic_entities", ItemConcept.item_id == entity.id
        )
    ).all()
    human_facets = {label.facet for label in existing if label.method.startswith("human:")}
    for facet in set(by_facet) | {label.facet for label in existing}:
        if facet in human_facets:
            continue
        session.execute(
            delete(ItemConcept).where(
                ItemConcept.item_table == "topic_entities",
                ItemConcept.item_id == entity.id,
                ItemConcept.facet == facet,
                ~ItemConcept.method.startswith("human:"),
            )
        )
        session.add_all(
            ItemConcept(
                item_table="topic_entities",
                item_id=entity.id,
                concept_id=concept_id,
                facet=facet,
                confidence=confidence,
                method="projection:survivorship",
            )
            for concept_id, confidence in by_facet.get(facet, {}).items()
        )


def _has_human_topic_scalar(session: Session, entity: TopicEntity) -> bool:
    """Whether an accepted human topic projection owns the entity scalar."""
    return (
        session.scalar(
            select(ItemConcept.id)
            .where(
                ItemConcept.item_table == "topic_entities",
                ItemConcept.item_id == entity.id,
                ItemConcept.facet == "topic",
                ItemConcept.method.startswith("human:"),
            )
            .limit(1)
        )
        is not None
    )


def _survivor_category(items: list[FeedItem]) -> str | None:
    """Highest confidence wins; tied cohort uses category majority, then earliest."""
    classified = [i for i in items if i.category]
    if not classified:
        return None
    confidence = max(i.classification_confidence or 0.0 for i in classified)
    tied = [item for item in classified if (item.classification_confidence or 0.0) == confidence]
    votes = {
        category: sum(item.category == category for item in tied)
        for category in {item.category for item in tied}
    }
    winners = {category for category, count in votes.items() if count == max(votes.values())}
    best = min(
        (item for item in tied if item.category in winners), key=lambda item: item.published_at
    )
    return best.category


def attach_item(
    session: Session,
    entity: TopicEntity,
    item: FeedItem,
    *,
    method: str,
    confidence: float | None,
    actor: str = "auto",
    evidence: dict | None = None,
    items: list[FeedItem] | None = None,
) -> None:
    """Attach a pointer to an entity: set match fields, write merge_log, recompute.

    ``items`` (the entity's full constituent list *including* ``item``) is forwarded to
    ``recompute_entity`` so the batched identity pass avoids a per-attach re-query.
    """
    item.topic_entity_id = entity.id
    item.match_method = method
    item.match_confidence = confidence
    session.add(
        MergeLog(
            topic_entity_id=entity.id,
            feed_item_id=item.id,
            match_method=method,
            match_confidence=confidence,
            actor=actor,
            action="merge",
            evidence=evidence,
        )
    )
    session.flush()
    recompute_entity(session, entity, items=items)


def create_entity(
    session: Session, item: FeedItem, *, method: str, confidence: float | None
) -> TopicEntity:
    """Seed a new entity from a pointer (its first constituent)."""
    entity = TopicEntity(
        canonical_title=item.title,
        canonical_url=item.canonical_url or item.url,
        earliest_seen=item.published_at,
        source_count=0,
    )
    session.add(entity)
    session.flush()
    attach_item(session, entity, item, method=method, confidence=confidence, items=[item])
    return entity


def record_non_duplicate(
    session: Session, entity_a: TopicEntity, entity_b: TopicEntity, *, resolved_by: str
) -> None:
    """Record a pair as ``non_duplicate`` (rejected) so it never re-merges automatically."""
    lo, hi = sorted((entity_a.id, entity_b.id))
    existing = session.scalars(
        select(MergeCandidate).where(
            MergeCandidate.entity_a_id == lo, MergeCandidate.entity_b_id == hi
        )
    ).first()
    if existing is None:
        session.add(
            MergeCandidate(
                entity_a_id=lo,
                entity_b_id=hi,
                reason="user_flagged",
                status="rejected",
                resolved_by=resolved_by,
            )
        )
    else:
        existing.status = "rejected"
        existing.resolved_by = resolved_by


def unmerge_item(session: Session, item: FeedItem) -> TopicEntity:
    """Split ``item`` out of its entity into a fresh one (design D-8 unmerge semantics).

    Writes ``merge_log.action='unmerge'``, recomputes both entities' survivorship /
    centroid / source_count, leaves ``feed_feedback`` on the surviving entity, and records
    the pair ``non_duplicate`` (no auto re-merge). ``feed_feedback`` stays with the survivor.
    """
    original = session.get(TopicEntity, item.topic_entity_id)
    session.add(
        MergeLog(
            topic_entity_id=original.id,
            feed_item_id=item.id,
            match_method="manual",
            actor="human",
            action="unmerge",
        )
    )
    fresh = TopicEntity(
        canonical_title=item.title,
        canonical_url=item.canonical_url or item.url,
        earliest_seen=item.published_at,
        source_count=0,
    )
    session.add(fresh)
    session.flush()
    item.topic_entity_id = fresh.id
    item.match_method = "manual"
    session.flush()
    recompute_entity(session, original)
    recompute_entity(session, fresh)
    record_non_duplicate(session, original, fresh, resolved_by="human")
    return fresh


def requeue_source_refresh(session: Session, item: FeedItem) -> None:
    """Clear only machine-owned downstream state after a source metadata change.

    Accepted human classification and manual grouping are deliberately retained.  The
    regular classifier/grouping pipeline will rebuild the cleared machine state.
    """
    has_human_correction = (
        session.scalar(
            select(ItemConcept.id)
            .where(
                ItemConcept.item_table == "feed_items",
                ItemConcept.item_id == item.id,
                ItemConcept.method.startswith("human:"),
            )
            .limit(1)
        )
        is not None
    )
    if not has_human_correction:
        item.category = None
        item.categories = None
        item.domains = None
        item.tags = None
        item.classification_confidence = None
        item.classification_method = None
        item.facets_classified_at = None
    item.matched_keywords = None
    item.embedding = None
    if item.topic_entity_id is not None and item.match_method != "manual":
        session.add(
            MergeLog(
                topic_entity_id=item.topic_entity_id,
                feed_item_id=item.id,
                match_method="source_refresh",
                match_confidence=None,
                actor="system:youtube",
                action="source_refresh",
                evidence={"reason": "provider metadata changed"},
            )
        )
        item.topic_entity_id = None
        item.match_method = None
        item.match_confidence = None
