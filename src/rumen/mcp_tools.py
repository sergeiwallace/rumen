"""MCP tool implementations (design Stage 7 / T-09).

Pure functions the MCP *host* (outside this package) wraps as tools; kept host-agnostic so the CLI
and any surface share them. Return JSON-serializable data.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.embeddings import decode_embedding, encode_embedding
from rumen.models import (
    Digest,
    FeedFeedback,
    FeedItem,
    InterestProfileRow,
    TopicEntity,
)
from rumen.ranking.interest_profile import load_or_seed_profile
from rumen.taxonomy.corrections import record_classification_review
from rumen.views import filter_items


def _item_dict(item: FeedItem) -> dict:
    return {
        "id": item.id,
        "source": item.source,
        "title": item.title,
        "url": item.url,
        "category": item.category,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "matched_keywords": item.matched_keywords,
    }


def get_feed_digest(session: Session) -> dict | None:
    """Return the latest digest's structured content, or ``None`` if none exists."""
    digest = session.scalars(
        select(Digest).order_by(Digest.generated_at.desc(), Digest.id.desc()).limit(1)
    ).first()
    if digest is None:
        return None
    return {
        "id": digest.id,
        "period_start": digest.period_start.isoformat(),
        "period_end": digest.period_end.isoformat(),
        "content": digest.content_json,
        "markdown": digest.content_markdown,
    }


def get_feed_items(
    session: Session,
    *,
    view: str | None = None,
    category: str | None = None,
    source: str | None = None,
    since: datetime | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return items for a named view or ad-hoc filter (presentation-layer)."""
    items = filter_items(
        session, view_name=view, category=category, source=source, since=since, limit=limit
    )
    return [_item_dict(i) for i in items]


def rate_feed_entity(session: Session, entity_id: int, rating: int) -> dict:
    """Record a ``-1/0/1`` rating on a topic entity (feeds the D-2 profile update)."""
    if rating not in (-1, 0, 1):
        raise ValueError("rating must be -1, 0, or 1")
    entity = session.get(TopicEntity, entity_id)
    if entity is None:
        raise ValueError(f"no such topic entity: {entity_id}")

    profile = load_or_seed_profile(session)
    session.add(FeedFeedback(topic_entity_id=entity_id, rating=rating))
    if entity.embedding is not None:
        profile.update(decode_embedding(entity.embedding), rating)
        profile_row = session.scalars(select(InterestProfileRow)).one()
        profile_row.centroid = encode_embedding(profile.centroid)
    session.commit()
    return {"topic_entity_id": entity_id, "rating": rating}


def correct_item_classification(
    session: Session,
    *,
    item_table: str,
    item_id: int,
    facet: str,
    corrected_concept_ids: list[int],
    actor: str,
    rationale: str,
    classifier_context: dict | None = None,
    out_of_scope: frozenset[str] = frozenset(),
    route: str | None = None,
) -> dict:
    """Accept one complete human item/facet review from the MCP surface."""
    try:
        correction = record_classification_review(
            session,
            item_table=item_table,
            item_id=item_id,
            facet=facet,
            corrected_concept_ids=corrected_concept_ids,
            actor=actor,
            rationale=rationale,
            source="mcp",
            classifier_context=classifier_context,
            out_of_scope=out_of_scope,
            route=route,
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    return {
        "id": correction.id,
        "item_table": correction.item_table,
        "item_id": correction.item_id,
        "facet": correction.facet,
        "action": correction.action,
        "status": correction.status,
    }
