"""T-24 MCP classification-review surface."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy import select

from rumen.embeddings import EMBEDDING_DIM, decode_embedding, encode_embedding
from rumen.mcp_tools import correct_item_classification, rate_feed_entity
from rumen.models import (
    ClassificationCorrection,
    DroppedItem,
    FeedFeedback,
    FeedItem,
    InterestProfileRow,
    ItemConcept,
    SkosConcept,
    TopicEntity,
)
from rumen.ranking.interest_profile import InterestProfile


def _unit(i: int) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[i] = 1.0
    return vector


def _entity(session, embedding: np.ndarray | None) -> TopicEntity:
    entity = TopicEntity(
        canonical_title="Rated entity",
        canonical_url="https://example.test/rated",
        earliest_seen=datetime(2026, 7, 20, tzinfo=UTC),
        embedding=encode_embedding(embedding) if embedding is not None else None,
    )
    session.add(entity)
    session.flush()
    return entity


def test_correct_item_classification_when_valid_then_commits_mcp_human_projection(session) -> None:
    item = FeedItem(
        source="mcp",
        source_id="review",
        url="https://example.test/review",
        title="MCP review",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    concept = SkosConcept(facet="topic", pref_label="MCP topic", scheme="test")
    session.add_all([item, concept])
    session.flush()

    result = correct_item_classification(
        session,
        item_table="feed_items",
        item_id=item.id,
        facet="topic",
        corrected_concept_ids=[concept.id],
        actor="human:mcp-reviewer",
        rationale="MCP review rationale",
    )

    assert result["status"] == "accepted"
    correction = session.get(ClassificationCorrection, result["id"])
    assert correction is not None and correction.source == "mcp"
    projection = session.scalar(select(ItemConcept).where(ItemConcept.item_id == item.id))
    assert projection is not None and projection.method == "human:mcp"


def test_correct_item_classification_when_invalid_then_rolls_back_and_raises_value_error(
    session,
) -> None:
    with pytest.raises(ValueError, match="no such feed_items target"):
        correct_item_classification(
            session,
            item_table="feed_items",
            item_id=99999,
            facet="topic",
            corrected_concept_ids=[1],
            actor="human:mcp-reviewer",
            rationale="MCP review rationale",
        )
    assert session.scalars(select(ClassificationCorrection)).all() == []


def test_correct_item_classification_when_out_of_scope_then_routes_or_rejects_missing_choice(
    session,
) -> None:
    routed = FeedItem(
        source="mcp",
        source_id="routed",
        url="https://example.test/routed",
        title="Routed MCP review",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    rejected = FeedItem(
        source="mcp",
        source_id="rejected",
        url="https://example.test/rejected",
        title="Rejected MCP review",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    out_of_scope = SkosConcept(facet="topic", pref_label="Out", scheme="test")
    session.add_all([routed, rejected, out_of_scope])
    session.flush()

    result = correct_item_classification(
        session,
        item_table="feed_items",
        item_id=routed.id,
        facet="topic",
        corrected_concept_ids=[out_of_scope.id],
        actor="human:mcp-reviewer",
        rationale="Route this item.",
        out_of_scope=frozenset({"Out"}),
        route="route",
    )

    assert result["status"] == "accepted"
    assert session.get(FeedItem, routed.id).status == "dropped"
    assert session.scalar(select(DroppedItem).where(DroppedItem.source_id == routed.source_id))

    with pytest.raises(ValueError, match="derived category 'Out'.*choose route or keep"):
        correct_item_classification(
            session,
            item_table="feed_items",
            item_id=rejected.id,
            facet="topic",
            corrected_concept_ids=[out_of_scope.id],
            actor="human:mcp-reviewer",
            rationale="This needs a routing choice.",
            out_of_scope=frozenset({"Out"}),
        )

    assert session.get(FeedItem, rejected.id).status == "active"
    assert (
        session.scalar(
            select(ClassificationCorrection).where(ClassificationCorrection.item_id == rejected.id)
        )
        is None
    )


def test_given_a_rating_when_rate_feed_entity_runs_then_it_updates_and_persists_the_profile_centroid(
    session,
) -> None:
    entity = _entity(session, _unit(1))
    session.add(InterestProfileRow(centroid=encode_embedding(_unit(0)), alpha=0.5))
    session.commit()

    result = rate_feed_entity(session, entity.id, 1)
    stored = session.scalar(select(InterestProfileRow))

    expected = InterestProfile(_unit(0), alpha=0.5)
    expected.update(_unit(1), rating=1)
    assert result == {"topic_entity_id": entity.id, "rating": 1}
    assert stored is not None
    assert np.allclose(decode_embedding(stored.centroid), expected.centroid)
    assert session.scalar(select(FeedFeedback).where(FeedFeedback.topic_entity_id == entity.id))


def test_given_an_entity_with_no_embedding_when_rated_then_the_profile_is_unchanged_but_feedback_is_recorded(
    session,
) -> None:
    entity = _entity(session, None)
    original = InterestProfile(_unit(0), alpha=0.25)
    row = InterestProfileRow(centroid=encode_embedding(original.centroid), alpha=original.alpha)
    session.add(row)
    session.commit()
    original_updated_at = row.updated_at

    rate_feed_entity(session, entity.id, -1)
    session.refresh(row)

    assert np.allclose(decode_embedding(row.centroid), original.centroid)
    assert row.alpha == original.alpha
    assert row.updated_at == original_updated_at
    assert session.scalar(select(FeedFeedback).where(FeedFeedback.topic_entity_id == entity.id))
