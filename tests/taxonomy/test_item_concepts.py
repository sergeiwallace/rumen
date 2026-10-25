"""T-15: facet-assignment model acceptance coverage."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from rumen.models import FeedItem, ItemConcept


def _feed_item(session) -> FeedItem:
    item = FeedItem(
        source="test",
        source_id="item-concepts",
        url="https://example.test/item-concepts",
        title="Item concepts",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    session.add(item)
    session.flush()
    return item


def _concept(session, *, concept_id: int, facet: str = "content-type") -> int:
    session.execute(
        text(
            "INSERT INTO skos_concepts "
            "(id, facet, pref_label, scheme, status) "
            "VALUES (:id, :facet, :label, 'test', 'active')"
        ),
        {"id": concept_id, "facet": facet, "label": f"concept-{concept_id}"},
    )
    return concept_id


def test_item_concept_when_written_then_roundtrips_and_facet_query_needs_no_concept_join(
    session,
) -> None:
    item = _feed_item(session)
    concept_id = _concept(session, concept_id=1001)
    assignment = ItemConcept(
        item_table="feed_items",
        item_id=item.id,
        concept_id=concept_id,
        facet="content-type",
        confidence=0.87,
        method="hmc",
    )
    session.add(assignment)
    session.commit()
    session.expire_all()

    assignments = session.scalars(
        select(ItemConcept).where(
            ItemConcept.item_table == "feed_items",
            ItemConcept.item_id == item.id,
            ItemConcept.facet == "content-type",
        )
    ).all()
    assert [(row.confidence, row.method) for row in assignments] == [(0.87, "hmc")]


def test_item_concept_when_item_has_no_assignments_for_facet_then_query_is_empty(session) -> None:
    item = _feed_item(session)
    concept_id = _concept(session, concept_id=1002, facet="topic")
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=concept_id,
            facet="topic",
            confidence=1.0,
            method="rule",
        )
    )
    session.commit()

    assert (
        session.scalars(
            select(ItemConcept).where(
                ItemConcept.item_table == "feed_items",
                ItemConcept.item_id == item.id,
                ItemConcept.facet == "content-type",
            )
        ).all()
        == []
    )


def test_item_concept_when_concept_is_missing_then_foreign_key_rejects_it(session) -> None:
    item = _feed_item(session)
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=999999,
            facet="content-type",
            confidence=0.8,
            method="hmc",
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_item_concept_when_assignment_duplicates_concept_then_unique_constraint_rejects_it(
    session,
) -> None:
    item = _feed_item(session)
    concept_id = _concept(session, concept_id=1003)
    assignment = {
        "item_table": "feed_items",
        "item_id": item.id,
        "concept_id": concept_id,
        "facet": "content-type",
        "confidence": 0.8,
        "method": "hmc",
    }
    session.add_all([ItemConcept(**assignment), ItemConcept(**assignment)])

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
