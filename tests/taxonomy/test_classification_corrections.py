"""T-24: human facet-review history and its gold-label projection."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from alembic import command
from sqlalchemy import event, select, text

from rumen.cli import run
from rumen.db import migrate
from rumen.models import (
    ClassificationCorrection,
    FeedItem,
    ItemConcept,
    SkosConcept,
    SkosRelation,
    TopicEntity,
)
from rumen.taxonomy.corrections import record_classification_review
from rumen.taxonomy.hmc.gold_eval import load_human_gold_examples

_NOW = datetime(2026, 7, 19, tzinfo=UTC)


def _item(session, source_id: str = "review-item") -> FeedItem:
    item = FeedItem(
        source="review",
        source_id=source_id,
        url=f"https://example.test/{source_id}",
        title="Reviewed item",
        content="Classification review fixture",
        published_at=_NOW,
    )
    session.add(item)
    session.flush()
    return item


def _concept(session, facet: str, label: str) -> SkosConcept:
    concept = SkosConcept(facet=facet, pref_label=label, scheme="test")
    session.add(concept)
    session.flush()
    return concept


def _assignment(
    session, item: FeedItem, concept: SkosConcept, *, method: str = "hmc"
) -> ItemConcept:
    row = ItemConcept(
        item_table="feed_items",
        item_id=item.id,
        concept_id=concept.id,
        facet=concept.facet,
        confidence=0.62,
        method=method,
    )
    session.add(row)
    session.flush()
    return row


def _review(session, item: FeedItem, facet: str, concepts: list[SkosConcept]):
    return record_classification_review(
        session,
        item_table="feed_items",
        item_id=item.id,
        facet=facet,
        corrected_concept_ids=[concept.id for concept in concepts],
        actor="human:reviewer",
        rationale="Reviewed against the complete facet.",
        source="test",
    )


def test_review_when_valid_input_then_creates_accepted_event_with_complete_prior_and_context(
    session,
) -> None:
    item = _item(session)
    prior = _concept(session, "topic", "Prior")
    corrected = _concept(session, "topic", "Corrected")
    _assignment(session, item, prior, method="bootstrap:llm-teacher")

    correction = record_classification_review(
        session,
        item_table="feed_items",
        item_id=item.id,
        facet="topic",
        corrected_concept_ids=[corrected.id],
        actor="human:reviewer",
        rationale="The item is about the corrected concept.",
        source="test",
        classifier_context={"artifact": "hmc-topic-v2", "tau": 0.9},
    )
    session.commit()

    stored = session.get(ClassificationCorrection, correction.id)
    assert stored is not None
    assert stored.prior_concept_ids == [prior.id]
    assert stored.corrected_concept_ids == [corrected.id]
    assert stored.action == "correction"
    assert stored.status == "accepted"
    assert stored.source == "test"
    assert stored.actor == "human:reviewer"
    assert stored.classifier_context_json == {
        "artifact": "hmc-topic-v2",
        "tau": 0.9,
        "topic_closure": {
            "selected_concept_ids": [corrected.id],
            "inferred_concept_ids": [],
        },
    }
    assert stored.submitted_at is not None and stored.decided_at is not None


def test_review_when_multilabel_correction_then_replaces_only_target_facet(session) -> None:
    item = _item(session)
    other_item = _item(session, "other-item")
    prior = _concept(session, "topic", "Prior")
    first = _concept(session, "topic", "First")
    second = _concept(session, "topic", "Second")
    other_facet = _concept(session, "content-type", "Paper")
    _assignment(session, item, prior)
    _assignment(session, item, other_facet)
    _assignment(session, other_item, prior)

    _review(session, item, "topic", [first, second])
    session.commit()

    rows = session.scalars(
        select(ItemConcept).order_by(ItemConcept.item_id, ItemConcept.facet)
    ).all()
    target = [row for row in rows if row.item_id == item.id and row.facet == "topic"]
    assert {(row.concept_id, row.confidence, row.method) for row in target} == {
        (first.id, 1.0, "human:test"),
        (second.id, 1.0, "human:test"),
    }
    assert {(row.item_id, row.concept_id, row.facet) for row in rows if row not in target} == {
        (item.id, other_facet.id, "content-type"),
        (other_item.id, prior.id, "topic"),
    }


def test_review_when_unchanged_then_records_approval_and_projects_human_gold(session) -> None:
    item = _item(session)
    concept = _concept(session, "topic", "Approved")
    _assignment(session, item, concept)

    correction = _review(session, item, "topic", [concept])
    session.commit()

    assert correction.action == "approval"
    stored = session.scalar(select(ItemConcept).where(ItemConcept.item_id == item.id))
    assert stored is not None and stored.method == "human:test"
    assert load_human_gold_examples(session, facet="topic")[0].concept_ids == frozenset(
        {concept.id}
    )


def test_review_when_later_review_arrives_then_supersedes_prior_event_without_deleting_history(
    session,
) -> None:
    item = _item(session)
    first = _concept(session, "topic", "First")
    second = _concept(session, "topic", "Second")

    original = _review(session, item, "topic", [first])
    session.commit()
    replacement = _review(session, item, "topic", [second])
    session.commit()

    events = session.scalars(
        select(ClassificationCorrection).where(ClassificationCorrection.item_id == item.id)
    ).all()
    assert len(events) == 2
    prior = session.get(ClassificationCorrection, original.id)
    current = session.get(ClassificationCorrection, replacement.id)
    assert prior is not None and prior.status == "superseded"
    assert current is not None and current.status == "accepted" and current.supersedes == prior.id
    assert current.corrected_concept_ids == [second.id]


def test_migration_when_existing_human_rows_then_backfills_events_without_changing_projection(
    migrated_engine,
) -> None:
    # Downgrade only T-24, seed a real legacy human projection, then upgrade through its
    # data migration.  ``migrated_engine`` exercises both SQLite and real Postgres.
    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0008")
        connection.execute(
            text(
                "INSERT INTO feed_items (source, source_id, url, title, published_at, ingested_at) "
                "VALUES ('legacy-human', 'seed', 'https://example.test/seed', 'seed', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        item_id = connection.execute(
            text("SELECT id FROM feed_items WHERE source = 'legacy-human'")
        ).scalar_one()
        concept_id = connection.execute(
            text("SELECT id FROM skos_concepts WHERE facet = 'topic' ORDER BY id LIMIT 1")
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO item_concepts (item_table, item_id, concept_id, facet, confidence, method) "
                "VALUES ('feed_items', :item_id, :concept_id, 'topic', 1.0, 'human:legacy')"
            ),
            {"item_id": item_id, "concept_id": concept_id},
        )
    with migrated_engine.connect() as connection:
        command.upgrade(migrate._config(connection), "head")
        projection = connection.execute(
            text(
                "SELECT concept_id, confidence, method FROM item_concepts "
                "WHERE item_table = 'feed_items' AND item_id = :item_id"
            ),
            {"item_id": item_id},
        ).one()
        correction = connection.execute(
            text(
                "SELECT prior_concept_ids, corrected_concept_ids, actor, source, rationale, status "
                "FROM classification_corrections WHERE item_table = 'feed_items' AND item_id = :item_id"
            ),
            {"item_id": item_id},
        ).one()
    assert projection == (concept_id, 1.0, "human:legacy")
    assert correction.prior_concept_ids == correction.corrected_concept_ids
    assert correction.actor == "system:legacy-human-import"
    assert correction.source == "migration"
    assert "provenance is unavailable" in correction.rationale.lower()
    assert correction.status == "accepted"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"item_id": 99999}, "no such feed_items target"),
        ({"corrected_concept_ids": []}, "must not be empty"),
        ({"corrected_concept_ids": [1, 1]}, "must not contain duplicates"),
        ({"corrected_concept_ids": [-99999]}, "does not exist"),
        ({"corrected_concept_ids": [99999]}, "does not belong"),
        ({"actor": " "}, "actor must not be blank"),
        ({"rationale": " "}, "rationale must not be blank"),
    ],
)
def test_review_when_invalid_input_then_rejects_before_changing_event_or_projection(
    session, kwargs, message
) -> None:
    item = _item(session)
    topic = _concept(session, "topic", "Topic")
    other_facet = _concept(session, "source", "Source")
    _assignment(session, item, topic)
    session.commit()
    call = {
        "item_table": "feed_items",
        "item_id": item.id,
        "facet": "topic",
        "corrected_concept_ids": [topic.id],
        "actor": "human:reviewer",
        "rationale": "Valid rationale",
        "source": "test",
    }
    call.update(kwargs)
    if kwargs.get("corrected_concept_ids") == [99999]:
        call["corrected_concept_ids"] = [other_facet.id]

    with pytest.raises(ValueError, match=message):
        record_classification_review(session, **call)

    assert session.scalars(select(ClassificationCorrection)).all() == []
    remaining = session.scalars(select(ItemConcept).where(ItemConcept.item_id == item.id)).all()
    assert [(row.concept_id, row.method) for row in remaining] == [(topic.id, "hmc")]


@pytest.mark.parametrize("fail_on_flush", [1, 2])
def test_review_when_event_or_projection_write_fails_then_rolls_back_both(
    session, fail_on_flush: int
) -> None:
    item = _item(session)
    prior = _concept(session, "topic", "Prior")
    corrected = _concept(session, "topic", "Corrected")
    _assignment(session, item, prior)
    session.commit()
    flushes = 0

    def fail_write(*_args) -> None:
        nonlocal flushes
        flushes += 1
        if flushes == fail_on_flush:
            raise RuntimeError("simulated database write failure")

    event.listen(session, "before_flush", fail_write)
    try:
        with pytest.raises(RuntimeError, match="simulated database write failure"):
            _review(session, item, "topic", [corrected])
    finally:
        event.remove(session, "before_flush", fail_write)

    assert session.scalars(select(ClassificationCorrection)).all() == []
    remaining = session.scalars(select(ItemConcept).where(ItemConcept.item_id == item.id)).all()
    assert [(row.concept_id, row.method) for row in remaining] == [(prior.id, "hmc")]


def test_cli_review_when_valid_then_projects_cli_human_label(session, capsys) -> None:
    item = _item(session)
    concept = _concept(session, "topic", "CLI topic")
    session.commit()

    result = run(
        [
            "classify",
            "review",
            "feed_items",
            str(item.id),
            "topic",
            "--concept",
            str(concept.id),
            "--actor",
            "human:cli-reviewer",
            "--rationale",
            "CLI review",
        ],
        session,
    )

    assert result == 0
    assert '"action": "correction"' in capsys.readouterr().out
    row = session.scalar(select(ItemConcept).where(ItemConcept.item_id == item.id))
    assert row is not None and row.method == "human:cli"


def test_given_human_feed_topic_review_when_accepted_then_it_projects_closure_and_derives_scalar(
    session,
) -> None:
    item = _item(session)
    root = _concept(session, "topic", "Root")
    child = _concept(session, "topic", "Child")
    session.add(
        SkosRelation(source_concept_id=child.id, target_concept_id=root.id, relation_type="broader")
    )

    correction = _review(session, item, "topic", [child])
    session.commit()

    assert correction.corrected_concept_ids == [root.id, child.id]
    assert {
        row.concept_id
        for row in session.scalars(
            select(ItemConcept).where(ItemConcept.item_id == item.id, ItemConcept.facet == "topic")
        )
    } == {root.id, child.id}
    stored = session.get(FeedItem, item.id)
    assert (stored.category, stored.classification_confidence, stored.classification_method) == (
        "Root",
        1.0,
        "human:test",
    )


def test_given_human_entity_topic_review_when_accepted_then_it_derives_and_preserves_entity_category(
    session,
) -> None:
    item = _item(session)
    entity = TopicEntity(
        canonical_title=item.title,
        canonical_url=item.url,
        earliest_seen=item.published_at,
        source_count=1,
    )
    root = _concept(session, "topic", "Human root")
    child = _concept(session, "topic", "Human child")
    session.add_all(
        [
            entity,
            SkosRelation(
                source_concept_id=child.id, target_concept_id=root.id, relation_type="broader"
            ),
        ]
    )
    session.flush()
    item.topic_entity_id = entity.id

    record_classification_review(
        session,
        item_table="topic_entities",
        item_id=entity.id,
        facet="topic",
        corrected_concept_ids=[child.id],
        actor="human:reviewer",
        rationale="The entity is about the selected child.",
        source="test",
    )
    session.flush()
    assert session.get(TopicEntity, entity.id).category == "Human root"

    from rumen.grouping.merge import recompute_entity

    recompute_entity(session, entity)
    session.commit()

    assert session.get(TopicEntity, entity.id).category == "Human root"


def test_given_reviewer_selects_a_leaf_concept_when_accepted_then_the_full_active_parent_closure_is_projected_and_disclosed(
    session,
) -> None:
    item = _item(session)
    root = _concept(session, "topic", "Root")
    parent = _concept(session, "topic", "Parent")
    leaf = _concept(session, "topic", "Leaf")
    session.add_all(
        [
            SkosRelation(
                source_concept_id=leaf.id, target_concept_id=parent.id, relation_type="broader"
            ),
            SkosRelation(
                source_concept_id=parent.id, target_concept_id=root.id, relation_type="broader"
            ),
        ]
    )

    correction = _review(session, item, "topic", [leaf])
    session.commit()

    assert correction.corrected_concept_ids == [root.id, parent.id, leaf.id]
    assert correction.classifier_context_json["topic_closure"] == {
        "selected_concept_ids": [leaf.id],
        "inferred_concept_ids": [root.id, parent.id],
    }


def test_given_out_of_scope_derived_category_when_route_chosen_then_it_drops_with_audit(
    session,
) -> None:
    item = _item(session)
    topic = _concept(session, "topic", "Out")

    record_classification_review(
        session,
        item_table="feed_items",
        item_id=item.id,
        facet="topic",
        corrected_concept_ids=[topic.id],
        actor="human:reviewer",
        rationale="Out of scope.",
        source="test",
        out_of_scope=frozenset({"Out"}),
        route="route",
    )
    session.commit()

    from rumen.models import DroppedItem

    assert session.get(FeedItem, item.id).status == "dropped"
    dropped = session.scalar(select(DroppedItem).where(DroppedItem.source_id == item.source_id))
    assert dropped is not None and dropped.drop_reason == "classified_out_of_scope"


def test_given_out_of_scope_derived_category_when_keep_chosen_then_status_and_audit_are_unchanged(
    session,
) -> None:
    item = _item(session)
    topic = _concept(session, "topic", "Out")

    _review_kwargs = {
        "item_table": "feed_items",
        "item_id": item.id,
        "facet": "topic",
        "corrected_concept_ids": [topic.id],
        "actor": "human:reviewer",
        "rationale": "Keep this item.",
        "source": "test",
        "out_of_scope": frozenset({"Out"}),
        "route": "keep",
    }
    record_classification_review(session, **_review_kwargs)
    session.commit()

    from rumen.models import DroppedItem

    assert session.get(FeedItem, item.id).status == "active"
    assert session.scalars(select(DroppedItem)).all() == []


def test_given_out_of_scope_derived_category_when_no_choice_supplied_then_it_is_rejected(
    session,
) -> None:
    item = _item(session)
    topic = _concept(session, "topic", "Out")
    session.commit()

    with pytest.raises(ValueError, match="derived category 'Out'.*choose route or keep"):
        record_classification_review(
            session,
            item_table="feed_items",
            item_id=item.id,
            facet="topic",
            corrected_concept_ids=[topic.id],
            actor="human:reviewer",
            rationale="Out of scope.",
            source="test",
            out_of_scope=frozenset({"Out"}),
        )

    assert session.scalars(select(ClassificationCorrection)).all() == []
    assert session.get(FeedItem, item.id).category is None


def test_given_invalid_or_non_derivable_human_topic_review_when_captured_then_it_rolls_back_event_projection_and_scalars(
    session,
) -> None:
    item = _item(session)
    prior = _concept(session, "topic", "Prior")
    inactive = _concept(session, "topic", "Inactive")
    inactive.status = "inactive"
    _assignment(session, item, prior)
    item.category = "Prior"
    item.classification_confidence = 0.62
    item.classification_method = "hmc"
    session.commit()

    with pytest.raises(ValueError, match="complete active parent closure"):
        _review(session, item, "topic", [inactive])

    remaining = session.scalars(select(ItemConcept).where(ItemConcept.item_id == item.id)).all()
    stored = session.get(FeedItem, item.id)
    assert [(row.concept_id, row.method) for row in remaining] == [(prior.id, "hmc")]
    assert (stored.category, stored.classification_confidence, stored.classification_method) == (
        "Prior",
        0.62,
        "hmc",
    )
    assert session.scalars(select(ClassificationCorrection)).all() == []


def test_cli_review_when_out_of_scope_route_is_omitted_then_it_returns_a_clear_error(
    session, capsys, tmp_path
) -> None:
    item = _item(session)
    topic = _concept(session, "topic", "Out")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "sources": [],
                "out_of_scope": ["Out"],
                "classifier_escalation_alarm_threshold": 0.25,
            }
        )
    )
    session.commit()

    result = run(
        [
            "classify",
            "review",
            "feed_items",
            str(item.id),
            "topic",
            "--concept",
            str(topic.id),
            "--actor",
            "human:cli-reviewer",
            "--rationale",
            "CLI review",
            "--config",
            str(config),
        ],
        session,
    )

    assert result == 1
    assert "derived category 'Out' is out of scope; choose route or keep" in capsys.readouterr().err
