"""Entity absorption preserves references on both supported database dialects."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select

from rumen.embeddings import encode_embedding
from rumen.grouping import group_pending
from rumen.models import (
    BacklogCandidate,
    BacklogMappingReview,
    BacklogMiningRun,
    BacklogTaskMatch,
    Base,
    ClassificationCorrection,
    Digest,
    DigestEntity,
    FeedFeedback,
    FeedItem,
    ItemConcept,
    MergeCandidate,
    MergeLog,
    SkosConcept,
    TopicEntity,
)

_NOW = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.fixture
def absorption(session):
    entities = [
        TopicEntity(
            canonical_title=name,
            canonical_url=f"https://example.com/{name}",
            earliest_seen=_NOW,
            embedding=encode_embedding(np.ones(384)) if name != "unrelated" else None,
        )
        for name in ("keep", "drop", "unrelated")
    ]
    session.add_all(entities)
    session.flush()
    keep, drop, unrelated = entities
    item = FeedItem(
        source="article",
        source_id="absorbed",
        url=drop.canonical_url,
        title="absorbed",
        published_at=_NOW,
        topic_entity_id=drop.id,
        embedding=drop.embedding,
    )
    # A pending pointer exercises the public grouping entry point, including its commits.
    pending = FeedItem(
        source="article",
        source_id="pending",
        url="https://example.com/pending",
        title="",
        published_at=_NOW,
    )
    session.add_all([item, pending])
    session.flush()
    return keep, drop, unrelated, item


@pytest.mark.parametrize(
    "reference",
    [
        "feed_items",
        "merge_log",
        "feed_feedback",
        "digest_entities",
        "backlog_mapping_reviews",
        "backlog_candidates",
        "merge_candidates_a",
        "merge_candidates_b",
        "item_concepts",
        "classification_corrections",
    ],
)
def test_given_absorbed_entity_references_when_group_pending_runs_then_rows_follow_survivor(
    session, absorption, reference
):
    keep, drop, unrelated, item = absorption
    if reference == "feed_items":
        row, column = item, "topic_entity_id"
    elif reference == "merge_log":
        row = MergeLog(
            topic_entity_id=drop.id, feed_item_id=item.id, match_method="manual", actor="human"
        )
        column = "topic_entity_id"
    elif reference == "feed_feedback":
        row = FeedFeedback(topic_entity_id=drop.id, rating=1)
        column = "topic_entity_id"
    elif reference == "digest_entities":
        digest = Digest(period_start=_NOW, period_end=_NOW, content_json={}, content_markdown="")
        session.add(digest)
        session.flush()
        row = DigestEntity(digest_id=digest.id, topic_entity_id=drop.id, rank=3)
        column = "topic_entity_id"
    elif reference == "backlog_mapping_reviews":
        run = BacklogMiningRun(status="completed")
        session.add(run)
        session.flush()
        row = BacklogMappingReview(
            mining_run_id=run.id,
            topic_entity_id=drop.id,
            status="reviewed",
            confidence=0.8,
            threshold=0.7,
            decision_inputs={"review": "preserve this decision"},
        )
        column = "topic_entity_id"
    elif reference == "backlog_candidates":
        row = BacklogCandidate(
            topic_entity_id=drop.id,
            fingerprint="distinct-candidate",
            source_snapshot={},
            content_snapshot={},
            assessment={},
            configuration_snapshot={},
            state="accepted",
        )
        column = "topic_entity_id"
    elif reference.startswith("merge_candidates"):
        a, b = (drop.id, unrelated.id) if reference.endswith("a") else (unrelated.id, drop.id)
        row = MergeCandidate(
            entity_a_id=a, entity_b_id=b, status="rejected", resolved_by="human:reviewer"
        )
        column = "entity_a_id" if reference.endswith("a") else "entity_b_id"
    elif reference == "item_concepts":
        concept = SkosConcept(facet="topic", pref_label="Reviewed", scheme="test")
        session.add(concept)
        session.flush()
        row = ItemConcept(
            item_table="topic_entities",
            item_id=drop.id,
            concept_id=concept.id,
            facet="topic",
            confidence=1.0,
            method="human:reviewer",
        )
        column = "item_id"
    else:
        row = ClassificationCorrection(
            item_table="topic_entities",
            item_id=drop.id,
            facet="topic",
            action="correct",
            prior_concept_ids=[],
            corrected_concept_ids=[],
            actor="human:reviewer",
            source="test",
            rationale="Preserve review history",
        )
        column = "item_id"
    session.add(row)
    session.flush()
    row_id, drop_id, keep_id = row.id, drop.id, keep.id
    if reference == "backlog_candidates":
        task_match = BacklogTaskMatch(
            candidate_id=row.id,
            task_store="example",
            external_task_key="example-task",
            title="Keep the indirect reference",
            state="open",
        )
        session.add(task_match)
    session.commit()

    counts = group_pending(session, now=_NOW)

    assert counts["embedding_merges"] == 1
    assert session.get(TopicEntity, drop_id) is None
    session.expire_all()
    stored = session.get(type(row), row_id)
    assert stored is not None
    if reference.startswith("merge_candidates"):
        assert {stored.entity_a_id, stored.entity_b_id} == {keep_id, unrelated.id}
        assert stored.resolved_by == "human:reviewer"
    else:
        assert getattr(stored, column) == keep_id
    assert session.get(TopicEntity, unrelated.id) is not None
    if reference == "backlog_mapping_reviews":
        assert stored.decision_inputs == {"review": "preserve this decision"}
    if reference == "backlog_candidates":
        assert stored.state == "accepted"
        assert session.get(BacklogTaskMatch, task_match.id).candidate_id == row_id


def test_given_colliding_references_when_group_pending_runs_then_best_values_survive(
    session, absorption
):
    keep, drop, unrelated, _ = absorption
    digest = Digest(period_start=_NOW, period_end=_NOW, content_json={}, content_markdown="")
    concept = SkosConcept(facet="topic", pref_label="Reviewed", scheme="test")
    session.add_all([digest, concept])
    session.flush()
    for entity, rank, method, confidence in (
        (keep, 4, "hmc", 0.99),
        (drop, 1, "human:reviewer", 0.8),
    ):
        session.add_all(
            [
                DigestEntity(digest_id=digest.id, topic_entity_id=entity.id, rank=rank),
                ItemConcept(
                    item_table="topic_entities",
                    item_id=entity.id,
                    concept_id=concept.id,
                    facet="topic",
                    confidence=confidence,
                    method=method,
                ),
                MergeCandidate(
                    entity_a_id=entity.id,
                    entity_b_id=unrelated.id,
                    status="rejected" if entity is drop else "pending",
                    resolved_by="human:reviewer" if entity is drop else None,
                    created_at=_NOW + timedelta(hours=1 if entity is drop else 0),
                ),
            ]
        )
    # A review of the absorbed pair itself must also survive as audit history.
    self_review = MergeCandidate(
        entity_a_id=keep.id, entity_b_id=drop.id, status="accepted", resolved_by="human:reviewer"
    )
    session.add(self_review)
    session.commit()
    keep_id, drop_id, self_review_id = keep.id, drop.id, self_review.id

    counts = group_pending(session, now=_NOW)

    assert counts["embedding_merges"] == 1
    session.expire_all()
    assert session.get(TopicEntity, drop_id) is None
    membership = session.scalars(select(DigestEntity)).one()
    assert (membership.topic_entity_id, membership.rank) == (keep_id, 1)
    label = session.scalars(select(ItemConcept)).one()
    assert (label.item_id, label.method, label.confidence) == (keep_id, "human:reviewer", 0.8)
    review = session.scalar(
        select(MergeCandidate).where(MergeCandidate.entity_b_id == unrelated.id)
    )
    assert (review.entity_a_id, review.status, review.resolved_by) == (
        keep_id,
        "rejected",
        "human:reviewer",
    )
    self_review = session.get(MergeCandidate, self_review_id)
    assert (self_review.entity_a_id, self_review.entity_b_id) == (keep_id, keep_id)


def test_given_metadata_foreign_keys_when_absorption_policies_are_checked_then_every_reference_is_handled():
    from rumen.grouping.references import ENTITY_REFERENCE_POLICIES

    references = {
        (table.name, fk.parent.name)
        for table in Base.metadata.tables.values()
        for fk in table.foreign_keys
        if fk.column.table is TopicEntity.__table__
    }
    assert references == set(ENTITY_REFERENCE_POLICIES)
