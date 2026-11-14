"""Reference policies for absorbing a topic entity without losing its history."""

from collections import defaultdict

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from rumen.models import (
    Base,
    ClassificationCorrection,
    DigestEntity,
    ItemConcept,
    MergeCandidate,
    TopicEntity,
)

# Every FK discovered from metadata needs an explicit policy. A new reference must
# be reviewed rather than silently cascaded away or guessed to be collision-free.
ENTITY_REFERENCE_POLICIES = {
    ("feed_items", "topic_entity_id"): "repoint",
    ("merge_log", "topic_entity_id"): "repoint",
    ("feed_feedback", "topic_entity_id"): "repoint",
    ("backlog_mapping_reviews", "topic_entity_id"): "repoint",
    ("backlog_candidates", "topic_entity_id"): "repoint",
    ("digest_entities", "topic_entity_id"): "digest",
    ("merge_candidates", "entity_a_id"): "pair",
    ("merge_candidates", "entity_b_id"): "pair",
}


def topic_entity_reference_columns():
    return {
        (table.name, fk.parent.name): fk.parent
        for table in Base.metadata.tables.values()
        for fk in table.foreign_keys
        if fk.column.table is TopicEntity.__table__
    }


def repoint_entity_references(session: Session, keep: TopicEntity, drop: TopicEntity) -> None:
    """Move references in the caller's transaction, before deleting the absorbed entity."""
    columns = topic_entity_reference_columns()
    if columns.keys() != ENTITY_REFERENCE_POLICIES.keys():
        raise ValueError("Topic entity references need an absorption policy")
    session.flush()
    models = {mapper.local_table: mapper.class_ for mapper in Base.registry.mappers}
    for key, column in columns.items():
        if ENTITY_REFERENCE_POLICIES[key] == "repoint":
            # These FKs participate in no unique key, so retain every row, including
            # all human mapping reviews and candidates with their dependent history.
            session.execute(
                update(models[column.table]).where(column == drop.id).values({column.name: keep.id})
            )

    for membership in session.scalars(
        select(DigestEntity).where(DigestEntity.topic_entity_id == drop.id)
    ).all():
        existing = session.scalar(
            select(DigestEntity).where(
                DigestEntity.digest_id == membership.digest_id,
                DigestEntity.topic_entity_id == keep.id,
            )
        )
        if existing is None:
            membership.topic_entity_id = keep.id
        else:
            # One membership per digest: retain the better historical rank.
            if membership.rank is not None and (
                existing.rank is None or membership.rank < existing.rank
            ):
                existing.rank = membership.rank
            session.delete(membership)

    _repoint_merge_candidates(session, keep.id, drop.id)
    _repoint_concepts(session, keep, drop)
    # Corrections are append-only review evidence, not a unique projection.
    session.execute(
        update(ClassificationCorrection)
        .where(
            ClassificationCorrection.item_table == "topic_entities",
            ClassificationCorrection.item_id == drop.id,
        )
        .values(item_id=keep.id)
    )
    session.flush()


def _repoint_merge_candidates(session: Session, keep_id: int, drop_id: int) -> None:
    groups = defaultdict(list)
    for row in session.scalars(
        select(MergeCandidate).where(
            MergeCandidate.entity_a_id.in_([keep_id, drop_id])
            | MergeCandidate.entity_b_id.in_([keep_id, drop_id])
        )
    ):
        pair = tuple(
            sorted(
                keep_id if value == drop_id else value
                for value in (row.entity_a_id, row.entity_b_id)
            )
        )
        groups[pair].append(row)
    winners = []
    for pair, rows in groups.items():
        # Preserve human decisions over pending proposals. On collision a rejected
        # decision is the safer veto; within that priority retain the newer review.
        winner = max(
            rows,
            key=lambda row: (
                bool(row.resolved_by),
                row.status == "rejected",
                row.resolved_at or row.created_at,
                row.id,
            ),
        )
        winners.append((pair, winner))
        for row in rows:
            if row is not winner:
                session.delete(row)
    # Delete collisions before UPDATEs, since SQLAlchemy flushes updates first.
    session.flush()
    for (a_id, b_id), row in winners:
        # A self-pair retains the absorbed pair's review as inert audit history.
        row.entity_a_id, row.entity_b_id = a_id, b_id
    session.flush()


def _repoint_concepts(session: Session, keep: TopicEntity, drop: TopicEntity) -> None:
    groups = defaultdict(list)
    for row in session.scalars(
        select(ItemConcept).where(
            ItemConcept.item_table == "topic_entities",
            ItemConcept.item_id.in_([keep.id, drop.id]),
        )
    ):
        groups[row.concept_id].append(row)
    winners = []
    human_topics = []
    for rows in groups.values():
        # Human assignments take precedence; keep the newer human assignment or,
        # for machine projections, the higher confidence then the newer assignment.
        winner = max(
            rows,
            key=lambda row: (
                row.method.startswith("human:"),
                0.0 if row.method.startswith("human:") else row.confidence,
                row.created_at,
                row.id,
            ),
        )
        winners.append(winner)
        if winner.facet == "topic" and winner.method.startswith("human:"):
            human_topics.append(winner)
        for row in rows:
            if row is not winner:
                session.delete(row)
    if human_topics:
        latest = max(human_topics, key=lambda row: (row.created_at, row.id))
        # Retain the scalar derived when the most recent human topic was reviewed.
        if latest.item_id == drop.id:
            keep.category = drop.category
    session.flush()
    for row in winners:
        row.item_id = keep.id
