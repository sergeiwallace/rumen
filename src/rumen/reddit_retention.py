"""Retention and termination for OAuth Reddit content."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from rumen.adapters.reddit_subreddits import RedditGateway, RedditSubredditSettings
from rumen.grouping.merge import recompute_entity
from rumen.grouping.references import topic_entity_reference_columns
from rumen.models import (
    BacklogCandidate,
    BacklogCandidateDisposition,
    BacklogCandidateEvidence,
    Base,
    ClassificationCorrection,
    Digest,
    DigestEntity,
    DroppedItem,
    FeedItem,
    FeedRun,
    ItemConcept,
    SourceMaintenanceRun,
    TopicEntity,
)

# Every direct and transitive FK needs a reviewed policy before any purge writes.
# Human reviews with free text, snapshots or undo images cannot safely outlive the
# retained content. Delete that local history, including accepted decisions; this
# does not change external tasks. Numeric feedback can survive on a live entity.
PURGE_REFERENCE_POLICIES = {
    ("feed_items", "topic_entity_id"): "recompute",
    ("merge_log", "feed_item_id"): "delete",  # May contain copied match evidence.
    ("merge_log", "topic_entity_id"): "delete",
    ("feed_feedback", "topic_entity_id"): "delete_orphan",  # Content-free rating.
    ("digest_entities", "topic_entity_id"): "digests",  # Delete copied digest text too.
    ("digest_entities", "digest_id"): "delete",
    ("merge_candidates", "entity_a_id"): "delete",  # Free-text review reason.
    ("merge_candidates", "entity_b_id"): "delete",
    ("backlog_mapping_reviews", "topic_entity_id"): "delete",  # Decision inputs/errors.
    ("backlog_candidates", "topic_entity_id"): "candidates",  # Whole snapshot family.
    ("backlog_candidates", "digest_id"): "candidates",
    ("backlog_candidate_evidence", "feed_item_id"): "candidates",  # Excerpts taint parent.
    ("backlog_candidate_evidence", "candidate_id"): "delete",
    ("backlog_task_matches", "candidate_id"): "delete",  # Titles and match evidence.
    ("backlog_candidate_recommendations", "candidate_id"): "delete",  # Rationale.
    ("backlog_candidate_dispositions", "target_candidate_id"): "delete",  # Undo images.
    ("backlog_candidate_task_links", "origin_candidate_id"): "delete",  # Observed titles.
    ("backlog_candidate_task_links", "last_reconciled_candidate_id"): "delete",
    ("backlog_candidate_promotions", "candidate_id"): "delete",  # Payloads and receipts.
    ("backlog_candidates", "superseded_by_candidate_id"): "null",  # Content-free links.
    ("backlog_candidates", "merge_target_candidate_id"): "null",
    ("youtube_video_states", "feed_item_id"): "delete",  # Required owning pointer.
    ("youtube_video_states", "transcript_feed_item_id"): "null",  # No transcript text.
    ("classification_corrections", "supersedes"): "null",  # Content-free history link.
}


def _purge_reference_columns():
    columns = topic_entity_reference_columns()
    columns.update(
        {
            (table.name, fk.parent.name): fk.parent
            for table in Base.metadata.tables.values()
            for fk in table.foreign_keys
            if fk.column.table
            in {
                FeedItem.__table__,
                BacklogCandidate.__table__,
                Digest.__table__,
                ClassificationCorrection.__table__,
            }
        }
    )
    if columns.keys() != PURGE_REFERENCE_POLICIES.keys():
        raise ValueError("Foreign key references need a Reddit purge policy")
    return columns


def _purge_references(session, columns, target, ids, *, orphan_ids=()):
    models = {mapper.local_table: mapper.class_ for mapper in Base.registry.mappers}
    for key, column in columns.items():
        if not any(fk.column.table is target for fk in column.foreign_keys):
            continue
        policy = PURGE_REFERENCE_POLICIES[key]
        model = models[column.table]
        if policy == "delete":
            session.execute(delete(model).where(column.in_(ids)))
        elif policy == "null":
            session.execute(update(model).where(column.in_(ids)).values({column.name: None}))
        elif policy == "delete_orphan":
            session.execute(delete(model).where(column.in_(orphan_ids)))
        # Snapshot families, digests and entity survivorship are handled below.


def _purge_candidates(session, columns, item_ids, entity_ids, digest_ids):
    fingerprints = set(
        session.scalars(
            select(BacklogCandidate.fingerprint).where(
                BacklogCandidate.topic_entity_id.in_(entity_ids)
                | BacklogCandidate.digest_id.in_(digest_ids)
                | BacklogCandidate.id.in_(
                    select(BacklogCandidateEvidence.candidate_id).where(
                        BacklogCandidateEvidence.feed_item_id.in_(item_ids)
                    )
                )
            )
        )
    )
    if not fingerprints:
        return
    # Superseded revisions share snapshots even when their entity/digest FK is NULL.
    candidate_ids = set(
        session.scalars(
            select(BacklogCandidate.id).where(BacklogCandidate.fingerprint.in_(fingerprints))
        )
    )
    snapshot_keys = {str(candidate_id) for candidate_id in candidate_ids}
    history_fingerprints = set(fingerprints)
    for frame in session.scalars(select(BacklogCandidateDisposition)):
        if (
            frame.target_candidate_id in candidate_ids
            or candidate_ids.intersection(frame.affected_candidate_ids)
            or snapshot_keys.intersection(frame.preimage)
            or snapshot_keys.intersection(frame.postimage)
        ):
            history_fingerprints.add(frame.fingerprint)
    # Controls may have no FK at all. Remove the whole affected undo/redo stack so
    # neither copied content nor an invalid restorable preimage remains.
    session.execute(
        delete(BacklogCandidateDisposition).where(
            BacklogCandidateDisposition.fingerprint.in_(history_fingerprints)
        )
    )
    _purge_references(session, columns, BacklogCandidate.__table__, candidate_ids)
    session.execute(delete(BacklogCandidate).where(BacklogCandidate.id.in_(candidate_ids)))


def _purge_classifications(session, columns, item_table, ids, *, concepts=True):
    correction_ids = list(
        session.scalars(
            select(ClassificationCorrection.id).where(
                ClassificationCorrection.item_table == item_table,
                ClassificationCorrection.item_id.in_(ids),
            )
        )
    )
    _purge_references(session, columns, ClassificationCorrection.__table__, correction_ids)
    session.execute(
        delete(ClassificationCorrection).where(ClassificationCorrection.id.in_(correction_ids))
    )
    if concepts:
        session.execute(
            delete(ItemConcept).where(
                ItemConcept.item_table == item_table, ItemConcept.item_id.in_(ids)
            )
        )


def _now() -> datetime:
    return datetime.now(UTC)


def _removed(obj: object) -> bool:
    if getattr(obj, "author", object()) is None:
        body = getattr(obj, "body", getattr(obj, "selftext", None))
        return body in {"[deleted]", "[removed]", None}
    return getattr(obj, "body", getattr(obj, "selftext", "")) in {"[deleted]", "[removed]"}


class RedditRetentionMaintainer:
    def __init__(self, settings: RedditSubredditSettings, gateway: RedditGateway) -> None:
        self.settings = settings
        self.gateway = gateway

    def source_name(self) -> str:
        return f"reddit:{self.settings.profile}"

    def maintain(self, session: Session) -> SourceMaintenanceRun:
        run = SourceMaintenanceRun(source=self.source_name(), started_at=_now(), status="running")
        session.add(run)
        session.flush()
        reasons: Counter[str] = Counter()
        try:
            expired = list(
                session.scalars(
                    select(FeedItem).where(
                        FeedItem.source == self.source_name(),
                        FeedItem.published_at
                        < _now() - timedelta(hours=self.settings.retention_hours),
                    )
                )
            )
            self._purge(session, expired)
            reasons["expired"] += len(expired)
            session.commit()  # expiry is a compliance action even when provider is down.
            retained = list(
                session.scalars(select(FeedItem).where(FeedItem.source == self.source_name()))
            )
            run.items_checked = len(retained)
            response = [
                obj
                for start in range(0, len(retained), 100)
                for obj in self.gateway.revalidate(
                    [item.source_id for item in retained[start : start + 100]]
                )
            ]
            returned = {getattr(obj, "name", None): obj for obj in response}
            removed = [
                item
                for item in retained
                if item.source_id not in returned or _removed(returned[item.source_id])
            ]
            reasons["missing"] += sum(item.source_id not in returned for item in removed)
            reasons["removed"] += sum(item.source_id in returned for item in removed)
            self._purge(session, removed)
            run.items_purged = len(expired) + len(removed)
            run.reason_counts = dict(reasons)
            run.status = "completed"
            run.completed_at = _now()
            session.commit()
        except Exception:
            session.rollback()
            # Avoid provider payloads or credentials in durable errors.
            run = session.get(SourceMaintenanceRun, run.id)
            if run is None:
                raise
            run.status = "failed"
            run.completed_at = _now()
            run.items_purged = len(expired) if "expired" in locals() else 0
            run.reason_counts = dict(reasons)
            run.error_message = "Reddit lifecycle maintenance failed"
            session.commit()
        return run

    def terminate(self, session: Session) -> int:
        items = list(session.scalars(select(FeedItem).where(FeedItem.source == self.source_name())))
        self._purge(session, items)
        session.execute(delete(DroppedItem).where(DroppedItem.source == self.source_name()))
        session.execute(delete(FeedRun).where(FeedRun.source == self.source_name()))
        session.execute(
            delete(SourceMaintenanceRun).where(SourceMaintenanceRun.source == self.source_name())
        )
        session.commit()
        return len(items)

    @staticmethod
    def _purge(session: Session, items: list[FeedItem]) -> None:
        if not items:
            return
        columns = _purge_reference_columns()
        item_ids = [item.id for item in items]
        entity_ids = {item.topic_entity_id for item in items if item.topic_entity_id is not None}
        surviving_entities = set(
            session.scalars(
                select(FeedItem.topic_entity_id).where(
                    FeedItem.topic_entity_id.in_(entity_ids), FeedItem.id.not_in(item_ids)
                )
            )
        )
        orphan_ids = entity_ids - surviving_entities
        digest_ids = (
            session.scalars(
                select(DigestEntity.digest_id).where(DigestEntity.topic_entity_id.in_(entity_ids))
            ).all()
            if entity_ids
            else []
        )
        _purge_candidates(session, columns, item_ids, entity_ids, digest_ids)
        _purge_classifications(session, columns, "feed_items", item_ids)
        # Review rationale/context may quote the removed item even on a live entity.
        _purge_classifications(session, columns, "topic_entities", entity_ids, concepts=False)
        session.execute(
            delete(ItemConcept).where(
                ItemConcept.item_table == "topic_entities", ItemConcept.item_id.in_(orphan_ids)
            )
        )
        _purge_references(session, columns, FeedItem.__table__, item_ids)
        _purge_references(
            session, columns, TopicEntity.__table__, entity_ids, orphan_ids=orphan_ids
        )
        # A digest can contain copied text, so any digest associated with changed entities is unsafe.
        if digest_ids:
            _purge_references(session, columns, Digest.__table__, digest_ids)
            session.execute(delete(Digest).where(Digest.id.in_(digest_ids)))
        session.execute(
            delete(DroppedItem).where(
                DroppedItem.source == items[0].source,
                DroppedItem.source_id.in_([item.source_id for item in items]),
            )
        )
        session.execute(delete(FeedItem).where(FeedItem.id.in_(item_ids)))
        session.flush()
        for entity_id in entity_ids:
            entity = session.get(TopicEntity, entity_id)
            survivors = list(
                session.scalars(select(FeedItem).where(FeedItem.topic_entity_id == entity_id))
            )
            if survivors:
                # A cached display title can quote a purged constituent.
                entity.display_title = None
                recompute_entity(session, entity, items=survivors)
                entity.rank_position = None
                continue
            session.delete(entity)
