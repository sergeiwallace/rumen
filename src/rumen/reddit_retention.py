"""Retention and termination for OAuth Reddit content."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rumen.adapters.reddit_subreddits import RedditGateway, RedditSubredditSettings
from rumen.grouping.merge import recompute_entity
from rumen.models import (
    ClassificationCorrection,
    Digest,
    DigestEntity,
    DroppedItem,
    FeedFeedback,
    FeedItem,
    FeedRun,
    ItemConcept,
    MergeCandidate,
    MergeLog,
    SourceMaintenanceRun,
    TopicEntity,
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
        item_ids = [item.id for item in items]
        entity_ids = {item.topic_entity_id for item in items if item.topic_entity_id is not None}
        digest_ids = (
            session.scalars(
                select(DigestEntity.digest_id).where(DigestEntity.topic_entity_id.in_(entity_ids))
            ).all()
            if entity_ids
            else []
        )
        session.execute(
            delete(ItemConcept).where(
                ItemConcept.item_table == "feed_items", ItemConcept.item_id.in_(item_ids)
            )
        )
        session.execute(
            delete(ClassificationCorrection).where(
                ClassificationCorrection.item_table == "feed_items",
                ClassificationCorrection.item_id.in_(item_ids),
            )
        )
        session.execute(delete(MergeLog).where(MergeLog.feed_item_id.in_(item_ids)))
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
            session.execute(delete(DigestEntity).where(DigestEntity.topic_entity_id == entity_id))
            if survivors:
                recompute_entity(session, entity, items=survivors)
                entity.rank_position = None
                continue
            session.execute(
                delete(ItemConcept).where(
                    ItemConcept.item_table == "topic_entities", ItemConcept.item_id == entity_id
                )
            )
            session.execute(
                delete(ClassificationCorrection).where(
                    ClassificationCorrection.item_table == "topic_entities",
                    ClassificationCorrection.item_id == entity_id,
                )
            )
            session.execute(delete(FeedFeedback).where(FeedFeedback.topic_entity_id == entity_id))
            session.execute(delete(MergeLog).where(MergeLog.topic_entity_id == entity_id))
            session.execute(
                delete(MergeCandidate).where(
                    (MergeCandidate.entity_a_id == entity_id)
                    | (MergeCandidate.entity_b_id == entity_id)
                )
            )
            session.delete(entity)
        # A digest can contain copied text, so any digest associated with changed entities is unsafe.
        if digest_ids:
            session.execute(delete(DigestEntity).where(DigestEntity.digest_id.in_(digest_ids)))
            session.execute(delete(Digest).where(Digest.id.in_(digest_ids)))
