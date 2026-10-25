"""Ingestion orchestration — high-water-mark bookkeeping over source adapters (T-03).

``run_ingestion`` computes the incremental ``since`` from the last *completed* run for
the source, persists new ``RawItem``s into ``feed_items`` (deduped on ``(source, source_id)``),
and records a ``feed_runs`` row. Adapter failures are isolated: a raising adapter records a
``failed`` run (with ``error_message``) and does **not** advance the high-water mark, so the
next run retries from the last good cursor and a scheduler retry never double-ingests.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from math import isfinite

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.adapters.base import (
    ChangeSourceAdapter,
    FetchResult,
    RawItem,
    SourceAdapter,
    StatefulFetchError,
    StatefulSourceAdapter,
)
from rumen.models import FeedItem, FeedRun


def _utcnow() -> datetime:
    return datetime.now(UTC)


def last_high_water_mark(session: Session, source: str) -> datetime | None:
    """Return the ``high_water_mark`` of the most recent *completed* run for ``source``.

    Failed/running runs are ignored so a failure never advances (or regresses) the cursor.
    """
    hwm, _ = last_completed_context(session, source)
    return hwm


def last_completed_context(session: Session, source: str) -> tuple[datetime | None, dict | None]:
    """Return the latest completed high-water mark and copied opaque source state."""
    row = session.execute(
        select(FeedRun.high_water_mark, FeedRun.source_state)
        .where(FeedRun.source == source, FeedRun.status == "completed")
        .order_by(FeedRun.started_at.desc())
        .limit(1)
    ).one_or_none()
    if row is None:
        return None, None
    hwm, state = row
    return datetime.fromisoformat(hwm) if hwm else None, deepcopy(
        state
    ) if state is not None else None


def _is_bounded_json(value: object, *, depth: int = 0) -> bool:
    """Reject non-JSON and unbounded adapter payloads before database persistence."""
    if depth > 8:
        return False
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, str):
        return len(value) <= 4096
    if isinstance(value, list):
        return len(value) <= 100 and all(_is_bounded_json(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 100 and all(
            isinstance(key, str) and len(key) <= 128 and _is_bounded_json(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def _validate_fetch_result(result: object) -> FetchResult:
    if not isinstance(result, FetchResult):
        raise TypeError("stateful adapter must return FetchResult")
    if not isinstance(result.items, list) or not all(
        isinstance(item, RawItem) for item in result.items
    ):
        raise TypeError("stateful adapter FetchResult.items must be a list of RawItem")
    if not isinstance(result.next_state, dict) or not _is_bounded_json(result.next_state):
        raise TypeError("stateful adapter FetchResult.next_state must be a bounded JSON object")
    if not isinstance(result.diagnostics, dict) or not _is_bounded_json(result.diagnostics):
        raise TypeError("stateful adapter FetchResult.diagnostics must be a bounded JSON object")
    return result


def run_ingestion(
    session: Session,
    adapter: SourceAdapter | ChangeSourceAdapter,
    *,
    since: datetime | None = None,
    commit_batch_size: int = 500,
) -> FeedRun:
    """Fetch + persist new items for one adapter, recording a ``feed_runs`` row.

    ``since`` defaults to the last completed run's high-water mark. Returns the run row
    (``status`` ``completed`` or ``failed``); adapter exceptions are captured, not raised,
    so one failing source never blocks the others.

    Inserts commit every ``commit_batch_size`` items (design D-14) so a large source
    doesn't lose all its work if interrupted mid-fetch. The ``(source,
    source_id)`` dedup makes a re-run skip already-persisted rows, and the high-water mark is
    only advanced on the final completion commit, so a partial run never regresses the cursor.
    """
    source = adapter.source_name()
    previous_state: dict | None = None
    if since is None:
        since, previous_state = last_completed_context(session, source)

    run = FeedRun(source=source, started_at=_utcnow(), status="running")
    session.add(run)
    session.flush()

    stateful = isinstance(adapter, StatefulSourceAdapter)
    try:
        if isinstance(adapter, ChangeSourceAdapter):
            return _run_changes(session, adapter, run, since, commit_batch_size)
        if stateful:
            result = _validate_fetch_result(adapter.fetch_new_with_state(since, previous_state))
            items = result.items
        else:
            items = adapter.fetch_new(since)  # type: ignore[union-attr]
            result = None
    except StatefulFetchError as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.diagnostics = exc.diagnostics if _is_bounded_json(exc.diagnostics) else None
        run.completed_at = _utcnow()
        run.high_water_mark = None
        session.commit()
        return run
    except Exception as exc:  # noqa: BLE001 — isolate adapter failures per Risk #1
        run.status = "failed"
        run.error_message = str(exc)
        run.completed_at = _utcnow()
        run.high_water_mark = None  # never advance on failure
        session.commit()
        return run

    max_published = since
    inserted = 0
    for item in items:
        if max_published is None or item.published_at > max_published:
            max_published = item.published_at
        exists = session.scalars(
            select(FeedItem.id).where(
                FeedItem.source == source, FeedItem.source_id == item.source_id
            )
        ).first()
        if exists:
            continue
        session.add(
            FeedItem(
                source=source,
                source_id=item.source_id,
                url=item.url,
                canonical_url=None,
                title=item.title,
                content=item.content,
                author=item.author,
                published_at=item.published_at,
                category=item.category,
                categories=item.categories,
                domains=item.domains,
                link_metadata=item.metadata or None,
            )
        )
        inserted += 1
        # Persist in batches so a long source's progress survives interruption. The run row
        # stays ``running`` (high_water_mark unset) until completion, so this only commits
        # deduped feed_items — never a premature cursor advance.
        if inserted % commit_batch_size == 0:
            session.commit()

    run.items_fetched = len(items)
    run.items_passed_filter = inserted
    # Safe before downstream stages: the unique (source, source_id) constraint makes
    # retries idempotent even when classification/grouping/ranking subsequently fails.
    run.high_water_mark = max_published.isoformat() if max_published else None
    run.completed_at = _utcnow()
    # A tripped staleness guard is a FAILED run, not a completed one carrying a note.
    # The previous design recorded status="completed" with the diagnostic in error_message,
    # which made a month of zero ingestion indistinguishable from healthy runs in every query
    # that filters on status — bookmarks silently ingested nothing from 2026-07-06 to 08-02.
    warning = getattr(adapter, "last_staleness_warning", None)
    if warning:
        run.status = "failed"
        run.error_message = warning
    else:
        run.status = "completed"
        if result is not None:
            run.source_state = result.next_state
            run.diagnostics = result.diagnostics
    session.commit()
    return run


def _set_item_projection(row: FeedItem, item: RawItem) -> None:
    """Update mutable source fields without replacing the pointer/audit row."""
    row.url, row.title, row.content, row.author = item.url, item.title, item.content, item.author
    row.published_at = item.published_at
    row.category, row.categories, row.domains = item.category, item.categories, item.domains
    row.link_metadata = item.metadata or None
    row.embedding = None
    row.classification_confidence = None
    row.classification_method = None
    row.facets_classified_at = None


def _run_changes(
    session: Session,
    adapter: ChangeSourceAdapter,
    run: FeedRun,
    since: datetime | None,
    commit_batch_size: int,
) -> FeedRun:
    """Apply an ordered mutable-source log; publish its cursor only at completion."""
    try:
        changes = adapter.fetch_changes(since)
        if changes != sorted(changes, key=lambda change: change.occurred_at):
            raise ValueError("change adapter returned non-monotonic changes")
        if since is not None and any(change.occurred_at <= since for change in changes):
            raise ValueError("change adapter returned a change at or before the cursor")
        applied, cursor = 0, since
        for change in changes:
            row = session.scalars(
                select(FeedItem).where(
                    FeedItem.source == adapter.source_name(), FeedItem.source_id == change.source_id
                )
            ).first()
            if change.operation == "tombstone":
                if row is not None:
                    row.source_deleted_at = change.occurred_at
                    row.source_last_seen_at = change.occurred_at
                    row.source_revision_digest = change.revision_digest
                    row.status = "deleted"
            elif row is None:
                item = change.item
                assert item is not None
                session.add(
                    FeedItem(
                        source=adapter.source_name(),
                        source_id=change.source_id,
                        url=item.url,
                        canonical_url=None,
                        title=item.title,
                        content=item.content,
                        author=item.author,
                        published_at=item.published_at,
                        category=item.category,
                        categories=item.categories,
                        domains=item.domains,
                        link_metadata=item.metadata or None,
                        source_revision_digest=change.revision_digest,
                        source_last_seen_at=change.occurred_at,
                    )
                )
                applied += 1
            else:
                row.source_last_seen_at = change.occurred_at
                if row.source_revision_digest != change.revision_digest:
                    assert change.item is not None
                    _set_item_projection(row, change.item)
                    row.source_revision_digest = change.revision_digest
                    applied += 1
                row.source_deleted_at = None
                row.status = "active"
            if row is not None and row.topic_entity_id is not None:
                # Re-project only the affected entity; deleted rows are excluded by recompute.
                from rumen.grouping.merge import recompute_entity
                from rumen.models import TopicEntity

                entity = session.get(TopicEntity, row.topic_entity_id)
                if entity is not None:
                    recompute_entity(session, entity)
            cursor = change.occurred_at
            if applied and applied % commit_batch_size == 0:
                session.commit()
        run.items_fetched, run.items_passed_filter = len(changes), applied
        run.high_water_mark = cursor.isoformat() if cursor else None
        run.completed_at, run.status = _utcnow(), "completed"
        session.commit()
    except Exception as exc:  # noqa: BLE001
        run.status, run.error_message, run.completed_at, run.high_water_mark = (
            "failed",
            str(exc),
            _utcnow(),
            None,
        )
        session.commit()
    return run
