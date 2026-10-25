"""False-negative audit — log every exclusion, never hard-delete (design D-12).

Any drop (classified out-of-scope, keyword-miss, suppressed, low-relevance) is recorded
in ``dropped_items`` with a reason and confidence. A periodic "what got left out" pass
surfaces the **low-confidence** drops for review — the mirror of the merge review queue.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.models import DroppedItem


def log_drop(
    session: Session,
    *,
    source: str,
    source_id: str,
    reason: str,
    url: str | None = None,
    title: str | None = None,
    drop_confidence: float | None = None,
) -> DroppedItem:
    """Record a drop (never a hard delete). Returns the persisted row."""
    row = DroppedItem(
        source=source,
        source_id=source_id,
        url=url,
        title=title,
        drop_reason=reason,
        drop_confidence=drop_confidence,
    )
    session.add(row)
    session.flush()
    return row


def surface_low_confidence_drops(
    session: Session, *, max_confidence: float, limit: int | None = None
) -> list[DroppedItem]:
    """Return unreviewed drops with ``drop_confidence <= max_confidence`` (lowest first).

    These are the prime false-negative candidates — the model was *least* sure it should
    have dropped them. High-confidence drops are intentionally excluded.
    """
    stmt = (
        select(DroppedItem)
        .where(
            DroppedItem.reviewed == 0,
            DroppedItem.drop_confidence.is_not(None),
            DroppedItem.drop_confidence <= max_confidence,
        )
        .order_by(DroppedItem.drop_confidence.asc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))
