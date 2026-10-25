"""Keyword filter stage (design Stage 3 / T-06).

Applies the ``feed_keywords`` inclusion + suppression lists:

- **Keyword-scored sources** (GitHub/HF/Reddit/HN — anything not in ``bypass_sources``)
  drop when they match **zero** inclusion keywords.
- **Bookmarks bypass** the inclusion drop (already user-curated) but are still annotated
  with any inclusion keywords they match.
- A **suppress** match drops an item of **any** source, even if it also matched an
  inclusion keyword.

Every drop is logged to ``dropped_items`` (``keyword_miss`` / ``suppressed``) and marked
``dropped`` on the ``feed_items`` row; nothing is silently lost. Matched inclusion keywords
are stored on active items for the "why was this included?" transparency.

``open_when_empty`` (default **True**): with an empty inclusion list, keyword-scored
items are **kept** (an unconfigured keyword list should not silently drop the whole
corpus). Set it ``False`` to require an explicit match (empty list → drop all
keyword-scored items). Suppression always applies regardless.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.db.engine import batched_write_mode
from rumen.dropped_items import log_drop
from rumen.models import FeedItem, FeedKeyword

_BYPASS_SOURCES = frozenset({"bookmarks"})


def _matches(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw.lower() in lowered]


def apply_keyword_filter(
    session: Session,
    *,
    open_when_empty: bool = True,
    bypass_sources: frozenset[str] = _BYPASS_SOURCES,
    commit_batch_size: int = 500,
) -> dict[str, int]:
    """Apply inclusion/suppression to all ``feed_items``. Returns stage counts.

    Commits every ``commit_batch_size`` items (design D-14) so progress is durable across
    interruption.
    """
    inclusion = list(
        session.scalars(select(FeedKeyword.keyword).where(FeedKeyword.keyword_type == "include"))
    )
    suppress = list(
        session.scalars(select(FeedKeyword.keyword).where(FeedKeyword.keyword_type == "suppress"))
    )

    counts = {"kept": 0, "dropped_keyword_miss": 0, "dropped_suppressed": 0}
    # expire_on_commit=False: without it, each batched commit expires every loaded row and the
    # next `item.title`/`.content` access re-fetches it one-at-a-time over the network.
    with batched_write_mode(session):
        for i, item in enumerate(
            session.scalars(select(FeedItem).where(FeedItem.status != "dropped")).all(), start=1
        ):
            text = f"{item.title or ''} {item.content or ''}"
            matched_inclusion = _matches(text, inclusion)
            item.matched_keywords = matched_inclusion or None

            # Suppression drops any source.
            if _matches(text, suppress):
                _drop(session, item, "suppressed")
                counts["dropped_suppressed"] += 1
            # Inclusion drop applies only to keyword-scored (non-bypass) sources.
            elif (
                item.source not in bypass_sources
                and not matched_inclusion
                and (inclusion or not open_when_empty)
            ):
                _drop(session, item, "keyword_miss")
                counts["dropped_keyword_miss"] += 1
            else:
                counts["kept"] += 1

            if i % commit_batch_size == 0:
                session.commit()

        session.commit()
    return counts


def _drop(session: Session, item: FeedItem, reason: str) -> None:
    log_drop(
        session,
        source=item.source,
        source_id=item.source_id,
        url=item.url,
        title=item.title,
        reason=reason,
        drop_confidence=1.0,
    )
    item.status = "dropped"
