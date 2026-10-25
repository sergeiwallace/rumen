"""Read a locally synchronized Reddit saved-items cache into ``RawItem`` values.

The operator-run ``scripts/ops/reddit_sync.py`` is the only component that makes an
authenticated Reddit request.  This adapter is deliberately read-only.  Reddit's saved
listing omits a saved timestamp, so ``created_at`` is used as ``RawItem.published_at`` and
``synced_at`` only measures cache freshness.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from rumen.adapters.base import RawItem

logger = logging.getLogger(__name__)


class RedditSchemaError(RuntimeError):
    """Configured Reddit saved-items cache lacks the required schema."""


_COLUMNS = (
    "fullname",
    "kind",
    "title",
    "body",
    "author",
    "permalink",
    "created_at",
    "subreddit",
    "synced_at",
)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


class RedditSavedAdapter:
    """Read a Reddit saved-items SQLite cache selected by operator configuration."""

    def __init__(
        self,
        db_path: str,
        *,
        table: str = "reddit_saved_items",
        staleness_threshold: timedelta = timedelta(hours=48),
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path
        self.table = table
        self.staleness_threshold = staleness_threshold
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self.last_staleness_warning: str | None = None

    def source_name(self) -> str:
        return "reddit_saved"

    @property
    def _table_safe(self) -> str:
        if not self.table.isidentifier():
            raise ValueError(f"reddit table must be an identifier; got {self.table!r}")
        return self.table

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _validate_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(f"PRAGMA table_info({self._table_safe})").fetchall()
        if not rows:
            raise RedditSchemaError(f"reddit table {self.table!r} does not exist")
        missing = sorted(set(_COLUMNS) - {row["name"] for row in rows})
        if missing:
            raise RedditSchemaError(
                f"reddit table {self.table!r} is missing required columns: {', '.join(missing)}"
            )

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        columns = ", ".join(_COLUMNS)
        conn = self._connect()
        try:
            self._validate_schema(conn)
            rows = conn.execute(f"SELECT {columns} FROM {self._table_safe}").fetchall()
        finally:
            conn.close()

        max_synced: datetime | None = None
        items: list[RawItem] = []
        for row in rows:
            synced = _parse_dt(row["synced_at"])
            if synced and (max_synced is None or synced > max_synced):
                max_synced = synced
            created = _parse_dt(row["created_at"])
            if created is None:
                logger.warning(
                    "reddit_saved: skipping fullname=%s with missing/unparseable created_at",
                    row["fullname"],
                )
                continue
            if since is not None and created <= since:
                continue
            permalink = row["permalink"]
            url = (
                permalink if permalink.startswith("http") else f"https://www.reddit.com{permalink}"
            )
            items.append(
                RawItem(
                    source=self.source_name(),
                    source_id=row["fullname"],
                    url=url,
                    title=row["title"] or "",
                    published_at=created,
                    content=row["body"],
                    author=row["author"],
                    metadata={"kind": row["kind"], "subreddit": row["subreddit"]},
                )
            )
        self.last_staleness_warning = self._staleness_warning(max_synced)
        return items

    def _staleness_warning(self, max_synced: datetime | None) -> str | None:
        if max_synced is None:
            return (
                "SYNC_STALE: Reddit saved-items cache has no synced_at values — sync may be broken."
            )
        age = self._now_fn() - max_synced
        if age > self.staleness_threshold:
            return (
                f"SYNC_STALE: Reddit saved-items cache last synced {max_synced.isoformat()} "
                f"({age} ago, > {self.staleness_threshold}) — upstream saved-items sync may be broken."
            )
        return None
