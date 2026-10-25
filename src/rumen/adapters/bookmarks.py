"""Bookmarks connector — reads a synced-bookmarks SQLite DB into ``RawItem``s (T-04, D-6).

Open-core boundary: the **DB path and schema are operator config**, never a baked-in
personal path. The connector reads a documented column subset of the ``bookmarks`` table that
the upstream bookmarks tool maintains, ingests the **full corpus** (no category drop at ingest — filtering
is presentation-layer, D-6), and carries the pre-computed ``primary_category``/``categories``/
``domains`` tags through onto the item.

Sync-staleness (Risk #8): an empty incremental fetch is ambiguous — "caught up" vs
"upstream sync broke". When the DB's max ``synced_at`` exceeds a configured threshold the
adapter records a warning (``last_staleness_warning``) that ``run_ingestion`` persists onto the
``feed_runs`` row for ``feed status`` to surface.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from rumen.adapters.base import RawItem

logger = logging.getLogger(__name__)


class BookmarksSchemaError(RuntimeError):
    """Configured bookmarks database lacks the required schema."""


# Documented column subset read from the ``bookmarks`` table.
_COLUMNS = (
    "tweet_id",
    "text",
    "author_handle",
    "posted_at",
    "synced_at",
    "primary_category",
    "categories",
    "domains",
    "github_urls",
    "links_json",
    "article_title",
    "article_text",
    "article_site",
    "quoted_tweet_json",
)


# The bookmarks tool stores `posted_at` in X/Twitter's native format ("Fri Apr 10 03:30:28 +0000
# 2026"); `synced_at` is ISO-8601. Try both so neither column silently fails to parse.
_TWITTER_FMT = "%a %b %d %H:%M:%S %z %Y"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    # strptime() takes format positionally (no keyword arg), so a functools.partial can't
    # supply it out of order here — a lambda is required. The result is made tz-aware two
    # lines below regardless of which parser matched, so DTZ007 doesn't apply.
    for parse in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, _TWITTER_FMT),  # noqa: DTZ007
    ):
        try:
            dt = parse(value)
        except (ValueError, TypeError):
            continue
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    return None


def _parse_json_list(value: str | None) -> list | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_json_object(value: str | None) -> dict | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


class BookmarksAdapter:
    """Reads a synced-bookmarks SQLite DB (path + staleness threshold are config)."""

    def __init__(
        self,
        db_path: str,
        *,
        table: str = "bookmarks",
        staleness_threshold: timedelta = timedelta(hours=48),
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path
        self.table = table
        self.staleness_threshold = staleness_threshold
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self.last_staleness_warning: str | None = None

    def source_name(self) -> str:
        return "bookmarks"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        columns = ", ".join(self._COLUMNS_SAFE)
        conn = self._connect()
        try:
            self._validate_schema(conn)
            rows = conn.execute(f"SELECT {columns} FROM {self._table_safe}").fetchall()
        finally:
            conn.close()

        items: list[RawItem] = []
        max_synced: datetime | None = None
        for row in rows:
            synced = _parse_dt(row["synced_at"])
            if synced and (max_synced is None or synced > max_synced):
                max_synced = synced

            published = _parse_dt(row["posted_at"])
            if published is None:
                # Failure path: a NULL/unparseable posted_at is skipped, not fatal.
                logger.warning(
                    "bookmarks: skipping tweet_id=%s with missing/unparseable posted_at",
                    row["tweet_id"],
                )
                continue
            if since is not None and published <= since:
                continue

            metadata = {
                "github_urls": _parse_json_list(row["github_urls"]) or [],
                "links": _parse_json_list(row["links_json"]) or [],
            }
            for column in ("article_title", "article_text", "article_site"):
                if value := row[column]:
                    metadata[column] = value
            if quoted_tweet := _parse_json_object(row["quoted_tweet_json"]):
                metadata["quoted_tweet_json"] = quoted_tweet

            items.append(
                RawItem(
                    source=self.source_name(),
                    source_id=str(row["tweet_id"]),
                    url=f"https://x.com/{row['author_handle']}/status/{row['tweet_id']}",
                    title=row["text"] or "",
                    published_at=published,
                    content=row["text"],
                    author=row["author_handle"],
                    category=row["primary_category"],
                    categories=_parse_json_list(row["categories"]),
                    domains=_parse_json_list(row["domains"]),
                    metadata=metadata,
                )
            )

        self.last_staleness_warning = self._staleness_warning(max_synced)
        return items

    def _validate_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(f"PRAGMA table_info({self._table_safe})").fetchall()
        if not rows:
            raise BookmarksSchemaError(f"bookmarks table {self.table!r} does not exist")
        missing = sorted(set(_COLUMNS) - {row["name"] for row in rows})
        if missing:
            raise BookmarksSchemaError(
                f"bookmarks table {self.table!r} is missing required columns: {', '.join(missing)}"
            )

    def _staleness_warning(self, max_synced: datetime | None) -> str | None:
        now = self._now_fn()
        if max_synced is None:
            return "SYNC_STALE: bookmarks DB has no synced_at values — upstream sync may be broken."
        age = now - max_synced
        if age > self.staleness_threshold:
            return (
                f"SYNC_STALE: bookmarks DB last synced {max_synced.isoformat()} "
                f"({age} ago, > {self.staleness_threshold}) — upstream X sync may be broken "
                "(GraphQL doc_id rotation / auth_token expiry)."
            )
        return None

    # Identifier allow-listing so the interpolated table/column names can never carry
    # untrusted SQL (they are code constants, but keep the guard explicit).
    @property
    def _table_safe(self) -> str:
        if not self.table.isidentifier():
            raise ValueError(f"bookmarks table must be an identifier; got {self.table!r}")
        return self.table

    @property
    def _COLUMNS_SAFE(self) -> tuple[str, ...]:
        return _COLUMNS
