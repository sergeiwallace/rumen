"""Private schema-versioned SQLite cache for committed Paperpile snapshots."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCHEMA_VERSION = 1


class PaperpileCacheError(RuntimeError):
    pass


def connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    _create_schema(connection)
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS paperpile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS paperpile_snapshots (
          snapshot_id INTEGER PRIMARY KEY, observed_at TEXT NOT NULL, digest TEXT NOT NULL,
          status TEXT NOT NULL, item_count INTEGER NOT NULL DEFAULT 0, summary_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS paperpile_works (
          paperpile_work_id TEXT PRIMARY KEY, revision_digest TEXT NOT NULL, payload_json TEXT NOT NULL,
          active INTEGER NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paperpile_aliases (
          kind TEXT NOT NULL, value TEXT NOT NULL, paperpile_work_id TEXT NOT NULL,
          first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
          PRIMARY KEY(kind, value), FOREIGN KEY(paperpile_work_id) REFERENCES paperpile_works(paperpile_work_id)
        );
        CREATE TABLE IF NOT EXISTS paperpile_attachments (
          paperpile_work_id TEXT NOT NULL, relative_path TEXT NOT NULL, sha256 TEXT NOT NULL,
          size INTEGER NOT NULL, kind TEXT NOT NULL, provenance TEXT NOT NULL,
          PRIMARY KEY(paperpile_work_id, relative_path),
          FOREIGN KEY(paperpile_work_id) REFERENCES paperpile_works(paperpile_work_id)
        );
        CREATE TABLE IF NOT EXISTS paperpile_changes (
          change_time TEXT PRIMARY KEY, operation TEXT NOT NULL, paperpile_work_id TEXT NOT NULL,
          revision_digest TEXT NOT NULL, payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS paperpile_quarantine (
          id INTEGER PRIMARY KEY, snapshot_id INTEGER, reason TEXT NOT NULL, candidate_ids TEXT NOT NULL DEFAULT '[]'
        );
        """
    )
    current = connection.execute(
        "SELECT value FROM paperpile_meta WHERE key='schema_version'"
    ).fetchone()
    if current is None:
        connection.execute(
            "INSERT INTO paperpile_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.commit()
    elif int(current[0]) > SCHEMA_VERSION:
        raise PaperpileCacheError("Paperpile cache schema is newer than this reader")


def last_snapshot(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM paperpile_snapshots WHERE status='complete' ORDER BY snapshot_id DESC LIMIT 1"
    ).fetchone()


def next_change_time(connection: sqlite3.Connection, observed_at: datetime) -> datetime:
    observed_at = observed_at.astimezone(UTC)
    row = connection.execute(
        "SELECT change_time FROM paperpile_changes ORDER BY change_time DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return observed_at
    prior = datetime.fromisoformat(row[0])
    return max(observed_at, prior + timedelta(microseconds=1))


def changes_after(connection: sqlite3.Connection, since: datetime | None) -> list[sqlite3.Row]:
    if since is None:
        return list(connection.execute("SELECT * FROM paperpile_changes ORDER BY change_time"))
    return list(
        connection.execute(
            "SELECT * FROM paperpile_changes WHERE change_time > ? ORDER BY change_time",
            (since.isoformat(),),
        )
    )


def cache_age(connection: sqlite3.Connection, now: datetime) -> timedelta | None:
    row = last_snapshot(connection)
    return now.astimezone(UTC) - datetime.fromisoformat(row["observed_at"]) if row else None


def payload(row: sqlite3.Row) -> dict:
    return json.loads(row["payload_json"]) if row["payload_json"] else {}
