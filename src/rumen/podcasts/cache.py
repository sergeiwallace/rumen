"""SQLite storage owned by the Spotify podcast sync command.

This deliberately is not part of Rumen's Alembic schema: the reader adapter only opens a
completed local cache and never mutates it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1


class CacheError(RuntimeError):
    """The private connector cache cannot safely be read or updated."""


class CacheSchemaError(CacheError):
    """The cache is missing, corrupt, or from an unsupported implementation."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS podcast_cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spotify_podcast_imports (
  import_id INTEGER PRIMARY KEY, file_sha256 TEXT NOT NULL UNIQUE, basename TEXT NOT NULL,
  imported_at TEXT NOT NULL, podcast_records INTEGER NOT NULL, non_podcast_records INTEGER NOT NULL,
  min_played_at TEXT, max_played_at TEXT
);
CREATE TABLE IF NOT EXISTS spotify_podcast_listening_events (
  event_hash TEXT PRIMARY KEY, spotify_episode_uri TEXT NOT NULL, episode_name TEXT NOT NULL,
  show_name TEXT NOT NULL, played_at TEXT NOT NULL, milliseconds_played INTEGER NOT NULL,
  platform TEXT, offline INTEGER, raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spotify_podcast_import_events (
  import_id INTEGER NOT NULL REFERENCES spotify_podcast_imports(import_id) ON DELETE CASCADE,
  event_hash TEXT NOT NULL REFERENCES spotify_podcast_listening_events(event_hash) ON DELETE CASCADE,
  record_index INTEGER NOT NULL, PRIMARY KEY (import_id, event_hash, record_index)
);
CREATE TABLE IF NOT EXISTS podcast_feeds (
  feed_url TEXT PRIMARY KEY, show_key TEXT NOT NULL, etag TEXT, last_modified TEXT,
  fetched_at TEXT, content_hash TEXT, body TEXT, last_error TEXT
);
CREATE TABLE IF NOT EXISTS podcast_episodes (
  episode_key TEXT PRIMARY KEY, feed_url TEXT NOT NULL REFERENCES podcast_feeds(feed_url) ON DELETE CASCADE,
  guid TEXT, link TEXT, enclosure_url TEXT, enclosure_mime TEXT, title TEXT NOT NULL,
  show_title TEXT NOT NULL, published_at TEXT NOT NULL, duration_seconds INTEGER,
  match_status TEXT NOT NULL, match_evidence TEXT NOT NULL, transcript_state TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS podcast_episodes_ready ON podcast_episodes(transcript_state, published_at, episode_key);
CREATE TABLE IF NOT EXISTS podcast_episode_aliases (
  spotify_episode_uri TEXT NOT NULL, episode_key TEXT NOT NULL REFERENCES podcast_episodes(episode_key) ON DELETE CASCADE,
  method TEXT NOT NULL, confidence TEXT NOT NULL, evidence TEXT NOT NULL,
  PRIMARY KEY (spotify_episode_uri, episode_key)
);
CREATE TABLE IF NOT EXISTS podcast_transcript_segments (
  episode_key TEXT NOT NULL REFERENCES podcast_episodes(episode_key) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, start_ms INTEGER, end_ms INTEGER, speaker TEXT, text TEXT NOT NULL,
  PRIMARY KEY (episode_key, ordinal)
);
CREATE TABLE IF NOT EXISTS podcast_transcripts (
  episode_key TEXT PRIMARY KEY REFERENCES podcast_episodes(episode_key) ON DELETE CASCADE,
  source_class TEXT NOT NULL, source_url TEXT NOT NULL, mime_type TEXT NOT NULL, language TEXT,
  retrieved_at TEXT NOT NULL, content_hash TEXT NOT NULL, plain_text TEXT NOT NULL,
  generator_identifier TEXT
);
CREATE TABLE IF NOT EXISTS podcast_quarantines (
  spotify_episode_uri TEXT PRIMARY KEY, show_name TEXT NOT NULL, episode_name TEXT NOT NULL,
  reason TEXT NOT NULL, evidence TEXT NOT NULL, updated_at TEXT NOT NULL
);
"""


def _schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM podcast_cache_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        raise CacheSchemaError("spotify podcast cache has no schema version")
    try:
        return int(row["value"])
    except ValueError as exc:
        raise CacheSchemaError("spotify podcast cache has an invalid schema version") from exc


def initialize(path: str | Path) -> sqlite3.Connection:
    """Open (and create) a writer cache, rejecting a future schema before mutation."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    _configure(conn)
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='podcast_cache_meta'"
    ).fetchone()
    if exists:
        version = _schema_version(conn)
        if version > SCHEMA_VERSION:
            conn.close()
            raise CacheSchemaError(
                f"spotify podcast cache schema {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version != SCHEMA_VERSION:
            conn.close()
            raise CacheSchemaError(f"unsupported spotify podcast cache schema {version}")
        return conn
    with conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO podcast_cache_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
    return conn


def open_reader(path: str | Path) -> sqlite3.Connection:
    """Open an existing cache without creating it, validating the public reader contract."""
    cache_path = Path(path)
    if not cache_path.is_file():
        raise CacheSchemaError(f"spotify podcast cache does not exist: {cache_path}")
    try:
        conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
        _configure(conn)
        version = _schema_version(conn)
    except sqlite3.Error as exc:
        raise CacheSchemaError(f"spotify podcast cache is unreadable: {cache_path}") from exc
    if version != SCHEMA_VERSION:
        conn.close()
        raise CacheSchemaError(
            f"spotify podcast cache schema {version} is incompatible with supported {SCHEMA_VERSION}"
        )
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Use an explicit transaction so each import and transcript commit is atomic."""
    try:
        conn.execute("BEGIN")
        yield
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def ready_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT e.*, t.source_class, t.source_url, t.mime_type, t.language, t.retrieved_at,
               t.content_hash, t.plain_text, t.generator_identifier,
               COUNT(s.ordinal) AS segment_count,
               COUNT(a.spotify_episode_uri) AS play_count,
               MIN(v.played_at) AS first_played_at, MAX(v.played_at) AS last_played_at,
               GROUP_CONCAT(a.spotify_episode_uri) AS spotify_uris
        FROM podcast_episodes e
        JOIN podcast_transcripts t ON t.episode_key = e.episode_key
        LEFT JOIN podcast_transcript_segments s ON s.episode_key = e.episode_key
        LEFT JOIN podcast_episode_aliases a ON a.episode_key = e.episode_key
        LEFT JOIN spotify_podcast_listening_events v ON v.spotify_episode_uri = a.spotify_episode_uri
        WHERE e.transcript_state = 'ready'
        GROUP BY e.episode_key
        ORDER BY e.published_at, e.episode_key
        """
    ).fetchall()


def record_feed(
    conn: sqlite3.Connection,
    *,
    feed_url: str,
    show_key: str,
    etag: str | None = None,
    last_modified: str | None = None,
    body: str | None = None,
    content_hash: str | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO podcast_feeds(feed_url, show_key, etag, last_modified, fetched_at, content_hash, body, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(feed_url) DO UPDATE SET show_key=excluded.show_key, etag=COALESCE(excluded.etag, podcast_feeds.etag),
          last_modified=COALESCE(excluded.last_modified, podcast_feeds.last_modified),
          fetched_at=CASE WHEN excluded.body IS NULL THEN podcast_feeds.fetched_at ELSE excluded.fetched_at END,
          content_hash=COALESCE(excluded.content_hash, podcast_feeds.content_hash),
          body=COALESCE(excluded.body, podcast_feeds.body), last_error=excluded.last_error
        """,
        (feed_url, show_key, etag, last_modified, utc_now(), content_hash, body, error),
    )


def store_ready_transcript(
    conn: sqlite3.Connection,
    episode_key: str,
    *,
    source_class: str,
    source_url: str,
    mime_type: str,
    language: str | None,
    plain_text: str,
    segments: Iterable[object],
    generator_identifier: str | None = None,
) -> None:
    """Replace transcript and segments then flip readiness in one caller transaction."""
    if not plain_text.strip():
        raise CacheError("cannot mark an episode ready without transcript text")
    conn.execute("DELETE FROM podcast_transcripts WHERE episode_key = ?", (episode_key,))
    conn.execute(
        """INSERT INTO podcast_transcripts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            episode_key,
            source_class,
            source_url,
            mime_type,
            language,
            utc_now(),
            sha256_text(plain_text),
            plain_text,
            generator_identifier,
        ),
    )
    conn.execute("DELETE FROM podcast_transcript_segments WHERE episode_key = ?", (episode_key,))
    for ordinal, segment in enumerate(segments):
        conn.execute(
            "INSERT INTO podcast_transcript_segments VALUES (?, ?, ?, ?, ?, ?)",
            (
                episode_key,
                ordinal,
                getattr(segment, "start_ms"),
                getattr(segment, "end_ms"),
                getattr(segment, "speaker"),
                getattr(segment, "text"),
            ),
        )
    conn.execute(
        "UPDATE podcast_episodes SET transcript_state = 'ready', updated_at = ? WHERE episode_key = ?",
        (utc_now(), episode_key),
    )


def purge_import(conn: sqlite3.Connection, digest: str) -> list[str]:
    """Delete one source import and its now-unreferenced downstream cache data."""
    row = conn.execute(
        "SELECT import_id FROM spotify_podcast_imports WHERE file_sha256 = ?", (digest,)
    ).fetchone()
    if row is None:
        raise CacheError(f"no spotify podcast import with digest {digest}")
    before = {r["episode_key"] for r in conn.execute("SELECT episode_key FROM podcast_episodes")}
    conn.execute("DELETE FROM spotify_podcast_imports WHERE import_id = ?", (row["import_id"],))
    conn.execute(
        "DELETE FROM spotify_podcast_listening_events WHERE NOT EXISTS "
        "(SELECT 1 FROM spotify_podcast_import_events i WHERE i.event_hash = spotify_podcast_listening_events.event_hash)"
    )
    conn.execute(
        "DELETE FROM podcast_episodes WHERE NOT EXISTS (SELECT 1 FROM podcast_episode_aliases a "
        "JOIN spotify_podcast_listening_events e ON e.spotify_episode_uri = a.spotify_episode_uri "
        "WHERE a.episode_key = podcast_episodes.episode_key)"
    )
    after = {r["episode_key"] for r in conn.execute("SELECT episode_key FROM podcast_episodes")}
    return sorted(before - after)
