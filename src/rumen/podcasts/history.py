"""Deterministic Spotify Extended Streaming History import, with no Spotify requests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rumen.podcasts.cache import canonical_json, transaction, utc_now


class HistoryImportError(ValueError):
    """A selected source file is not a valid podcast-history export."""


@dataclass(frozen=True)
class ImportResult:
    path: Path
    file_digest: str
    imported: int = 0
    duplicates: int = 0
    non_podcast: int = 0
    already_imported: bool = False


def _parse_timestamp(value: object, *, path: Path, index: int) -> str:
    if not isinstance(value, str):
        raise HistoryImportError(f"{path}: record {index}: ts must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoryImportError(f"{path}: record {index}: invalid ts") from exc
    if parsed.tzinfo is None:
        raise HistoryImportError(f"{path}: record {index}: ts must include timezone")
    return parsed.astimezone(UTC).isoformat()


def _podcast_event(raw: object, *, path: Path, index: int) -> dict | None:
    if not isinstance(raw, dict):
        raise HistoryImportError(f"{path}: record {index}: record must be an object")
    uri = raw.get("spotify_episode_uri")
    if uri is None:
        return None
    if not isinstance(uri, str) or not uri.startswith("spotify:episode:"):
        raise HistoryImportError(f"{path}: record {index}: invalid spotify_episode_uri")
    episode = raw.get("episode_name")
    show = raw.get("episode_show_name")
    milliseconds = raw.get("ms_played")
    if not isinstance(episode, str) or not episode.strip():
        raise HistoryImportError(f"{path}: record {index}: missing episode_name")
    if not isinstance(show, str) or not show.strip():
        raise HistoryImportError(f"{path}: record {index}: missing episode_show_name")
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, int) or milliseconds < 0:
        raise HistoryImportError(f"{path}: record {index}: invalid ms_played")
    played_at = _parse_timestamp(raw.get("ts"), path=path, index=index)
    identity = {
        "spotify_episode_uri": uri,
        "episode_name": episode,
        "episode_show_name": show,
        "ts": played_at,
        "ms_played": milliseconds,
        "platform": raw.get("platform"),
        "offline": raw.get("offline"),
    }
    return {
        "event_hash": hashlib.sha256(canonical_json(identity).encode()).hexdigest(),
        "raw": raw,
        **identity,
    }


def discover_json_inputs(inputs: list[str | Path]) -> list[Path]:
    """Resolve explicit files/directories to a stable JSON-file order."""
    paths: list[Path] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            paths.extend(
                sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file())
            )
        else:
            paths.append(path)
    return sorted(paths, key=lambda candidate: str(candidate))


def import_file(conn: sqlite3.Connection, path: str | Path) -> ImportResult:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        payload = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryImportError(f"{source}: malformed JSON") from exc
    if not isinstance(payload, list):
        raise HistoryImportError(f"{source}: export payload must be an array")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if conn.execute(
        "SELECT 1 FROM spotify_podcast_imports WHERE file_sha256 = ?", (digest,)
    ).fetchone():
        return ImportResult(source, digest, already_imported=True)
    events: list[tuple[int, dict]] = []
    non_podcast = 0
    for index, record in enumerate(payload):
        event = _podcast_event(record, path=source, index=index)
        if event is None:
            non_podcast += 1
        else:
            events.append((index, event))
    # Parsing precedes the transaction so malformed file N cannot partially survive.
    with transaction(conn):
        cursor = conn.execute(
            """INSERT INTO spotify_podcast_imports(file_sha256, basename, imported_at, podcast_records,
            non_podcast_records, min_played_at, max_played_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                digest,
                source.name,
                utc_now(),
                len(events),
                non_podcast,
                min((event["ts"] for _, event in events), default=None),
                max((event["ts"] for _, event in events), default=None),
            ),
        )
        import_id = cursor.lastrowid
        duplicates = 0
        for index, event in events:
            inserted = conn.execute(
                """INSERT OR IGNORE INTO spotify_podcast_listening_events(event_hash, spotify_episode_uri,
                episode_name, show_name, played_at, milliseconds_played, platform, offline, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["event_hash"],
                    event["spotify_episode_uri"],
                    event["episode_name"],
                    event["episode_show_name"],
                    event["ts"],
                    event["ms_played"],
                    event["platform"],
                    int(bool(event["offline"])) if event["offline"] is not None else None,
                    canonical_json(event["raw"]),
                ),
            )
            duplicates += not bool(inserted.rowcount)
            conn.execute(
                "INSERT INTO spotify_podcast_import_events(import_id, event_hash, record_index) VALUES (?, ?, ?)",
                (import_id, event["event_hash"], index),
            )
    return ImportResult(source, digest, len(events) - duplicates, duplicates, non_podcast)


def import_inputs(conn: sqlite3.Connection, inputs: list[str | Path]) -> list[ImportResult]:
    """Import each file independently: one bad export never rolls back a prior valid one."""
    results: list[ImportResult] = []
    first_error: HistoryImportError | None = None
    for path in discover_json_inputs(inputs):
        try:
            results.append(import_file(conn, path))
        except HistoryImportError as exc:
            first_error = first_error or exc
    if first_error is not None:
        raise first_error
    return results
