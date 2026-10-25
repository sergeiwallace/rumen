"""Import private Spotify history and synchronize mapped publisher podcast text.

The manifest is intentionally separate from runtime configuration.  It contains no
Spotify credentials: ``exports`` and ``feeds`` are local/operator-owned inputs, while
network requests target only configured publisher URLs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from rumen.podcasts.asr import AsrConfig, AsrError, transcribe
from rumen.podcasts.cache import (
    initialize,
    record_feed,
    store_ready_transcript,
    transaction,
    utc_now,
)
from rumen.podcasts.history import import_inputs
from rumen.podcasts.http import HttpError, PublisherHttpClient, redact_url
from rumen.podcasts.rss import episode_key, normalize_name, parse_feed, resolve_episode
from rumen.podcasts.transcripts import (
    TranscriptError,
    normalize_transcript,
    select_candidate,
    validate_candidate,
)


class SyncError(ValueError):
    """A private sync manifest is malformed or asks for a forbidden content path."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError("sync manifest is unreadable JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cache_path"), str):
        raise SyncError("sync manifest requires cache_path")
    if not isinstance(data.get("feeds", []), list):
        raise SyncError("sync manifest feeds must be a list")
    return data


def _feed_specs(manifest_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for feed in manifest_data.get("feeds", []):
        if (
            not isinstance(feed, dict)
            or not isinstance(feed.get("show"), str)
            or not isinstance(feed.get("url"), str)
        ):
            raise SyncError("every feed requires show and url strings")
        key = normalize_name(feed["show"])
        if not key or key in result:
            raise SyncError("feed show mappings must have unique non-empty normalized names")
        result[key] = feed
    return result


def _limits(manifest_data: dict[str, Any]) -> dict[str, int | float]:
    limits = manifest_data.get("limits", {})
    if not isinstance(limits, dict):
        raise SyncError("limits must be an object")
    defaults: dict[str, int | float] = {
        "feed_byte_ceiling": 5 * 1024 * 1024,
        "transcript_byte_ceiling": 20 * 1024 * 1024,
        "enclosure_byte_ceiling": 200 * 1024 * 1024,
        "timeout_seconds": 30.0,
        "max_attempts": 3,
        "request_interval_seconds": 1.0,
        "max_retry_after_seconds": 300.0,
    }
    result = {key: limits.get(key, default) for key, default in defaults.items()}
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in result.values()
    ):
        raise SyncError("all sync limits must be positive numbers")
    return result


def _summary() -> Counter:
    return Counter(
        {
            key: 0
            for key in (
                "imported",
                "duplicate",
                "resolved",
                "quarantined",
                "publisher_transcript",
                "asr",
                "skipped_rights",
                "retry",
                "failure",
            )
        }
    )


def _quarantine(conn: sqlite3.Connection, event: sqlite3.Row, reason: str, evidence: str) -> None:
    conn.execute(
        """INSERT INTO podcast_quarantines VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(spotify_episode_uri) DO UPDATE SET reason=excluded.reason, evidence=excluded.evidence, updated_at=excluded.updated_at""",
        (
            event["spotify_episode_uri"],
            event["show_name"],
            event["episode_name"],
            reason,
            evidence,
            utc_now(),
        ),
    )


def _store_resolution(
    conn: sqlite3.Connection, event: sqlite3.Row, feed_url: str, episode, method: str, evidence: str
) -> str:
    key = episode_key(feed_url, episode)
    assert key is not None
    conn.execute(
        """INSERT INTO podcast_episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(episode_key) DO UPDATE SET title=excluded.title, link=excluded.link,
          enclosure_url=excluded.enclosure_url, enclosure_mime=excluded.enclosure_mime,
          published_at=excluded.published_at, match_status='resolved', match_evidence=excluded.match_evidence,
          transcript_state=CASE WHEN podcast_episodes.transcript_state = 'ready' THEN 'ready' ELSE excluded.transcript_state END,
          updated_at=excluded.updated_at""",
        (
            key,
            feed_url,
            episode.guid,
            episode.link,
            episode.enclosure_url,
            episode.enclosure_mime,
            episode.title,
            episode.show_title,
            episode.published_at,
            None,
            "resolved",
            evidence,
            "pending",
            utc_now(),
        ),
    )
    conn.execute(
        """INSERT INTO podcast_episode_aliases VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(spotify_episode_uri, episode_key) DO UPDATE SET method=excluded.method, confidence=excluded.confidence, evidence=excluded.evidence""",
        (event["spotify_episode_uri"], key, method, "high", evidence),
    )
    return key


def _fetch_feed(
    conn: sqlite3.Connection,
    client: PublisherHttpClient,
    feed: dict[str, Any],
    limits: dict[str, int | float],
):
    existing = conn.execute(
        "SELECT * FROM podcast_feeds WHERE feed_url = ?", (feed["url"],)
    ).fetchone()
    headers: dict[str, str] = {}
    if existing and existing["etag"]:
        headers["If-None-Match"] = existing["etag"]
    if existing and existing["last_modified"]:
        headers["If-Modified-Since"] = existing["last_modified"]
    try:
        response = client.get(
            feed["url"], headers=headers, byte_ceiling=int(limits["feed_byte_ceiling"])
        )
        if response.status == 304:
            if not existing or not existing["body"]:
                raise SyncError("publisher returned not-modified without a cached feed")
            return parse_feed(existing["body"])
        parsed = parse_feed(response.body)
        with transaction(conn):
            record_feed(
                conn,
                feed_url=feed["url"],
                show_key=normalize_name(feed["show"]),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                body=response.body.decode("utf-8"),
                content_hash=hashlib.sha256(response.body).hexdigest(),
            )
        return parsed
    except (HttpError, UnicodeDecodeError, ValueError, SyncError) as exc:
        with transaction(conn):
            if existing:
                record_feed(
                    conn,
                    feed_url=feed["url"],
                    show_key=normalize_name(feed["show"]),
                    error=str(exc),
                )
        raise SyncError(f"feed refresh failed: {redact_url(feed['url'])}") from exc


def _asr_config(feed: dict[str, Any]) -> AsrConfig:
    raw = feed.get("asr")
    if not isinstance(raw, dict):
        raise SyncError("allow_asr feed requires an asr object")
    config = AsrConfig(
        argv=raw.get("argv", []),
        tool_model_identifier=raw.get("tool_model_identifier", ""),
        timeout_seconds=raw.get("timeout_seconds", 900),
        output_mime_type=raw.get("output_mime_type", "text/vtt"),
        output_byte_ceiling=raw.get("output_byte_ceiling", 20 * 1024 * 1024),
    )
    config.validate()
    return config


def synchronize(
    manifest: dict[str, Any],
    *,
    client: PublisherHttpClient | None = None,
    import_only: bool = False,
) -> dict[str, int | list[str]]:
    """Run all local work; callers inject ``client`` for deterministic fixture tests."""
    limits = _limits(manifest)
    client = client or PublisherHttpClient(
        timeout_seconds=float(limits["timeout_seconds"]),
        max_attempts=int(limits["max_attempts"]),
        request_interval_seconds=float(limits["request_interval_seconds"]),
        max_retry_after_seconds=float(limits["max_retry_after_seconds"]),
    )
    summary = _summary()
    conn = initialize(manifest["cache_path"])
    try:
        for result in import_inputs(conn, manifest.get("exports", [])):
            summary["imported"] += result.imported
            summary["duplicate"] += result.duplicates + int(result.already_imported)
        if import_only:
            return dict(summary)
        feeds = _feed_specs(manifest)
        aliases = manifest.get("episode_aliases", {})
        if not isinstance(aliases, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in aliases.items()
        ):
            raise SyncError("episode_aliases must map Spotify URI strings to GUID strings")
        events = conn.execute(
            "SELECT * FROM spotify_podcast_listening_events ORDER BY played_at, event_hash"
        ).fetchall()
        parsed_feeds: dict[str, list] = {}
        for show_key in sorted(
            {normalize_name(event["show_name"]) for event in events} & set(feeds)
        ):
            try:
                parsed_feeds[show_key] = _fetch_feed(conn, client, feeds[show_key], limits)
            except SyncError:
                summary["failure"] += 1
        for event in events:
            show_key = normalize_name(event["show_name"])
            feed = feeds.get(show_key)
            if feed is None:
                with transaction(conn):
                    _quarantine(conn, event, "no-explicit-feed-mapping", "")
                summary["quarantined"] += 1
                continue
            episode_list = parsed_feeds.get(show_key)
            if episode_list is None:
                continue
            resolution = resolve_episode(
                event["spotify_episode_uri"],
                event["episode_name"],
                feed["url"],
                episode_list,
                aliases,
            )
            if resolution.episode is None:
                with transaction(conn):
                    _quarantine(conn, event, resolution.reason or "unresolved", resolution.evidence)
                summary["quarantined"] += 1
                continue
            with transaction(conn):
                key = _store_resolution(
                    conn,
                    event,
                    feed["url"],
                    resolution.episode,
                    resolution.method or "",
                    resolution.evidence,
                )
            summary["resolved"] += 1
            candidate = select_candidate(
                resolution.episode.transcripts, manifest.get("preferred_language")
            )
            if candidate is not None:
                try:
                    validate_candidate(candidate)
                    response = client.get(
                        candidate.url, byte_ceiling=int(limits["transcript_byte_ceiling"])
                    )
                    normalized = normalize_transcript(response.body, candidate.mime_type)
                    with transaction(conn):
                        store_ready_transcript(
                            conn,
                            key,
                            source_class="publisher",
                            source_url=candidate.url,
                            mime_type=candidate.mime_type,
                            language=candidate.language,
                            plain_text=normalized.plain_text,
                            segments=normalized.segments,
                        )
                    summary["publisher_transcript"] += 1
                    continue
                except (HttpError, TranscriptError, ValueError):
                    summary["failure"] += 1
            if not feed.get("allow_asr", False):
                summary["skipped_rights"] += 1
                continue
            try:
                if (
                    not resolution.episode.enclosure_url
                    or not resolution.episode.enclosure_mime
                    or not resolution.episode.enclosure_mime.casefold().startswith("audio/")
                ):
                    raise SyncError("episode enclosure is missing or disallowed")
                response = client.get(
                    resolution.episode.enclosure_url,
                    byte_ceiling=int(limits["enclosure_byte_ceiling"]),
                )
                normalized = transcribe(response.body, _asr_config(feed))
                with transaction(conn):
                    store_ready_transcript(
                        conn,
                        key,
                        source_class="generated-local-asr",
                        source_url=resolution.episode.enclosure_url,
                        mime_type="text/vtt",
                        language=None,
                        plain_text=normalized.plain_text,
                        segments=normalized.segments,
                        generator_identifier=_asr_config(feed).tool_model_identifier,
                    )
                summary["asr"] += 1
            except (HttpError, AsrError, SyncError, ValueError):
                summary["failure"] += 1
        summary["retry"] = client.retry_count
        return dict(summary)
    finally:
        conn.close()
