"""Atomic local-artifact snapshot acquisition.  It performs no network request."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rumen.paperpile.attachments import AttachmentError, manifest, safe_attachment
from rumen.paperpile.cache import connect, last_snapshot, next_change_time
from rumen.paperpile.identity import aliases_for, resolve_exact
from rumen.paperpile.parsers import BibtexParseError, PaperRecord, parse_bibtex


class PaperpileSyncError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SyncConfig:
    bibtex_path: str
    drive_root: str
    db_path: str
    json_path: str | None = None
    minimum_scan_interval: timedelta = timedelta(0)
    max_metadata_bytes: int = 20_000_000
    max_attachment_count: int = 10_000
    max_attachment_bytes: int = 2_000_000_000


@dataclass(frozen=True, slots=True)
class SyncSummary:
    status: str
    digest_prefix: str | None
    records: int = 0
    upserts: int = 0
    tombstones: int = 0
    attachments: int = 0
    orphans: int = 0
    quarantine: int = 0
    next_eligible_at: datetime | None = None


def _stable_read(path: Path, max_bytes: int) -> tuple[str, str]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise PaperpileSyncError("metadata input is missing or exceeds configured limit")
    before = path.stat()
    value = path.read_text(encoding="utf-8")
    after = path.stat()
    digest = hashlib.sha256(value.encode()).hexdigest()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PaperpileSyncError("metadata input changed during scanning")
    return value, digest


def _json_enrichment(path: Path | None, max_bytes: int) -> tuple[dict[str, dict], str | None]:
    if path is None:
        return {}, None
    value, digest = _stable_read(path, max_bytes)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PaperpileSyncError("JSON enrichment is malformed") from exc
    records = decoded.get("records") if isinstance(decoded, dict) else decoded
    if not isinstance(records, list):
        raise PaperpileSyncError("JSON enrichment has unsupported top-level shape")
    result: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise PaperpileSyncError("JSON enrichment contains invalid record")
        key = record.get("citation_key") or record.get("citationKey")
        if isinstance(key, str):
            result[key] = record
    return result, digest


def _record_payload(record: PaperRecord, enrichment: dict | None, observed_at: datetime) -> dict:
    enrichment = enrichment or {}
    notes = enrichment.get("notes") if isinstance(enrichment.get("notes"), str) else None
    attachments = (
        enrichment.get("attachments") if isinstance(enrichment.get("attachments"), list) else []
    )
    return {
        "citation_key": record.citation_key,
        "title": record.title,
        "authors": record.authors,
        "published_at": record.published_at.isoformat(),
        "date_precision": record.date_precision,
        "identifiers": record.identifiers,
        "abstract": record.abstract,
        "url": record.url,
        "folders": enrichment.get("folders", []),
        "labels": enrichment.get("labels", []),
        "notes": notes,
        "attachment_locators": [x for x in attachments if isinstance(x, str)],
        "field_provenance": {
            "bibliography": "bibtex",
            "organization": "json" if enrichment else None,
        },
        "json_snapshot_at": observed_at.isoformat() if enrichment else None,
    }


def synchronize(
    config: SyncConfig, *, now: datetime | None = None, force: bool = False
) -> SyncSummary:
    """Promote one stable candidate atomically, retaining last-known-good state on failure."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    bibtex, drive, json_path = (
        Path(config.bibtex_path),
        Path(config.drive_root),
        (Path(config.json_path) if config.json_path else None),
    )
    cache = connect(config.db_path)
    try:
        previous = last_snapshot(cache)
        if previous and not force:
            eligible = (
                datetime.fromisoformat(previous["observed_at"]) + config.minimum_scan_interval
            )
            if now < eligible:
                return SyncSummary("skipped", None, next_eligible_at=eligible)
        if not drive.is_dir():
            raise PaperpileSyncError("Drive root is missing")
        before_manifest = manifest(drive)
        text, bib_digest = _stable_read(bibtex, config.max_metadata_bytes)
        json_records, json_digest = _json_enrichment(json_path, config.max_metadata_bytes)
        try:
            records = parse_bibtex(text, observed_at=now)
        except BibtexParseError as exc:
            raise PaperpileSyncError(str(exc)) from exc
        digest = hashlib.sha256(
            (bib_digest + (json_digest or "") + repr(before_manifest)).encode()
        ).hexdigest()
        if previous and previous["digest"] == digest:
            return SyncSummary("unchanged", digest[:12], records=len(records))
        known = {
            (row["kind"], row["value"]): row["paperpile_work_id"]
            for row in cache.execute("SELECT * FROM paperpile_aliases")
        }
        prepared: list[tuple[str, dict, set[tuple[str, str]], list]] = []
        quarantined = 0
        for record in records:
            payload = _record_payload(record, json_records.get(record.citation_key), now)
            locations = payload.pop("attachment_locators")
            attachments = []
            try:
                for location in locations:
                    attachment = safe_attachment(drive, location, provenance="json:attachments")
                    if attachment.size > config.max_attachment_bytes:
                        raise PaperpileSyncError("attachment exceeds configured limit")
                    attachments.append(attachment)
            except (AttachmentError, PaperpileSyncError) as exc:
                raise PaperpileSyncError("invalid exported attachment relationship") from exc
            aliases = aliases_for(record, [attachment.sha256 for attachment in attachments])
            resolution = resolve_exact(aliases, known)
            if resolution.quarantine_reason:
                quarantined += 1
                continue
            work_id = resolution.work_id or uuid.uuid4().hex
            known.update({alias: work_id for alias in aliases})
            payload["attachments"] = [asdict(attachment) for attachment in attachments]
            prepared.append((work_id, payload, aliases, attachments))
        if len(before_manifest) > config.max_attachment_count:
            raise PaperpileSyncError("attachment count exceeds configured limit")
        if manifest(drive) != before_manifest:
            raise PaperpileSyncError("Drive manifest changed during scanning")
        upserts, tombstones = _commit(cache, digest, now, prepared, quarantined)
        attachment_count = sum(len(item[3]) for item in prepared)
        active_paths = {a.relative_path for _, _, _, ats in prepared for a in ats}
        return SyncSummary(
            "complete",
            digest[:12],
            len(records),
            upserts=upserts,
            tombstones=tombstones,
            attachments=attachment_count,
            orphans=len(set(before_manifest) - active_paths),
            quarantine=quarantined,
        )
    finally:
        cache.close()


def _commit(
    cache: sqlite3.Connection, digest: str, observed_at: datetime, prepared: list, quarantined: int
) -> tuple[int, int]:
    current = {
        row["paperpile_work_id"]: row for row in cache.execute("SELECT * FROM paperpile_works")
    }
    seen, upserts, tombstones = set(), 0, 0
    change_time = next_change_time(cache, observed_at)
    with cache:
        cursor = cache.execute(
            "INSERT INTO paperpile_snapshots(observed_at,digest,status,item_count) VALUES(?,?,?,?)",
            (observed_at.isoformat(), digest, "complete", len(prepared)),
        )
        snapshot_id = cursor.lastrowid
        for work_id, payload, aliases, attachments in prepared:
            seen.add(work_id)
            revision = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            prior = current.get(work_id)
            cache.execute(
                "INSERT INTO paperpile_works VALUES(?,?,?,?,?) ON CONFLICT(paperpile_work_id) DO UPDATE SET revision_digest=excluded.revision_digest,payload_json=excluded.payload_json,active=1,updated_at=excluded.updated_at",
                (
                    work_id,
                    revision,
                    json.dumps(payload, sort_keys=True),
                    1,
                    observed_at.isoformat(),
                ),
            )
            for kind, value in aliases:
                cache.execute(
                    "INSERT INTO paperpile_aliases VALUES(?,?,?,?,?) ON CONFLICT(kind,value) DO UPDATE SET last_seen=excluded.last_seen",
                    (kind, value, work_id, observed_at.isoformat(), observed_at.isoformat()),
                )
            cache.execute("DELETE FROM paperpile_attachments WHERE paperpile_work_id=?", (work_id,))
            cache.executemany(
                "INSERT INTO paperpile_attachments VALUES(?,?,?,?,?,?)",
                [
                    (work_id, a.relative_path, a.sha256, a.size, a.kind, a.relationship_provenance)
                    for a in attachments
                ],
            )
            if prior is None or prior["revision_digest"] != revision:
                cache.execute(
                    "INSERT INTO paperpile_changes VALUES(?,?,?,?,?)",
                    (
                        change_time.isoformat(),
                        "upsert",
                        work_id,
                        revision,
                        json.dumps(payload, sort_keys=True),
                    ),
                )
                change_time += timedelta(microseconds=1)
                upserts += 1
        for work_id, prior in current.items():
            if work_id not in seen and prior["active"]:
                cache.execute(
                    "UPDATE paperpile_works SET active=0,updated_at=? WHERE paperpile_work_id=?",
                    (observed_at.isoformat(), work_id),
                )
                cache.execute(
                    "INSERT INTO paperpile_changes VALUES(?,?,?,?,NULL)",
                    (change_time.isoformat(), "tombstone", work_id, prior["revision_digest"]),
                )
                change_time += timedelta(microseconds=1)
                tombstones += 1
        for _ in range(quarantined):
            cache.execute(
                "INSERT INTO paperpile_quarantine(snapshot_id,reason) VALUES(?,?)",
                (snapshot_id, "conflicting exact aliases"),
            )
        cache.execute(
            "UPDATE paperpile_snapshots SET summary_json=? WHERE snapshot_id=?",
            (
                json.dumps(
                    {"upserts": upserts, "tombstones": tombstones, "quarantine": quarantined}
                ),
                snapshot_id,
            ),
        )
    return upserts, tombstones
