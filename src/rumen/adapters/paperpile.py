"""Read committed local Paperpile cache changes; never scans files or calls a provider."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rumen.adapters.base import RawItem, SourceChange
from rumen.paperpile.cache import (
    PaperpileCacheError,
    cache_age,
    changes_after,
    connect,
    payload,
)


class PaperpileAdapterError(RuntimeError):
    pass


class PaperpileAdapter:
    def __init__(self, profile: str, db_path: str, *, staleness_threshold: timedelta) -> None:
        self.profile = profile
        self.db_path = db_path
        self.staleness_threshold = staleness_threshold

    def source_name(self) -> str:
        return f"paperpile:{self.profile}"

    def fetch_changes(self, since: datetime | None) -> list[SourceChange]:
        if not Path(self.db_path).is_file():
            raise PaperpileAdapterError("Paperpile cache is missing")
        try:
            cache = connect(self.db_path)
            try:
                age = cache_age(cache, datetime.now(UTC))
                if age is None:
                    raise PaperpileAdapterError("Paperpile cache has no complete snapshot")
                if age > self.staleness_threshold:
                    raise PaperpileAdapterError("Paperpile cache is stale")
                result = []
                for row in changes_after(cache, since):
                    occurred_at = datetime.fromisoformat(row["change_time"])
                    if row["operation"] == "tombstone":
                        result.append(
                            SourceChange(
                                "tombstone",
                                row["paperpile_work_id"],
                                occurred_at,
                                row["revision_digest"],
                            )
                        )
                        continue
                    data = payload(row)
                    identifiers = data.get("identifiers", {})
                    url = _url(data, identifiers, row["paperpile_work_id"], self.profile)
                    notes = data.get("notes")
                    content = data.get("abstract") or ""
                    if notes:
                        content = f"{content}\n\n[Exported notes]\n{notes}".strip()
                    result.append(
                        SourceChange(
                            "upsert",
                            row["paperpile_work_id"],
                            occurred_at,
                            row["revision_digest"],
                            RawItem(
                                source=self.source_name(),
                                source_id=row["paperpile_work_id"],
                                url=url,
                                title=data["title"],
                                published_at=datetime.fromisoformat(data["published_at"]),
                                content=content or None,
                                author="; ".join(data.get("authors", [])) or None,
                                metadata={
                                    "identifiers": identifiers,
                                    "citation_key": data.get("citation_key"),
                                    "folders": data.get("folders", []),
                                    "labels": data.get("labels", []),
                                    "attachments": data.get("attachments", []),
                                    "field_provenance": data.get("field_provenance", {}),
                                    "date_precision": data.get("date_precision"),
                                },
                            ),
                        )
                    )
                return result
            finally:
                cache.close()
        except (OSError, sqlite3.DatabaseError, PaperpileCacheError) as exc:
            raise PaperpileAdapterError("Paperpile cache is unavailable or incompatible") from exc


def _url(data: dict, identifiers: dict, work_id: str, profile: str) -> str:
    if identifiers.get("doi"):
        return f"https://doi.org/{identifiers['doi']}"
    if identifiers.get("arxiv"):
        return f"https://arxiv.org/abs/{identifiers['arxiv']}"
    if identifiers.get("pmid"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{identifiers['pmid']}/"
    if identifiers.get("isbn"):
        return f"urn:isbn:{identifiers['isbn']}"
    return f"paperpile://{profile}/{work_id}"
