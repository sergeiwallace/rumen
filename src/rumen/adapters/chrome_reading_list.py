"""Chrome Reading List snapshot inbox adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rumen.adapters.base import RawItem
from rumen.chrome_reading_list import SnapshotValidationError, validate_snapshot

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class ChromeReadingListError(RuntimeError):
    """The locally delivered Chrome Reading List inbox cannot be consumed."""


class ChromeReadingListAdapter:
    """Read one native-host inbox using the Chrome Reading List's URL identity."""

    def __init__(self, inbox_path: str, profile: str) -> None:
        self.inbox_path = inbox_path
        self.profile = profile

    def source_name(self) -> str:
        return f"chrome_reading_list:{self.profile}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        snapshot = self._load_snapshot()
        items = [self._map_item(item) for item in snapshot["items"]]
        return [item for item in items if since is None or item.published_at > since]

    def _load_snapshot(self) -> dict:
        try:
            snapshot = json.loads(Path(self.inbox_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChromeReadingListError(
                f"cannot read Chrome Reading List inbox {self.inbox_path!r}"
            ) from exc
        try:
            return validate_snapshot(snapshot)
        except SnapshotValidationError as exc:
            raise ChromeReadingListError(
                "Chrome Reading List inbox has an invalid snapshot"
            ) from exc

    def _map_item(self, item: dict) -> RawItem:
        try:
            updated_at = _UNIX_EPOCH + timedelta(milliseconds=item["lastUpdateTime"])
        except (OverflowError, TypeError, ValueError) as exc:
            raise ChromeReadingListError(
                "Chrome Reading List inbox has an unrepresentable lastUpdateTime"
            ) from exc
        return RawItem(
            source=self.source_name(),
            source_id=item["url"],
            url=item["url"],
            title=item["title"],
            published_at=updated_at,
        )
