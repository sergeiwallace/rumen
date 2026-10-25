"""Chrome Reading List snapshot contract shared by the host and later ingestion work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SNAPSHOT_SCHEMA_VERSION = 1
MINIMUM_CHROME_VERSION = 120
STABLE_EXTENSION_ID = "kcmhpnajmplpnlajfagepopcglodhgla"
EXTENSION_ORIGIN = f"chrome-extension://{STABLE_EXTENSION_ID}/"

_SNAPSHOT_KEYS = frozenset({"schemaVersion", "snapshotId", "items"})
_ITEM_KEYS = frozenset({"url", "title", "hasBeenRead", "creationTime", "lastUpdateTime"})


class SnapshotValidationError(ValueError):
    """A Reading List snapshot does not conform to the versioned contract."""


@dataclass(frozen=True, slots=True)
class ChromeReadingListSourceConfig:
    """Validated Batch-1 source settings, pending T-04 ingestion implementation."""

    profile: str
    inbox_path: str
    minimum_chrome_version: int


def validate_snapshot(snapshot: object) -> dict[str, Any]:
    """Return a validated snapshot without transforming its Chrome-provided values."""
    if not isinstance(snapshot, dict):
        raise SnapshotValidationError("snapshot must be an object")
    _reject_unknown_keys(snapshot, _SNAPSHOT_KEYS, "snapshot")
    if snapshot.get("schemaVersion") != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotValidationError(f"schemaVersion must be {SNAPSHOT_SCHEMA_VERSION}")
    snapshot_id = snapshot.get("snapshotId")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise SnapshotValidationError("snapshotId must be a non-empty string")
    items = snapshot.get("items")
    if not isinstance(items, list):
        raise SnapshotValidationError("items must be a list")
    for index, item in enumerate(items):
        _validate_item(item, index)
    return snapshot


def validate_native_message(message: object) -> dict[str, Any]:
    """Validate the native-messaging envelope and return its snapshot."""
    if not isinstance(message, dict):
        raise SnapshotValidationError("native message must be an object")
    _reject_unknown_keys(message, frozenset({"origin", "snapshot"}), "native message")
    if message.get("origin") != EXTENSION_ORIGIN:
        raise SnapshotValidationError("native message origin is not the registered extension")
    return validate_snapshot(message.get("snapshot"))


def _reject_unknown_keys(value: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise SnapshotValidationError(f"unknown {label} keys: {', '.join(sorted(unknown))}")


def _validate_item(item: object, index: int) -> None:
    if not isinstance(item, dict):
        raise SnapshotValidationError(f"items[{index}] must be an object")
    _reject_unknown_keys(item, _ITEM_KEYS, f"items[{index}]")
    url = item.get("url")
    if not isinstance(url, str) or not url:
        raise SnapshotValidationError(f"items[{index}].url must be a non-empty string")
    if not isinstance(item.get("title"), str):
        raise SnapshotValidationError(f"items[{index}].title must be a string")
    if not isinstance(item.get("hasBeenRead"), bool):
        raise SnapshotValidationError(f"items[{index}].hasBeenRead must be a boolean")
    for key in ("creationTime", "lastUpdateTime"):
        timestamp = item.get(key)
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise SnapshotValidationError(f"items[{index}].{key} must be a millisecond integer")
