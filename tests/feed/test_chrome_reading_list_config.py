"""Batch-1 Chrome Reading List contract and configuration coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rumen.adapters.chrome_reading_list import ChromeReadingListAdapter
from rumen.chrome_reading_list import (
    MINIMUM_CHROME_VERSION,
    ChromeReadingListSourceConfig,
    SnapshotValidationError,
    validate_snapshot,
)
from rumen.config import load_config

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "chrome_reading_list"


def _load(tmp_path: Path, sources: list[dict]):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"classifier_escalation_alarm_threshold": 0.25, "sources": sources})
    )
    return load_config(config_path)


def test_config_when_reading_list_source_is_enabled_then_constructs_one_validated_source(
    tmp_path: Path,
) -> None:
    config = _load(
        tmp_path,
        [
            {
                "type": "chrome_reading_list",
                "enabled": True,
                "profile": "Profile 1",
                "inbox_path": "/operator/rumen/reading-list.json",
                "minimum_chrome_version": MINIMUM_CHROME_VERSION,
            }
        ],
    )

    [adapter] = config.adapters
    assert isinstance(adapter, ChromeReadingListAdapter)
    assert adapter.source_name() == "chrome_reading_list:Profile 1"
    assert config.chrome_reading_list_sources == [
        ChromeReadingListSourceConfig("Profile 1", "/operator/rumen/reading-list.json", 120)
    ]


def test_config_when_reading_list_source_is_disabled_then_constructs_no_source_and_keeps_adapters(
    tmp_path: Path,
) -> None:
    config = _load(
        tmp_path,
        [
            {"type": "github_releases", "repo": "owner/repo"},
            {"type": "chrome_reading_list", "enabled": False},
        ],
    )

    assert [adapter.source_name() for adapter in config.adapters] == ["github_releases:owner/repo"]
    assert config.chrome_reading_list_sources == []


@pytest.mark.parametrize(
    "source",
    [
        {"type": "chrome_reading_list", "enabled": True, "extra": "no"},
        {"type": "chrome_reading_list", "enabled": True, "profile": "Default"},
        {
            "type": "chrome_reading_list",
            "enabled": "true",
            "profile": "Default",
            "inbox_path": "/operator/inbox.json",
            "minimum_chrome_version": 120,
        },
        {
            "type": "chrome_reading_list",
            "enabled": True,
            "profile": "Default",
            "inbox_path": "/operator/inbox.json",
            "minimum_chrome_version": 119,
        },
    ],
)
def test_config_when_reading_list_source_is_invalid_then_rejects_before_ingestion(
    tmp_path: Path, source: dict
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _load(tmp_path, [source])


def test_contract_when_valid_snapshot_is_validated_then_preserves_chrome_values_without_profile_access() -> (
    None
):
    snapshot = json.loads((_FIXTURES / "valid_snapshot.json").read_text())

    assert validate_snapshot(snapshot) is snapshot
    assert snapshot["items"][0] == {
        "url": "https://example.com/path?full=query%20value#fragment",
        "title": "A title with café",
        "hasBeenRead": False,
        "creationTime": 1788404400123,
        "lastUpdateTime": 1788404400456,
    }


def test_contract_when_snapshot_has_unknown_or_malformed_field_then_rejects() -> None:
    snapshot = json.loads((_FIXTURES / "valid_snapshot.json").read_text())
    snapshot["items"][0]["hasBeenRead"] = 1

    with pytest.raises(SnapshotValidationError, match="hasBeenRead"):
        validate_snapshot(snapshot)
