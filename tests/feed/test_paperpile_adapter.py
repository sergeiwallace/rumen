import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rumen.adapters.paperpile import PaperpileAdapter, PaperpileAdapterError
from rumen.config import load_config
from rumen.paperpile.sync import SyncConfig, synchronize


def _cache(tmp_path: Path) -> Path:
    bib, drive, db = tmp_path / "library.bib", tmp_path / "drive", tmp_path / "cache.db"
    drive.mkdir()
    bib.write_text("@article{a,title={A},doi={10.1/a},year={2026}}")
    synchronize(SyncConfig(str(bib), str(drive), str(db)), now=datetime.now(UTC))
    return db


def test_adapter_when_committed_cache_changes_exist_then_maps_ordered_upsert(
    tmp_path: Path,
) -> None:
    adapter = PaperpileAdapter(
        "personal", str(_cache(tmp_path)), staleness_threshold=timedelta(hours=1)
    )
    change = adapter.fetch_changes(None)[0]
    assert change.operation == "upsert"
    assert change.item.source == "paperpile:personal"
    assert change.item.url == "https://doi.org/10.1/a"
    assert change.item.metadata["identifiers"] == {"doi": "10.1/a"}


def test_adapter_when_cache_is_missing_or_stale_then_raises_source_specific_error(
    tmp_path: Path,
) -> None:
    missing = PaperpileAdapter(
        "personal", str(tmp_path / "missing.db"), staleness_threshold=timedelta(hours=1)
    )
    with pytest.raises(PaperpileAdapterError, match="missing"):
        missing.fetch_changes(None)

    db = _cache(tmp_path)
    stale = PaperpileAdapter("personal", str(db), staleness_threshold=timedelta(seconds=-1))
    with pytest.raises(PaperpileAdapterError, match="stale"):
        stale.fetch_changes(None)


def test_config_when_paperpile_spec_is_valid_or_invalid_then_validates_profile_and_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "classifier_escalation_alarm_threshold": 0.25,
                "sources": [
                    {
                        "type": "paperpile",
                        "profile": "personal",
                        "db_path": "/private/cache.db",
                        "staleness_hours": 48,
                    }
                ],
            }
        )
    )
    assert isinstance(load_config(path).adapters[0], PaperpileAdapter)
    path.write_text(
        json.dumps(
            {
                "classifier_escalation_alarm_threshold": 0.25,
                "sources": [{"type": "paperpile", "profile": "BAD", "db_path": "relative.db"}],
            }
        )
    )
    with pytest.raises(ValueError, match="profile"):
        load_config(path)
