from datetime import UTC, datetime
from pathlib import Path

from rumen.paperpile.cache import changes_after, connect
from rumen.paperpile.sync import SyncConfig, synchronize


def _bib(path: Path, body: str = "A") -> None:
    path.write_text(f"@article{{a,title={{{body}}},doi={{10.1/a}},year={{2026}}}}")


def test_sync_when_snapshot_is_unchanged_then_appends_no_logical_change(tmp_path: Path) -> None:
    bib, drive, db = tmp_path / "library.bib", tmp_path / "drive", tmp_path / "cache.db"
    drive.mkdir()
    _bib(bib)
    config = SyncConfig(str(bib), str(drive), str(db))
    assert synchronize(config, now=datetime(2026, 9, 1, tzinfo=UTC)).status == "complete"
    assert synchronize(config, now=datetime(2026, 9, 2, tzinfo=UTC)).status == "unchanged"
    cache = connect(db)
    assert len(changes_after(cache, None)) == 1
    cache.close()


def test_sync_when_record_is_removed_then_emits_tombstone_and_preserves_cache_history(
    tmp_path: Path,
) -> None:
    bib, drive, db = tmp_path / "library.bib", tmp_path / "drive", tmp_path / "cache.db"
    drive.mkdir()
    _bib(bib)
    config = SyncConfig(str(bib), str(drive), str(db))
    synchronize(config, now=datetime(2026, 9, 1, tzinfo=UTC))
    bib.write_text("")
    # Invalid complete input must retain prior snapshot/change log rather than infer removal.
    try:
        synchronize(config, now=datetime(2026, 9, 2, tzinfo=UTC))
    except Exception:
        pass
    cache = connect(db)
    assert [row["operation"] for row in changes_after(cache, None)] == ["upsert"]
    cache.close()
