"""Offline cache/import regressions for the Spotify podcast source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rumen.podcasts.cache import CacheSchemaError, initialize
from rumen.podcasts.history import HistoryImportError, import_file, import_inputs

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "spotify_podcasts" / "history.json"


def test_given_mixed_history_when_imported_then_persists_only_podcast_evidence(
    tmp_path: Path,
) -> None:
    conn = initialize(tmp_path / "podcasts.db")
    result = import_file(conn, _FIXTURE)

    assert (result.imported, result.non_podcast, result.duplicates) == (1, 1, 0)
    event = conn.execute("SELECT * FROM spotify_podcast_listening_events").fetchone()
    provenance = conn.execute("SELECT record_index FROM spotify_podcast_import_events").fetchone()
    assert event["spotify_episode_uri"] == "spotify:episode:episode-one"
    assert event["raw_json"] == json.dumps(
        json.loads(_FIXTURE.read_text())[0], separators=(",", ":"), sort_keys=True
    )
    assert provenance["record_index"] == 0


def test_given_repeat_and_overlap_when_imported_then_events_are_deduplicated(
    tmp_path: Path,
) -> None:
    conn = initialize(tmp_path / "podcasts.db")
    first = import_file(conn, _FIXTURE)
    repeat = import_file(conn, _FIXTURE)
    overlap = tmp_path / "overlap.json"
    overlap.write_text(json.dumps(json.loads(_FIXTURE.read_text())[:1]))
    second = import_file(conn, overlap)

    assert first.imported == 1
    assert repeat.already_imported is True
    assert (second.imported, second.duplicates) == (0, 1)
    assert conn.execute("SELECT COUNT(*) FROM spotify_podcast_listening_events").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM spotify_podcast_import_events").fetchone()[0] == 2


def test_given_one_invalid_file_when_importing_multiple_then_valid_file_remains_committed(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.json"
    valid.write_bytes(_FIXTURE.read_bytes())
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps([{"spotify_episode_uri": "spotify:episode:x"}]))
    conn = initialize(tmp_path / "podcasts.db")

    with pytest.raises(HistoryImportError, match=r"invalid.json: record 0"):
        import_inputs(conn, [valid, invalid])

    assert conn.execute("SELECT COUNT(*) FROM spotify_podcast_imports").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM spotify_podcast_listening_events").fetchone()[0] == 1


def test_given_newer_cache_schema_when_opening_then_rejects_before_mutation(tmp_path: Path) -> None:
    path = tmp_path / "podcasts.db"
    conn = initialize(path)
    conn.execute("UPDATE podcast_cache_meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(CacheSchemaError, match="newer"):
        initialize(path)
