"""Synthetic-only feasibility evidence for the Chrome Reading List fallback (T-05)."""

from __future__ import annotations

import json
from pathlib import Path

from rumen.chrome_reading_list_fallback import run_offline_fallback_gate

_INVENTORY = (
    Path(__file__).parents[1] / "fixtures" / "chrome_reading_list_offline" / "inventory.json"
)


def test_gate_when_source_pinned_synthetic_records_match_then_passes_all_supported_milestones() -> (
    None
):
    report = run_offline_fallback_gate(_INVENTORY)

    assert report.fallback_enabled is False
    assert report.live_leveldb_opened is False
    assert report.overall_feasible is True
    assert [result.milestone for result in report.fixtures] == [120, 130, 140]
    assert all(result.feasible for result in report.fixtures)
    assert all(result.corruption_rejected for result in report.fixtures)
    assert all(result.unknown_fields_preserved for result in report.fixtures)


def test_gate_when_records_have_unicode_fragment_update_and_deletion_then_validates_their_identity() -> (
    None
):
    report = run_offline_fallback_gate(_INVENTORY)
    by_milestone = {result.milestone: result for result in report.fixtures}

    assert by_milestone[120].urls == ("https://example.invalid/na%C3%AFve?q=%E2%98%95#part",)
    assert by_milestone[130].updated_urls == (
        "https://example.invalid/%E6%9D%B1%E4%BA%AC?state=old#%E6%9B%B4%E6%96%B0",
    )
    assert by_milestone[140].deleted_urls == ("https://example.invalid/deleted-m140#gone",)
    assert all(result.unicode_url_count == 1 for result in report.fixtures)


def test_gate_when_fixture_declares_unknown_storage_variant_then_marks_fallback_infeasible(
    tmp_path: Path,
) -> None:
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    inventory["fixtures"][1]["storage_variant"] = "account"
    path = tmp_path / "unknown-variant.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    report = run_offline_fallback_gate(path)

    assert report.overall_feasible is False
    assert report.fallback_enabled is False
    assert by_milestone(report)[130].reason == "unsupported storage variant: account"


def test_gate_when_fixture_requires_a_live_database_then_marks_fallback_infeasible(
    tmp_path: Path,
) -> None:
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    inventory["fixtures"][0]["requires_live_leveldb"] = True
    path = tmp_path / "live-database.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    report = run_offline_fallback_gate(path)

    assert report.overall_feasible is False
    assert report.fallback_enabled is False
    assert by_milestone(report)[120].reason == "fixture requires a live LevelDB"


def test_gate_when_fixture_loses_its_source_pin_then_marks_fallback_infeasible(
    tmp_path: Path,
) -> None:
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    inventory["fixtures"][2]["chromium_proto_sha256"] = "0" * 64
    path = tmp_path / "unverified-source.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    report = run_offline_fallback_gate(path)

    assert report.overall_feasible is False
    assert report.fallback_enabled is False
    assert by_milestone(report)[140].reason == "fixture is not source-pinned"


def by_milestone(report: object) -> dict[int, object]:
    return {result.milestone: result for result in report.fixtures}  # type: ignore[attr-defined]
