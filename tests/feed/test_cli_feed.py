"""T-09: rumen CLI surface — views, digest, unknown-view failure path."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from rumen.cli import run
from rumen.models import Digest, FeedItem, FeedView, SourceMaintenanceRun

_NOW = datetime(2026, 7, 10, tzinfo=UTC)


def _seed(session) -> None:
    session.add(FeedView(name="AI/ML", filter_json={"category": ["tool", "paper"]}))
    for sid, cat in [("t1", "tool"), ("pol1", "politics")]:
        session.add(
            FeedItem(
                source="s",
                source_id=sid,
                url=f"u/{sid}",
                title=sid,
                category=cat,
                published_at=_NOW,
            )
        )
    session.commit()


def test_cli_items_view_filters(session, capsys) -> None:
    _seed(session)
    code = run(["feed", "items", "--view", "AI/ML"], session)
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert {i["category"] for i in out} == {"tool"}  # only tool present in this seed


def test_cli_items_ad_hoc_category(session, capsys) -> None:
    _seed(session)
    code = run(["feed", "items", "--category", "politics"], session)
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert {i["category"] for i in out} == {"politics"}


def test_cli_unknown_view_exits_nonzero_with_message(session, capsys) -> None:
    _seed(session)
    code = run(["feed", "items", "--view", "nope"], session)
    assert code == 1
    assert "no such view" in capsys.readouterr().err


def test_cli_digest_outputs_latest(session, capsys) -> None:
    session.add(
        Digest(
            period_start=_NOW,
            period_end=_NOW,
            content_json={"active_tool_updates": []},
            content_markdown="# d",
        )
    )
    session.commit()
    code = run(["feed", "digest"], session)
    assert code == 0
    assert "active_tool_updates" in capsys.readouterr().out


def test_cli_digest_missing_exits_nonzero(session, capsys) -> None:
    code = run(["feed", "digest"], session)
    assert code == 1


def test_cli_run_with_digest_flag_is_end_to_end(session, capsys, tmp_path) -> None:
    # `rumen run --digest` runs the pipeline THEN synthesizes the digest in one command.
    cfg = tmp_path / "rumen.json"
    # A tiny stdin-reading LLM stub that emits valid empty-section digest JSON.
    cfg.write_text(
        json.dumps(
            {
                "sources": [],
                "classifier_escalation_alarm_threshold": 0.25,
                "llm_command": [
                    "python3",
                    "-c",
                    (
                        "import sys; sys.stdin.read(); "
                        'print(\'{"active_tool_updates":[],"immediate_techniques":[],'
                        '"horizon_developments":[]}\')'
                    ),
                ],
            }
        )
    )
    code = run(["run", "--config", str(cfg), "--digest"], session)
    assert code == 0
    assert session.query(Digest).count() == 1  # the refresh produced a digest end-to-end


def test_cli_run_digest_flag_without_llm_produces_extractive_digest(session, tmp_path) -> None:
    # D-21: the digest defaults to the zero-LLM extractive path, so `--digest` produces a
    # digest even with no llm_command configured (no more skip).
    cfg = tmp_path / "rumen.json"
    cfg.write_text(json.dumps({"sources": [], "classifier_escalation_alarm_threshold": 0.25}))
    code = run(["run", "--config", str(cfg), "--digest"], session)
    assert code == 0
    assert session.query(Digest).count() == 1  # extractive digest, no LLM needed


def test_cli_digest_without_llm_produces_extractive_digest(session, tmp_path) -> None:
    cfg = tmp_path / "rumen.json"
    cfg.write_text(json.dumps({"sources": [], "classifier_escalation_alarm_threshold": 0.25}))
    code = run(["digest", "--config", str(cfg)], session)
    assert code == 0
    assert session.query(Digest).count() == 1


def test_cli_reddit_purge_without_confirmation_has_no_mutation(session, capsys) -> None:
    session.add(
        FeedItem(
            source="reddit:ai",
            source_id="t3_one",
            url="https://example.test",
            title="title",
            published_at=_NOW,
        )
    )
    session.commit()
    assert run(["reddit", "purge", "--profile", "ai"], session) == 1
    assert "requires --confirm" in capsys.readouterr().err
    assert session.query(FeedItem).filter_by(source="reddit:ai").count() == 1


def test_cli_feed_status_shows_content_free_maintenance_counts(session, capsys) -> None:
    session.add(
        SourceMaintenanceRun(
            source="reddit:ai",
            started_at=_NOW,
            completed_at=_NOW,
            status="completed",
            items_checked=4,
            items_purged=1,
            reason_counts={"removed": 1},
        )
    )
    session.commit()
    assert run(["feed", "status"], session) == 0
    assert "maintenance=completed\tchecked=4\tpurged=1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        (["--help"], 0),
        (["feed", "--help"], 0),
        (["feed", "items", "--help"], 0),
        ([], 2),
        (["unknown"], 2),
        (["feed", "items", "--limit", "invalid"], 2),
        (["run", "--config"], 2),
        (["--version"], 2),
    ],
)
def test_given_no_database_when_main_parses_help_or_invalid_arguments_then_storage_is_untouched(
    monkeypatch, capsys, argv, exit_code
) -> None:
    from rumen.cli import main

    monkeypatch.delenv("RUMEN_DB_DSN", raising=False)
    monkeypatch.setenv("RUMEN_CONFIG", "missing-config.json")
    storage = Mock(side_effect=AssertionError("storage must not open"))
    engine = Mock(side_effect=AssertionError("engine must not be created"))
    migration = Mock(side_effect=AssertionError("migrations must not run"))
    monkeypatch.setattr("rumen.storage.open_storage", storage)
    monkeypatch.setattr("rumen.db.engine.create_engine", engine)
    monkeypatch.setattr("rumen.db.migrate.upgrade_to_head", migration)
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert raised.value.code == exit_code
    output = capsys.readouterr()
    assert "usage:" in (output.out if exit_code == 0 else output.err)
    storage.assert_not_called()
    engine.assert_not_called()
    migration.assert_not_called()
