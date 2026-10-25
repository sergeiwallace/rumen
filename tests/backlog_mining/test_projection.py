"""CC-task projection stays explicitly opted-in and session-scoped."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rumen.backlog_mining.projection import (
    CcTaskCreationDisabledError,
    NonCurrentSessionError,
    build_cc_task_plan,
)
from rumen.backlog_mining.repository import CandidateRepository
from rumen.backlog_mining.review import render_candidate
from rumen.backlog_mining.service import BacklogMiningService
from rumen.cli import run
from rumen.config import PipelineConfig, load_config
from rumen.digest import append_backlog_candidate_review_references


def _approved_draft(session):
    candidate = CandidateRepository(session).create_candidate(
        fingerprint="projection-family",
        source_snapshot={"source_claims": [{"url": "https://example.test/source"}]},
        content_snapshot={"title": "Projected draft", "claim_ids": ["source:1"]},
        assessment={"claims": [{"id": "source:1", "kind": "source"}]},
        configuration_snapshot={"template_revision": "v1"},
        evidence=[{"kind": "source", "metadata": {"url": "https://example.test/source"}}],
    )
    candidate.state = "accepted"
    return candidate, render_candidate(session, candidate.id)


def test_build_cc_task_plan_when_disabled_by_default_then_rejects_without_task_creation(
    session,
) -> None:
    candidate, draft = _approved_draft(session)

    with pytest.raises(CcTaskCreationDisabledError):
        build_cc_task_plan(
            candidate,
            draft,
            enabled=False,
            requested_session="session-a",
            invoking_session="session-a",
        )


def test_config_when_cc_task_creation_is_omitted_or_enabled_then_defaults_false_or_reads_nested_flag(
    tmp_path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"classifier_escalation_alarm_threshold": 0.25}))
    assert load_config(config_path).cc_task_creation.enabled is False

    config_path.write_text(
        json.dumps(
            {
                "classifier_escalation_alarm_threshold": 0.25,
                "cc_task_creation": {"enabled": True},
            }
        )
    )
    assert load_config(config_path).cc_task_creation.enabled is True


def test_backlog_cli_when_projection_is_disabled_or_session_is_noncurrent_then_rolls_back(
    capsys, monkeypatch, session, tmp_path
) -> None:
    candidate, _ = _approved_draft(session)
    session.commit()
    session.refresh(candidate)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"classifier_escalation_alarm_threshold": 0.25}))
    monkeypatch.setenv("RUMEN_CC_SESSION", "current")

    assert (
        run(
            [
                "backlog",
                "review",
                str(candidate.id),
                "--action",
                "edit",
                "--actor",
                "reviewer",
                "--rationale",
                "wording",
                "--create-cc-task",
                "--cc-session",
                "current",
                "--config",
                str(config_path),
            ],
            session,
        )
        == 1
    )
    assert "disabled" in capsys.readouterr().err
    assert candidate.state == "accepted"

    config_path.write_text(
        json.dumps(
            {
                "classifier_escalation_alarm_threshold": 0.25,
                "cc_task_creation": {"enabled": True},
            }
        )
    )
    assert (
        run(
            [
                "backlog",
                "review",
                str(candidate.id),
                "--action",
                "edit",
                "--actor",
                "reviewer",
                "--rationale",
                "wording",
                "--create-cc-task",
                "--cc-session",
                "other",
                "--config",
                str(config_path),
            ],
            session,
        )
        == 1
    )
    assert "current session" in capsys.readouterr().err
    assert candidate.state == "accepted"


def test_build_cc_task_plan_when_enabled_for_current_session_then_scopes_plan_to_one_candidate(
    session,
) -> None:
    candidate, draft = _approved_draft(session)

    plan = build_cc_task_plan(
        candidate,
        draft,
        enabled=True,
        requested_session="session-a",
        invoking_session="session-a",
    )

    assert plan["candidate_id"] == candidate.id
    assert plan["session"] == "session-a"
    assert plan["title"] == "Projected draft"
    assert plan["evidence_links"] == ["https://example.test/source"]


def test_build_cc_task_plan_when_session_is_not_current_then_rejects(session) -> None:
    candidate, draft = _approved_draft(session)

    with pytest.raises(NonCurrentSessionError, match="current session"):
        build_cc_task_plan(
            candidate,
            draft,
            enabled=True,
            requested_session="another-session",
            invoking_session="session-a",
        )


def test_review_surface_when_mining_transition_digest_and_projection_run_then_never_invokes_task_store_command(
    monkeypatch, session
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    BacklogMiningService(session, PipelineConfig()).mine()
    candidate, draft = _approved_draft(session)
    CandidateRepository(session).transition(
        candidate.id, action="edit", actor="reviewer", rationale="wording"
    )
    build_cc_task_plan(
        candidate,
        draft,
        enabled=True,
        requested_session="session-a",
        invoking_session="session-a",
    )
    markdown = append_backlog_candidate_review_references(
        "# Digest\n", [{"id": candidate.id, "fingerprint": candidate.fingerprint}]
    )

    assert calls == []
    assert f"rumen backlog show {candidate.fingerprint}" in markdown
    source_root = Path(__file__).resolve().parents[2] / "src" / "rumen"
    for module in (
        source_root / "backlog_mining" / "review.py",
        source_root / "backlog_mining" / "projection.py",
        source_root / "cli.py",
        source_root / "digest.py",
    ):
        assert "bd " not in module.read_text(encoding="utf-8")
