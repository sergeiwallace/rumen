from __future__ import annotations

import pytest

from rumen.config import PipelineConfig, StorageConfig
from rumen.jobs import PipelineResult, run_backlog_refresh


def test_refresh_when_digest_disabled_then_calls_pipeline_then_mining(monkeypatch, session) -> None:
    calls: list[str] = []

    def pipeline(*_args, **_kwargs):
        calls.append("pipeline")
        return PipelineResult()

    def mining(*_args, **_kwargs):
        calls.append("mining")
        from rumen.jobs import BacklogMiningJobResult

        return BacklogMiningJobResult()

    monkeypatch.setattr("rumen.jobs.run_pipeline", pipeline)
    monkeypatch.setattr("rumen.jobs.run_backlog_mining", mining)
    run_backlog_refresh(session, config=PipelineConfig(), digest_enabled=False)
    assert calls == ["pipeline", "mining"]


def test_refresh_when_digest_enabled_then_calls_pipeline_digest_then_mining(
    monkeypatch, session
) -> None:
    calls: list[str] = []

    def pipeline(*_args, **_kwargs):
        calls.append("pipeline")
        return PipelineResult()

    def digest(*_args, **_kwargs):
        calls.append("digest")
        return object()

    def mining(*_args, **_kwargs):
        calls.append("mining")
        from rumen.jobs import BacklogMiningJobResult

        return BacklogMiningJobResult()

    monkeypatch.setattr("rumen.jobs.run_pipeline", pipeline)
    monkeypatch.setattr("rumen.jobs.run_digest", digest)
    monkeypatch.setattr("rumen.jobs.run_backlog_mining", mining)
    run_backlog_refresh(session, config=PipelineConfig(), digest_enabled=True)
    assert calls == ["pipeline", "digest", "mining"]


def test_refresh_when_unsupported_backend_selected_then_fails_before_pipeline_work(
    monkeypatch, session
) -> None:
    monkeypatch.setattr(
        "rumen.jobs.run_pipeline",
        lambda *_args, **_kwargs: pytest.fail("pipeline ran"),
    )

    with pytest.raises(ValueError, match="storage.backend must be 'sqlalchemy'"):
        run_backlog_refresh(
            session,
            config=PipelineConfig(storage=StorageConfig(backend="unsupported")),
        )
