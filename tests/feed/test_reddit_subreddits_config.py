from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rumen.adapters.reddit_subreddits import (
    PrawRedditGateway,
    RedditFetchError,
)
from rumen.config import load_config


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"classifier_escalation_alarm_threshold": 0.25, "sources": [source]})
    )
    monkeypatch.delenv("AI_HOST", raising=False)
    return load_config(path)


def _valid(**updates: object) -> dict:
    source = {
        "type": "reddit_subreddits",
        "enabled": True,
        "profile": "ai",
        "subreddits": ["MachineLearning"],
        "modes": ["submissions"],
        "client_id_env": "REDDIT_CLIENT",
        "client_secret_env": "REDDIT_SECRET",
        "user_agent": "rumen tests",
        "approval_reference": "approved",
    }
    source.update(updates)
    return source


def test_given_valid_enabled_source_when_loaded_then_builds_adapter_and_maintainer_without_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _load(tmp_path, monkeypatch, _valid())
    assert [item.source_name() for item in cfg.adapters] == ["reddit:ai"]
    assert [item.source_name() for item in cfg.source_maintainers] == ["reddit:ai"]


def test_given_disabled_source_when_loaded_then_builds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _load(tmp_path, monkeypatch, _valid(enabled=False))
    assert cfg.adapters == []
    assert cfg.source_maintainers == []


def test_given_disabled_source_with_unknown_key_when_loaded_then_rejects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        _load(tmp_path, monkeypatch, _valid(enabled=False, unexpected=True))


@pytest.mark.parametrize(
    "update",
    [
        {"approval_reference": ""},
        {"client_id_env": "not valid"},
        {"subreddits": ["r/test"]},
        {"subreddits": ["test", "TEST"]},
        {"modes": ["other"]},
        {"retention_hours": True},
        {"unknown": 1},
    ],
)
def test_given_invalid_enabled_source_when_loaded_then_fails_before_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, update: dict
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _load(tmp_path, monkeypatch, _valid(**update))


def test_given_missing_credential_when_gateway_used_then_error_does_not_disclose_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REDDIT_SECRET", "secret-canary-53")
    adapter = _load(tmp_path, monkeypatch, _valid()).adapters[0]
    with pytest.raises(RedditFetchError) as exc:
        adapter.gateway.listing("MachineLearning", "submissions", limit=1)
    assert "secret-canary-53" not in str(exc.value)


def test_given_missing_optional_dependency_when_gateway_used_then_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _valid()
    monkeypatch.setenv("REDDIT_CLIENT", "id")
    monkeypatch.setenv("REDDIT_SECRET", "secret-canary")
    monkeypatch.setitem(sys.modules, "praw", None)
    from rumen.config import _validate_reddit_subreddits_spec

    gateway = PrawRedditGateway(_validate_reddit_subreddits_spec(settings))
    with pytest.raises(RedditFetchError, match="install") as exc:
        gateway.listing("MachineLearning", "submissions", limit=1)
    assert "secret-canary" not in str(exc.value)
