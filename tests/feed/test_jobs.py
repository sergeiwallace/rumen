"""T-10: scheduling entrypoints — full chain, digest one-per-period, idempotency."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from rumen.adapters.github_releases import GitHubReleasesAdapter
from rumen.adapters.youtube import YouTubeApiClient, YouTubePlaylistAdapter
from rumen.config import load_config
from rumen.embeddings import EMBEDDING_DIM, encode_embedding
from rumen.ingest import last_high_water_mark, run_ingestion
from rumen.jobs import run_digest, run_pipeline
from rumen.models import (
    Digest,
    DigestEntity,
    FeedItem,
    FeedRun,
    InterestProfileRow,
    TopicEntity,
)
from rumen.ranking.interest_profile import InterestProfile
from rumen.ranking.rank import rank_topic_entities
from rumen.youtube_enrichment import TranscriptTrack

_ATOM = (Path(__file__).parents[1] / "fixtures" / "github_releases.atom").read_text()
_NOW = datetime(2026, 7, 16, tzinfo=UTC)


def _gh() -> GitHubReleasesAdapter:
    return GitHubReleasesAdapter("anthropics/anthropic-sdk-python", fetcher=lambda _u: _ATOM)


class _RaisingAdapter:
    def source_name(self) -> str:
        return "github_releases:anthropics/anthropic-sdk-python"

    def fetch_new(self, since):
        raise RuntimeError("boom")


class _StubLlm:
    def run(self, prompt: str) -> str:
        return json.dumps(
            {"active_tool_updates": [], "immediate_techniques": [], "horizon_developments": []}
        )


def test_run_pipeline_executes_full_chain_and_succeeds(session) -> None:
    result = run_pipeline(session, adapters=[_gh()], now=_NOW)
    assert result.ok is True
    assert result.exit_code == 0
    # ingestion happened, entities were grouped + ranked.
    assert session.query(FeedItem).count() == 2
    assert session.query(TopicEntity).count() >= 1
    assert session.query(FeedRun).filter_by(status="completed").count() == 1
    assert result.ranked_count >= 1


def test_run_pipeline_failed_adapter_reports_nonzero_without_advancing_hwm(session) -> None:
    # Seed a good completed run first to establish a cursor.
    good = run_ingestion(session, _gh())
    good_hwm = good.high_water_mark

    result = run_pipeline(session, adapters=[_RaisingAdapter()], now=_NOW)
    assert result.ok is False
    assert result.exit_code == 1
    # The failed run did not advance the cursor (idempotency holds for a retry).
    assert last_high_water_mark(
        session, "github_releases:anthropics/anthropic-sdk-python"
    ) == datetime.fromisoformat(good_hwm)


def test_run_pipeline_is_idempotent_on_rerun(session) -> None:
    run_pipeline(session, adapters=[_gh()], now=_NOW)
    run_pipeline(session, adapters=[_gh()], now=_NOW)
    assert session.query(FeedItem).count() == 2  # no duplicate ingest


def test_given_youtube_discovery_when_pipeline_runs_then_persists_lifecycle_and_transcript_before_grouping(
    session, monkeypatch
) -> None:
    monkeypatch.setenv("YT_KEY", "fixture-only")
    pages = {
        "items": [
            {
                "id": "playlist-item",
                "snippet": {"publishedAt": "2026-07-15T00:00:00Z", "title": "fallback"},
                "contentDetails": {"videoId": "video-1"},
            }
        ]
    }
    details = {
        "items": [
            {
                "id": "video-1",
                "snippet": {
                    "title": "video",
                    "description": "description",
                    "channelTitle": "channel",
                },
                "contentDetails": {"duration": "PT1M"},
                "status": {"caption": "true"},
            }
        ]
    }
    client = YouTubeApiClient(
        "YT_KEY",
        100,
        transport=lambda endpoint, _params: (
            200,
            {},
            pages if endpoint == "playlistItems" else details,
        ),
    )
    adapter = YouTubePlaylistAdapter(
        "playlist-a",
        "YT_KEY",
        100,
        client=client,
        transcript_mode="unofficial_captions",
        transcript_languages=["en"],
    )

    class Provider:
        name = "fixture"
        version = "1"

        def list_tracks(self, _video_id):
            return [TranscriptTrack("en", "manual", "transcript")]

    result = run_pipeline(session, adapters=[adapter], transcript_provider=Provider(), now=_NOW)

    assert result.ok is True
    assert result.youtube[adapter.source_name()]["transcript_created"] == 1
    assert session.query(FeedItem).filter_by(source="youtube_playlist:playlist-a").count() == 1
    assert session.query(FeedItem).filter_by(source="youtube_transcript:playlist-a").count() == 1
    pointers = session.query(FeedItem).order_by(FeedItem.source).all()
    assert pointers[0].topic_entity_id == pointers[1].topic_entity_id


def test_given_no_explicit_profile_when_run_pipeline_runs_then_ranking_uses_the_persisted_profile(
    session,
) -> None:
    x_axis = TopicEntity(
        canonical_title="x-axis",
        canonical_url="https://example.test/x-axis",
        earliest_seen=_NOW,
        source_count=1,
        embedding=encode_embedding(_vector(1.0, 0.0)),
    )
    y_axis = TopicEntity(
        canonical_title="y-axis",
        canonical_url="https://example.test/y-axis",
        earliest_seen=_NOW,
        source_count=1,
        embedding=encode_embedding(_vector(0.0, 1.0)),
    )
    session.add_all([x_axis, y_axis])
    session.flush()
    session.add(InterestProfileRow(centroid=encode_embedding(_vector(0.0, 1.0)), alpha=0.1))
    session.commit()

    run_pipeline(
        session,
        adapters=[],
        now=_NOW,
        mmr_lambda=1.0,
        exploration_fraction=0.0,
    )

    assert y_axis.rank_position == 0
    assert x_axis.rank_position == 1


def test_run_digest_produces_one_per_period(session) -> None:
    session.add(
        TopicEntity(canonical_title="e", canonical_url="u", earliest_seen=_NOW, source_count=1)
    )
    session.commit()
    start, end = _NOW - timedelta(days=7), _NOW

    first = run_digest(session, llm_command=_StubLlm(), period_start=start, period_end=end)
    again = run_digest(session, llm_command=_StubLlm(), period_start=start, period_end=end)
    assert again.id == first.id  # re-invocation within the period is a no-op
    assert session.query(Digest).count() == 1


@pytest.mark.parametrize("model_used", [None, "custom-model"])
def test_given_abstractive_digest_when_generated_then_persists_model_label(
    session, model_used
) -> None:
    options = {} if model_used is None else {"model_used": model_used}
    digest = run_digest(
        session,
        llm_command=_StubLlm(),
        period_start=_NOW - timedelta(days=7),
        period_end=_NOW,
        extractive=False,
        **options,
    )

    assert session.get(Digest, digest.id).model_used == (model_used or "unspecified")


def test_run_digest_force_regenerates(session) -> None:
    session.add(
        TopicEntity(canonical_title="e", canonical_url="u", earliest_seen=_NOW, source_count=1)
    )
    session.commit()
    start, end = _NOW - timedelta(days=7), _NOW

    run_digest(session, llm_command=_StubLlm(), period_start=start, period_end=end)
    run_digest(session, llm_command=_StubLlm(), period_start=start, period_end=end, force=True)
    assert session.query(Digest).count() == 2


def _vector(*coordinates: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(coordinates)] = coordinates
    return vector


def _digest_entity_ids(session, digest_id: int) -> list[int]:
    return list(
        session.scalars(
            select(DigestEntity.topic_entity_id)
            .where(DigestEntity.digest_id == digest_id)
            .order_by(DigestEntity.rank)
        )
    )


def test_run_digest_uses_mmr_rank_position_not_plain_relevance_score(session) -> None:
    # A and B are near duplicates. B has higher raw relevance than C, but MMR (lambda=.4)
    # promotes the more diverse C ahead of B after selecting A.
    entities = [
        TopicEntity(
            canonical_title=title,
            canonical_url=f"https://example.com/{title}",
            earliest_seen=_NOW,
            source_count=1,
            embedding=encode_embedding(vector),
        )
        for title, vector in [
            ("A", _vector(1.0, 0.0)),
            ("B", _vector(0.9, (1.0 - 0.9**2) ** 0.5)),
            ("C", _vector(0.2, (1.0 - 0.2**2) ** 0.5)),
        ]
    ]
    session.add_all(entities)
    session.commit()

    ranked = rank_topic_entities(
        session,
        InterestProfile(_vector(1.0, 0.0)),
        now=_NOW,
        lambda_mmr=0.4,
        exploration_fraction=0.0,
    )
    rank_order = [result.entity_id for result in ranked]
    plain_relevance_order = [
        entity.id
        for entity in sorted(entities, key=lambda entity: entity.relevance_score, reverse=True)
    ]
    assert rank_order == [entities[0].id, entities[2].id, entities[1].id]
    assert plain_relevance_order == [entities[0].id, entities[1].id, entities[2].id]
    assert rank_order != plain_relevance_order

    digest = run_digest(
        session,
        period_start=_NOW - timedelta(days=7),
        period_end=_NOW,
        top_n=3,
    )
    assert _digest_entity_ids(session, digest.id) == rank_order


def test_run_digest_sorts_never_ranked_entities_after_ranked_entities(session) -> None:
    ranked_first = TopicEntity(
        canonical_title="first",
        canonical_url="https://example.com/first",
        earliest_seen=_NOW,
        source_count=1,
        relevance_score=0.1,
        rank_position=0,
    )
    ranked_second = TopicEntity(
        canonical_title="second",
        canonical_url="https://example.com/second",
        earliest_seen=_NOW,
        source_count=1,
        relevance_score=0.2,
        rank_position=1,
    )
    never_ranked = TopicEntity(
        canonical_title="never",
        canonical_url="https://example.com/never",
        earliest_seen=_NOW,
        source_count=1,
        relevance_score=1.0,
    )
    session.add_all([ranked_first, ranked_second, never_ranked])
    session.commit()

    digest = run_digest(
        session,
        period_start=_NOW - timedelta(days=7),
        period_end=_NOW,
        top_n=3,
    )
    assert _digest_entity_ids(session, digest.id) == [
        ranked_first.id,
        ranked_second.id,
        never_ranked.id,
    ]


def test_load_config_builds_adapters(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rumen.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"type": "github_releases", "repo": "owner/repo"},
                    {"type": "npm_registry", "package": "@a/b", "tool_name": "B"},
                ],
                "llm_command": ["model-cli", "--model", "model-name"],
                "out_of_scope": ["gambling"],
                "digest_top_n": 10,
                "classifier_escalation_alarm_threshold": 0.25,
            }
        )
    )
    cfg = load_config(cfg_path)
    assert len(cfg.adapters) == 2
    assert cfg.llm_command is not None
    assert cfg.out_of_scope == frozenset({"gambling"})
    assert cfg.digest_top_n == 10


def test_load_config_when_new_connector_specs_are_valid_then_builds_them_with_existing_adapters(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "rumen.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"type": "github_releases", "repo": "owner/releases"},
                    {"type": "github_stars", "token_env": "GITHUB_TOKEN"},
                    {
                        "type": "github_stars_list",
                        "token_env": "GITHUB_TOKEN",
                        "list_name": "AI Coding Tools",
                    },
                    {
                        "type": "chrome_bookmarks",
                        "bookmarks_path": "/operator/Bookmarks",
                        "profile": "Default",
                    },
                    {
                        "type": "github_repository",
                        "repo": "owner/tool",
                        "token_env": "GITHUB_TOKEN",
                    },
                ],
                "classifier_escalation_alarm_threshold": 0.25,
            }
        )
    )

    cfg = load_config(cfg_path)

    assert [adapter.source_name() for adapter in cfg.adapters] == [
        "github_releases:owner/releases",
        "github_stars",
        "github_stars_list:AI Coding Tools",
        "chrome_bookmarks:Default",
        "github_repositories:owner/tool",
    ]
    assert cfg.adapters[1].token_env == "GITHUB_TOKEN"
    assert cfg.adapters[2].token_env == "GITHUB_TOKEN"
    assert "test-token" not in repr(cfg.adapters[1])


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"type": "github_stars", "token_env": "GITHUB_TOKEN", "extra": True}, "github_stars"),
        (
            {"type": "github_stars_list", "token_env": "GITHUB_TOKEN", "list_name": ""},
            "github_stars_list list_name",
        ),
        (
            {"type": "chrome_bookmarks", "bookmarks_path": "/operator/Bookmarks", "profile": ""},
            "chrome_bookmarks profile",
        ),
        (
            {"type": "github_repository", "repo": "owner/repo/extra", "token_env": "GITHUB_TOKEN"},
            "github_repository repo",
        ),
        ({"type": "not_a_source"}, "unknown source type"),
    ],
)
def test_load_config_when_new_source_spec_is_invalid_then_rejects_before_ingestion(
    tmp_path: Path, source: dict, message: str
) -> None:
    cfg_path = tmp_path / "rumen.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": [source],
                "classifier_escalation_alarm_threshold": 0.25,
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        load_config(cfg_path)


def test_load_config_builds_fallback_ladder(tmp_path: Path, monkeypatch) -> None:
    # D-3 amendment: `llm_commands` (plural) builds a FallbackLlmCommand chain.
    from rumen.llm import FallbackLlmCommand

    cfg_path = tmp_path / "rumen.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": [],
                "local_llm_by_host": {
                    "test-host": {
                        "backend": "ollama",
                        "command": [
                            "env",
                            "OLLAMA_HOST=127.0.0.1:11434",
                            "ollama",
                            "run",
                            "llama3",
                        ],
                        "model_id": "llama3@fixture",
                    }
                },
                "llm_commands": [["model-cli"], ["codex"]],
                "llm_timeout": 15.0,
                "classifier_escalation_alarm_threshold": 0.25,
            }
        )
    )
    monkeypatch.setenv("AI_HOST", "test-host")
    cfg = load_config(cfg_path)
    assert isinstance(cfg.llm_command, FallbackLlmCommand)
    assert len(cfg.llm_command.commands) == 3
    assert cfg.llm_command.commands[0].timeout == 15.0  # timeout flows into each tier


def test_load_config_rejects_missing_classifier_alarm_threshold(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rumen.json"
    cfg_path.write_text(json.dumps({"sources": []}))
    import pytest

    with pytest.raises(ValueError, match="classifier_escalation_alarm_threshold"):
        load_config(cfg_path)


def test_load_config_overrides_every_tunable(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rumen.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": [],
                "llm_command": ["mymodel"],
                "commit_batch_size": 42,
                "max_llm_calls": 7,
                "classify_write_max_attempts": 4,
                "classify_write_retry_backoff_seconds": 0.25,
                "llm_timeout": 30.0,
                "tau_class": 0.9,
                "tau_high": 0.95,
                "tau_low": 0.6,
                "mmr_lambda": 0.3,
                "exploration_fraction": 0.2,
                "half_life_days": {"tool": 1.5},
                "default_half_life_days": 21.0,
                "classifier_escalation_alarm_threshold": 0.31,
            }
        )
    )
    cfg = load_config(cfg_path)
    # Every knob is config-driven — no magic numbers survive an override.
    assert cfg.commit_batch_size == 42
    assert cfg.max_llm_calls == 7
    assert cfg.classify_write_max_attempts == 4
    assert cfg.classify_write_retry_backoff_seconds == 0.25
    assert cfg.llm_timeout == 30.0
    assert cfg.llm_command.timeout == 30.0  # timeout flows into the subprocess command
    assert cfg.tau_class == 0.9
    assert cfg.tau_high == 0.95
    assert cfg.tau_low == 0.6
    assert cfg.mmr_lambda == 0.3
    assert cfg.exploration_fraction == 0.2
    assert cfg.half_life_days == {"tool": 1.5}
    assert cfg.default_half_life_days == 21.0
    assert cfg.classifier_escalation_alarm_threshold == 0.31


def test_budgeted_llm_caps_calls_and_breaks_on_repeated_failures() -> None:
    # Hardening: a per-run call budget + consecutive-failure breaker so a routing regression
    # or exhausted quota can never drain the backend with thousands of calls.
    import subprocess

    import pytest

    from rumen.llm import BudgetedLlmCommand, LlmUnavailable

    class _Ok:
        def run(self, prompt: str) -> str:
            return "ok"

    b = BudgetedLlmCommand(_Ok(), max_calls=2)
    assert b.run("x") == "ok"
    assert b.run("x") == "ok"
    with pytest.raises(LlmUnavailable):
        b.run("x")  # 3rd call refused without touching the backend
    assert b.calls == 2

    class _Fail:
        def run(self, prompt: str) -> str:
            raise subprocess.CalledProcessError(1, ["model-cli"], stderr="quota")

    b2 = BudgetedLlmCommand(_Fail(), max_calls=100, max_consecutive_failures=3)
    for _ in range(3):
        with pytest.raises(subprocess.CalledProcessError):
            b2.run("x")
    assert b2.breaker_open
    with pytest.raises(LlmUnavailable):
        b2.run("x")  # breaker open → fast refuse, inner not invoked
    assert b2.calls == 3
