"""T-21 local digest polish preserves the committed extractive provenance boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select

from rumen.embeddings import HashingEmbeddingProvider, encode_embedding
from rumen.jobs import run_digest
from rumen.models import DigestEntity, FeedItem, TopicEntity

_NOW = datetime(2026, 7, 19, tzinfo=UTC)


class _Response:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def run(self, prompt: str) -> str:
        self.calls += 1
        return self.response


class _Unavailable:
    def run(self, prompt: str) -> str:
        raise OSError("Ollama unavailable")


def _seed(session) -> TopicEntity:
    provider = HashingEmbeddingProvider()
    entity = TopicEntity(
        canonical_title="Original title",
        canonical_url="https://example.invalid/entity",
        category="tool",
        earliest_seen=_NOW,
        source_count=1,
        embedding=encode_embedding(provider.embed(["local model release"])[0]),
    )
    session.add(entity)
    session.flush()
    session.add(
        FeedItem(
            source="fixture",
            source_id="one",
            url="https://example.invalid/source",
            title="Original title",
            content="A local model release has a useful command line interface.",
            published_at=_NOW,
            topic_entity_id=entity.id,
        )
    )
    session.commit()
    return entity


def _polished_response(source: str, sources: list[str]) -> str:
    return json.dumps(
        {
            "active_tool_updates": [
                {
                    "title": "Original title",
                    "summary": "Polished summary.",
                    "action": "Use it.",
                    "source": source,
                    "sources": sources,
                }
            ],
            "immediate_techniques": [],
            "horizon_developments": [],
        }
    )


def test_given_valid_local_polish_when_digest_runs_then_only_prose_changes(session) -> None:
    entity = _seed(session)
    llm = _Response(_polished_response(entity.canonical_url, ["https://example.invalid/source"]))
    digest = run_digest(
        session,
        period_start=_NOW,
        period_end=_NOW,
        llm_command=llm,
        local_polish=True,
        local_model_id="fixture-local@q4",
    )
    entry = digest.content_json["active_tool_updates"][0]
    assert entry == {
        "title": "Original title",
        "summary": "Polished summary.",
        "action": "Use it.",
        "source": entity.canonical_url,
        "sources": ["https://example.invalid/source"],
    }
    assert digest.model_used == "fixture-local@q4"
    assert list(
        session.scalars(
            select(DigestEntity.topic_entity_id).where(DigestEntity.digest_id == digest.id)
        )
    ) == [entity.id]


def test_given_disabled_polish_or_no_local_model_when_digest_runs_then_no_command_is_called(
    session,
) -> None:
    _seed(session)
    llm = _Response("not used")
    digest = run_digest(session, period_start=_NOW, period_end=_NOW, llm_command=llm)
    assert llm.calls == 0
    assert digest.model_used == "extractive"


def test_given_invalid_or_unavailable_polish_when_digest_runs_then_extractive_digest_is_retained(
    session, caplog
) -> None:
    _seed(session)
    digest = run_digest(
        session,
        period_start=_NOW,
        period_end=_NOW,
        llm_command=_Unavailable(),
        local_polish=True,
        local_model_id="fixture-local@q4",
    )
    assert digest.model_used == "extractive"
    assert "local digest polish failed" in caplog.text


def test_given_provenance_changing_polish_when_digest_runs_then_it_is_rejected(session) -> None:
    entity = _seed(session)
    llm = _Response(_polished_response("https://wrong.invalid", ["https://example.invalid/source"]))
    digest = run_digest(
        session,
        period_start=_NOW,
        period_end=_NOW,
        llm_command=llm,
        local_polish=True,
        local_model_id="fixture-local@q4",
    )
    assert digest.model_used == "extractive"
    assert digest.content_json["active_tool_updates"][0]["source"] == entity.canonical_url
