"""T-08: digest synthesis — four sections, action routing, multi-period, clean failure."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from rumen.digest import (
    polish_extractive_digest,
    synthesize_digest,
    synthesize_digest_extractive,
)
from rumen.embeddings import HashingEmbeddingProvider
from rumen.llm import FallbackLlmCommand
from rumen.models import Digest, DigestEntity, FeedItem, TopicEntity

_START = datetime(2026, 7, 9, tzinfo=UTC)
_END = datetime(2026, 7, 16, tzinfo=UTC)


class _StubLlm:
    def __init__(self, response: str) -> None:
        self.response = response

    def run(self, prompt: str) -> str:
        return self.response


class _FailingLlm:
    def run(self, prompt: str) -> str:
        raise OSError("unavailable")


def _entity(session, title: str) -> TopicEntity:
    e = TopicEntity(
        canonical_title=title,
        canonical_url=f"https://example.com/{title}",
        earliest_seen=_START,
        source_count=2,
    )
    session.add(e)
    session.flush()
    return e


def _constituent(
    session,
    entity: TopicEntity,
    *,
    source: str,
    title: str,
    content: str | None = None,
    source_id: str | None = None,
) -> FeedItem:
    item = FeedItem(
        source=source,
        source_id=source_id or f"{source}-{entity.id}",
        url=f"https://example.com/{source}/{entity.id}",
        title=title,
        content=content,
        published_at=_START,
        topic_entity_id=entity.id,
    )
    session.add(item)
    session.flush()
    return item


def _extractive_digest(session, entity: TopicEntity, *, llm_command=None):
    return synthesize_digest_extractive(
        session,
        entity_ids=[entity.id],
        period_start=_START,
        period_end=_END,
        provider=HashingEmbeddingProvider(),
        llm_command=llm_command,
    )


def _good_response() -> str:
    return json.dumps(
        {
            "active_tool_updates": [
                {"title": "Tool A", "action": "run update", "summary": "s"},
                {"title": "Tool B no action", "summary": "s"},  # no action → horizon
            ],
            "immediate_techniques": [{"title": "Tech", "action": "try it", "summary": "s"}],
            "horizon_developments": [],
        }
    )


def test_produces_digest_with_four_sections(session) -> None:
    e = _entity(session, "A")
    session.commit()
    digest = synthesize_digest(
        session,
        llm_command=_StubLlm(_good_response()),
        entity_ids=[e.id],
        period_start=_START,
        period_end=_END,
        suppressed_count=7,
    )
    stored = session.get(Digest, digest.id)
    assert stored.model_used == "unspecified"
    assert set(stored.content_json) == {
        "active_tool_updates",
        "immediate_techniques",
        "horizon_developments",
        "noise_suppressed",
    }


def test_actionless_entry_routed_to_horizon(session) -> None:
    e = _entity(session, "A")
    session.commit()
    digest = synthesize_digest(
        session,
        llm_command=_StubLlm(_good_response()),
        entity_ids=[e.id],
        period_start=_START,
        period_end=_END,
    )
    cj = digest.content_json
    assert all(entry.get("action") for entry in cj["active_tool_updates"])
    horizon_titles = {entry["title"] for entry in cj["horizon_developments"]}
    assert e.display_title in horizon_titles


def test_noise_suppressed_count_matches_period(session) -> None:
    e = _entity(session, "A")
    session.commit()
    digest = synthesize_digest(
        session,
        llm_command=_StubLlm(_good_response()),
        entity_ids=[e.id],
        period_start=_START,
        period_end=_END,
        suppressed_count=23,
    )
    assert digest.content_json["noise_suppressed"]["count"] == 23


def test_entity_can_appear_in_multiple_period_digests(session) -> None:
    e = _entity(session, "A")
    session.commit()
    synthesize_digest(
        session,
        llm_command=_StubLlm(_good_response()),
        entity_ids=[e.id],
        period_start=_START,
        period_end=_END,
    )
    synthesize_digest(
        session,
        llm_command=_StubLlm(_good_response()),
        entity_ids=[e.id],
        period_start=_END,
        period_end=_END + timedelta(days=7),
    )
    memberships = session.query(DigestEntity).filter_by(topic_entity_id=e.id).all()
    assert len(memberships) == 2
    assert len({m.digest_id for m in memberships}) == 2  # distinct digests


def test_unparseable_output_writes_no_partial_digest(session) -> None:
    e = _entity(session, "A")
    session.commit()
    # A prior good digest is the latest.
    synthesize_digest(
        session,
        llm_command=_StubLlm(_good_response()),
        entity_ids=[e.id],
        period_start=_START,
        period_end=_END,
    )
    assert session.query(Digest).count() == 1

    with pytest.raises(ValueError, match="not valid JSON"):
        synthesize_digest(
            session,
            llm_command=_StubLlm("garbage not json"),
            entity_ids=[e.id],
            period_start=_END,
            period_end=_END + timedelta(days=7),
        )
    session.rollback()
    assert session.query(Digest).count() == 1  # no partial row; prior stays latest


def test_given_an_entity_with_a_real_title_constituent_when_digest_synthesizes_then_display_title_is_the_canonical_title(
    session,
) -> None:
    entity = _entity(session, "Canonical release title")
    entity.display_title = "stale generated title"
    _constituent(
        session,
        entity,
        source="article",
        title="An informative article title",
    )
    session.commit()

    digest = _extractive_digest(session, entity)

    assert entity.display_title == entity.canonical_title
    assert digest.content_json["horizon_developments"][0]["title"] == entity.canonical_title


def test_given_a_tweet_only_entity_when_digest_synthesizes_then_display_title_is_heuristically_generated_and_persisted(
    session,
) -> None:
    entity = _entity(session, "raw social text")
    _constituent(
        session,
        entity,
        source="reddit",
        title="@someone https://example.com/thread Useful release notes for local agents.",
    )
    session.commit()

    digest = _extractive_digest(session, entity)

    assert entity.display_title == "Useful release notes for local agents."
    assert digest.content_json["horizon_developments"][0]["title"] == entity.display_title


def test_given_a_cached_display_title_when_digest_synthesizes_again_then_it_is_reused_without_regeneration(
    session,
) -> None:
    entity = _entity(session, "raw social text")
    item = _constituent(
        session,
        entity,
        source="reddit",
        title="Initial social post title.",
    )
    entity.display_title = "Cached digest title"
    session.commit()

    _extractive_digest(session, entity)
    item.title = "A completely different title that would replace a regenerated cache."
    session.commit()
    digest = _extractive_digest(session, entity)

    assert entity.display_title == "Cached digest title"
    assert digest.content_json["horizon_developments"][0]["title"] == "Cached digest title"


def test_given_a_tweet_only_entity_gains_a_real_title_constituent_when_digest_synthesizes_then_display_title_self_heals_to_canonical_title(
    session,
) -> None:
    entity = _entity(session, "Raw social text")
    _constituent(session, entity, source="reddit", title="Raw social text")
    entity.display_title = "Old generated title"
    _constituent(
        session,
        entity,
        source="article",
        title="Informative article title",
        source_id="article-title",
    )
    entity.canonical_title = "Informative article title"
    session.commit()

    _extractive_digest(session, entity)

    assert entity.display_title == "Informative article title"


def test_given_a_poor_heuristic_result_when_an_llm_command_is_configured_then_it_escalates_through_the_fallback_ladder(
    session,
) -> None:
    entity = _entity(session, "raw social text")
    _constituent(session, entity, source="reddit", title="@someone https://example.com/post")
    fallback = FallbackLlmCommand([_FailingLlm(), _StubLlm("A concise generated headline")])
    session.commit()

    _extractive_digest(session, entity, llm_command=fallback)

    assert entity.display_title == "A concise generated headline"


def test_given_no_llm_command_when_the_heuristic_result_is_poor_then_display_title_falls_back_to_the_heuristic_output(
    session,
) -> None:
    entity = _entity(session, "raw social text")
    _constituent(session, entity, source="reddit", title="@someone https://example.com/post")
    session.commit()

    _extractive_digest(session, entity)

    assert entity.display_title == ""


def test_given_the_abstractive_digest_path_when_synthesized_then_the_llm_cannot_override_display_title(
    session,
) -> None:
    entity = _entity(session, "Informative article title")
    _constituent(session, entity, source="article", title="Informative article title")
    response = json.dumps(
        {
            "active_tool_updates": [
                {
                    "title": "Invented LLM title",
                    "source": entity.canonical_url,
                    "sources": ["https://example.com/article/1"],
                    "action": "Read it",
                    "summary": "Summary",
                }
            ],
            "immediate_techniques": [],
            "horizon_developments": [],
        }
    )
    session.commit()

    digest = synthesize_digest(
        session,
        llm_command=_StubLlm(response),
        entity_ids=[entity.id],
        period_start=_START,
        period_end=_END,
    )

    assert digest.content_json["active_tool_updates"][0]["title"] == entity.display_title


def test_given_local_polish_when_applied_then_it_preserves_display_title_exactly(session) -> None:
    entity = _entity(session, "Canonical title")
    entity.display_title = "Locked display title"
    _constituent(session, entity, source="article", title="Canonical title")
    session.commit()
    digest = _extractive_digest(session, entity)
    entry = digest.content_json["horizon_developments"][0]
    response = json.dumps(
        {
            "active_tool_updates": [],
            "immediate_techniques": [],
            "horizon_developments": [
                {
                    **entry,
                    "title": "Changed by local polish",
                    "summary": "Polished summary",
                    "action": "Take action",
                }
            ],
        }
    )

    polished = polish_extractive_digest(
        session,
        digest=digest,
        llm_command=_StubLlm(response),
        model_used="fixture-local",
    )

    assert polished.content_json["horizon_developments"][0]["title"] == entity.display_title
