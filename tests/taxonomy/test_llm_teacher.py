"""T-16 LLM-teacher acceptance coverage."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from sqlalchemy import select

from rumen.models import FeedItem, ItemConcept
from rumen.taxonomy.bootstrap.llm_teacher import (
    LLM_TEACHER_METHOD,
    classify_with_teacher,
)
from rumen.taxonomy.legacy_crosswalk import TOPIC_CONCEPTS


class _FixedLlm:
    def __init__(self, response: str) -> None:
        self.response = response

    def run(self, prompt: str) -> str:
        return self.response


class _UnavailableLlm:
    def run(self, prompt: str) -> str:
        raise subprocess.CalledProcessError(1, ["configured-llm"])


def _item(session, source_id: str = "teacher-item") -> FeedItem:
    item = FeedItem(
        source="arxiv",
        source_id=source_id,
        url=f"https://example.test/{source_id}",
        title="Machine learning research",
        content="A machine learning research paper.",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    session.add(item)
    session.flush()
    return item


def test_llm_teacher_when_valid_topic_response_then_writes_provenance_tagged_candidate(
    session,
) -> None:
    item = _item(session)
    concept = TOPIC_CONCEPTS[1]

    candidate = classify_with_teacher(
        session,
        item,
        _FixedLlm(f'{{"concept_id": {concept.id}, "confidence": 0.91}}'),
    )
    session.commit()

    assert candidate is not None
    stored = session.scalar(select(ItemConcept).where(ItemConcept.id == candidate.id))
    assert stored is not None
    assert (stored.concept_id, stored.facet, stored.method, stored.confidence) == (
        concept.id,
        "topic",
        LLM_TEACHER_METHOD,
        0.91,
    )


def test_llm_teacher_when_command_is_unavailable_then_skips_item_without_raising(session) -> None:
    item = _item(session)

    assert classify_with_teacher(session, item, _UnavailableLlm()) is None
    assert session.scalars(select(ItemConcept)).all() == []


def test_llm_teacher_when_different_human_concept_exists_in_facet_then_skips_whole_facet(
    session,
) -> None:
    item = _item(session)
    human_concept, attempted_concept = TOPIC_CONCEPTS[:2]
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=human_concept.id,
            facet="topic",
            confidence=1.0,
            method="human:cli",
        )
    )
    session.commit()

    candidate = classify_with_teacher(
        session,
        item,
        _FixedLlm(f'{{"concept_id": {attempted_concept.id}, "confidence": 0.91}}'),
    )

    assert candidate is None
    rows = session.scalars(select(ItemConcept).where(ItemConcept.item_id == item.id)).all()
    assert [(row.concept_id, row.method) for row in rows] == [(human_concept.id, "human:cli")]
