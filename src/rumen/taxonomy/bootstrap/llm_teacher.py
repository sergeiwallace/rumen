"""LLM-teacher candidate labels for the topic facet (T-16).

The teacher is deliberately a caller-supplied :class:`~rumen.llm.LlmCommand`.
It never opens a hosted API connection itself, and an unavailable configured command leaves
that item unlabeled for a later labeling pass.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.llm import LlmCommand, LlmUnavailable, parse_json_response
from rumen.models import FeedItem, ItemConcept, TopicEntity
from rumen.taxonomy.legacy_crosswalk import TOPIC_CONCEPTS, TopicConcept

logger = logging.getLogger(__name__)

LLM_TEACHER_METHOD = "bootstrap:llm-teacher"

_PROMPT_TEMPLATE = """You are a careful taxonomy teacher. Assign this item exactly one topic
concept from the supplied list. Respond with ONLY a JSON object:
{{"concept_id": <integer>, "confidence": <0.0-1.0>}}.

Topic concepts: {concepts}
Item source: {source}
Item title: {title}
Item content: {content}
"""


def build_prompt(item: FeedItem | TopicEntity, concepts: Sequence[TopicConcept]) -> str:
    """Build the constrained topic-teacher prompt for one persisted item."""
    if isinstance(item, FeedItem):
        source = item.source
        title = item.title
        content = item.content or ""
    else:
        source = "topic-entity"
        title = item.canonical_title
        content = ""
    concept_list = [{"id": concept.id, "label": concept.pref_label} for concept in concepts]
    return _PROMPT_TEMPLATE.format(
        concepts=json.dumps(concept_list), source=source, title=title, content=content[:2000]
    )


def item_target(item: FeedItem | TopicEntity) -> tuple[str, int]:
    """Return the polymorphic ``item_concepts`` target for a persisted model instance."""
    if item.id is None:
        raise ValueError("bootstrap labels require a persisted item")
    if isinstance(item, FeedItem):
        return "feed_items", item.id
    if isinstance(item, TopicEntity):
        return "topic_entities", item.id
    raise TypeError(f"unsupported bootstrap item type: {type(item)!r}")


def write_bootstrap_label(
    session: Session,
    item: FeedItem | TopicEntity,
    *,
    concept_id: int,
    facet: str,
    confidence: float,
    method: str,
) -> ItemConcept | None:
    """Write or refresh one bootstrap candidate without replacing a gold assignment.

    ``item_concepts`` intentionally has one row per item/concept rather than a parallel
    candidate table.  Bootstrap provenance is therefore encoded in ``method``.  A pre-existing
    non-bootstrap row is retained as the authoritative human/gold label.
    """
    item_table, item_id = item_target(item)
    human_assignment = session.scalar(
        select(ItemConcept.id)
        .where(
            ItemConcept.item_table == item_table,
            ItemConcept.item_id == item_id,
            ItemConcept.facet == facet,
            ItemConcept.method.startswith("human:"),
        )
        .limit(1)
    )
    if human_assignment is not None:
        logger.info("Keeping human-reviewed facet for %s/%s", item_table, item_id)
        return None
    existing = session.scalar(
        select(ItemConcept).where(
            ItemConcept.item_table == item_table,
            ItemConcept.item_id == item_id,
            ItemConcept.concept_id == concept_id,
        )
    )
    if existing is not None:
        if not existing.method.startswith("bootstrap:"):
            logger.info(
                "Keeping non-bootstrap label for %s/%s concept %s", item_table, item_id, concept_id
            )
            return None
        existing.facet = facet
        existing.confidence = confidence
        existing.method = method
        session.flush()
        return existing

    assignment = ItemConcept(
        item_table=item_table,
        item_id=item_id,
        concept_id=concept_id,
        facet=facet,
        confidence=confidence,
        method=method,
    )
    session.add(assignment)
    session.flush()
    return assignment


def classify_with_teacher(
    session: Session,
    item: FeedItem | TopicEntity,
    llm_command: LlmCommand,
    *,
    concepts: Sequence[TopicConcept] = TOPIC_CONCEPTS,
    facet: str = "topic",
) -> ItemConcept | None:
    """Ask the configured LLM teacher and persist its provenance-tagged candidate label."""
    try:
        raw = llm_command.run(build_prompt(item, concepts))
    except (subprocess.SubprocessError, OSError, LlmUnavailable) as exc:
        logger.warning("LLM teacher unavailable (%s) — skipping bootstrap item", exc)
        return None
    try:
        parsed = parse_json_response(raw)
        concept_id = int(parsed["concept_id"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "LLM teacher returned an invalid candidate (%s) — skipping bootstrap item", exc
        )
        return None
    if concept_id not in {concept.id for concept in concepts}:
        logger.warning("LLM teacher selected concept %s outside the topic list", concept_id)
        return None
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (AttributeError, TypeError, ValueError):
        confidence = 0.5
    return write_bootstrap_label(
        session,
        item,
        concept_id=concept_id,
        facet=facet,
        confidence=max(0.0, min(1.0, confidence)),
        method=LLM_TEACHER_METHOD,
    )


def run_teacher_pass(
    session: Session,
    items: Iterable[FeedItem | TopicEntity],
    llm_command: LlmCommand | None,
    *,
    concepts: Sequence[TopicConcept] = TOPIC_CONCEPTS,
    facet: str = "topic",
) -> list[ItemConcept]:
    """Classify every item possible; an unavailable command skips rather than aborts the pass."""
    if llm_command is None:
        return []
    assignments: list[ItemConcept] = []
    for item in items:
        assignment = classify_with_teacher(
            session, item, llm_command, concepts=concepts, facet=facet
        )
        if assignment is not None:
            assignments.append(assignment)
    return assignments
