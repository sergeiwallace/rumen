"""Held-out human gold-set evaluation for per-facet HMC classifiers (T-17)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.classifier.tier1_embedding import FacetHMCClassifier
from rumen.models import FeedItem, ItemConcept, TopicEntity

# Only a row explicitly tagged as human-reviewed counts as gold truth.  Everything else —
# T-16 weak-supervision (`bootstrap:*`) AND T-14's legacy-label crosswalk backfill
# (`method="rule"`, itself a mechanical remap of the original noisy machine-generated Flash
# label, not a human judgment) — must be excluded, or "evaluated against gold" collapses back
# into "agrees with its own noisy training signal" (the exact failure this task exists to fix).
HUMAN_GOLD_METHOD_PREFIX = "human:"


@dataclass(frozen=True)
class GoldExample:
    """An explicitly human-labeled, held-out item and its facet concepts."""

    text: str
    concept_ids: frozenset[int]


@dataclass(frozen=True)
class GoldEvaluation:
    facet: str
    precision: float | None
    recall: float | None
    true_positives: int
    false_positives: int
    false_negatives: int
    evaluated_count: int
    evaluated: bool


def evaluate_gold_set(
    classifier: FacetHMCClassifier,
    gold_examples: list[GoldExample],
    *,
    path_consistency_floor: float,
) -> GoldEvaluation:
    """Measure HMC assignments only against held-out human labels, never bootstrap rows."""
    if not gold_examples:
        return GoldEvaluation(classifier.facet, None, None, 0, 0, 0, 0, False)
    true_positives = false_positives = false_negatives = 0
    for example in gold_examples:
        predicted = set(
            classifier.assign(example.text, path_consistency_floor=path_consistency_floor)
        )
        true_positives += len(predicted & example.concept_ids)
        false_positives += len(predicted - example.concept_ids)
        false_negatives += len(example.concept_ids - predicted)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    return GoldEvaluation(
        classifier.facet,
        precision,
        recall,
        true_positives,
        false_positives,
        false_negatives,
        len(gold_examples),
        True,
    )


def load_human_gold_examples(session: Session, *, facet: str) -> list[GoldExample]:
    """Load explicitly human-reviewed ``item_concepts`` rows as gold labels for a facet.

    Deliberately an allow-list (``method`` startswith ``"human:"``), not a negation of
    ``bootstrap:``: T-14's legacy-label crosswalk backfill writes ``method="rule"`` rows that
    are a mechanical remap of the original noisy Flash label, not a human judgment, and must
    not silently count as gold truth.
    """
    labels = session.scalars(
        select(ItemConcept).where(
            ItemConcept.facet == facet,
            ItemConcept.method.startswith(HUMAN_GOLD_METHOD_PREFIX),
        )
    ).all()
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    for label in labels:
        grouped[(label.item_table, label.item_id)].add(label.concept_id)
    examples: list[GoldExample] = []
    for (item_table, item_id), concept_ids in grouped.items():
        item = _load_item(session, item_table, item_id)
        if item is not None and (text := _item_text(item)):
            examples.append(GoldExample(text, frozenset(concept_ids)))
    return examples


def _load_item(session: Session, item_table: str, item_id: int) -> FeedItem | TopicEntity | None:
    if item_table == "feed_items":
        return session.get(FeedItem, item_id)
    if item_table == "topic_entities":
        return session.get(TopicEntity, item_id)
    return None


def _item_text(item: FeedItem | TopicEntity) -> str:
    if isinstance(item, FeedItem):
        return f"{item.title or ''}\n{item.content or ''}".strip()
    return item.canonical_title.strip()
