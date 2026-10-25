"""Topic label IDs and the deferred weak-supervision API."""

from __future__ import annotations

from collections.abc import Sequence

from rumen.taxonomy.bootstrap.snorkel_label_model import _UNAVAILABLE
from rumen.taxonomy.legacy_crosswalk import TOPIC_CONCEPTS, TopicConcept

ABSTAIN = -1


def topic_label_map(concepts: Sequence[TopicConcept] = TOPIC_CONCEPTS) -> dict[int, int]:
    """Map stable topic concept IDs to contiguous class integers."""
    return {concept.id: index for index, concept in enumerate(concepts)}


def build_topic_labeling_functions(*args, **kwargs):
    """Report that weak supervision is deferred, even if its old dependency is installed."""
    raise RuntimeError(_UNAVAILABLE)
