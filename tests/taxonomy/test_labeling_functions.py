"""Deferred labeling-function imports remain safe on a base installation."""

from __future__ import annotations

from rumen.taxonomy.bootstrap.labeling_functions import topic_label_map
from rumen.taxonomy.legacy_crosswalk import TOPIC_CONCEPTS


def test_given_topic_concepts_when_mapped_then_ids_remain_contiguous():
    assert topic_label_map() == {concept.id: index for index, concept in enumerate(TOPIC_CONCEPTS)}
