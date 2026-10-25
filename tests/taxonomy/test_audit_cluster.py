"""T-18 ACs 2 and 3: crowded leaves and embedding duplicate clustering."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
from sqlalchemy import select

from rumen.models import ItemConcept, SkosConcept, SkosRelation, TaxonomyRevision
from rumen.taxonomy.audit import run_taxonomy_audit


class _AuditLlm:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = iter(responses)

    def run(self, _prompt: str) -> str:
        return json.dumps(next(self.responses))


class _DuplicateProvider:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            key = "duplicate" if text in {"Duplicate A", "Duplicate B"} else text
            vectors[index, int.from_bytes(sha256(key.encode()).digest()[:4], "big") % 1024] = 1.0
        return vectors


def _assignment(session, concept_id: int, item_id: int) -> None:
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item_id,
            concept_id=concept_id,
            facet="topic",
            confidence=1.0,
            method="fixture",
        )
    )


def test_audit_when_edit_is_applied_then_writes_one_complete_taxonomy_revision(session) -> None:
    parent = SkosConcept(id=100, facet="topic", pref_label="Crowded", scheme="fixture")
    session.add(parent)
    session.flush()
    _assignment(session, parent.id, 1)
    _assignment(session, parent.id, 2)
    session.commit()

    result = run_taxonomy_audit(
        session,
        llm_command=_AuditLlm(
            [
                {
                    "op_type": "deepen",
                    "concept_ids": [100],
                    "rationale": "Split the overloaded leaf.",
                    "children": [{"pref_label": "Crowded child"}],
                }
            ]
        ),
        provider=_DuplicateProvider(),
        crowded_item_threshold=1,
        sparse_item_threshold=-1,
        audit_run_id="audit-ac1",
    )

    revisions = session.scalars(select(TaxonomyRevision)).all()
    assert result.applied_revision_ids == [revisions[0].id]
    assert len(revisions) == 1
    revision = revisions[0]
    assert revision.audit_run_id == "audit-ac1"
    assert revision.op_type == "deepen"
    assert revision.concept_ids[0] == parent.id
    assert revision.before_json and revision.after_json
    assert revision.actor == "system:taxonomy-audit"
    assert revision.rationale


def test_audit_when_leaf_is_crowded_then_applies_deepen_with_stable_child_ids(session) -> None:
    parent = SkosConcept(id=110, facet="topic", pref_label="Crowded", scheme="fixture")
    session.add(parent)
    session.flush()
    _assignment(session, parent.id, 11)
    _assignment(session, parent.id, 12)
    session.commit()

    run_taxonomy_audit(
        session,
        llm_command=_AuditLlm(
            [
                {
                    "op_type": "deepen",
                    "concept_ids": [110],
                    "rationale": "Add a narrower child.",
                    "children": [{"pref_label": "Focused child", "slug": "focused-child"}],
                }
            ]
        ),
        provider=_DuplicateProvider(),
        crowded_item_threshold=1,
        sparse_item_threshold=-1,
    )

    child = session.scalar(select(SkosConcept).where(SkosConcept.pref_label == "Focused child"))
    assert child is not None
    assert child.source_uri == "https://rumen.dev/ns#audit/topic/110/focused-child"
    assert (
        session.scalar(
            select(SkosRelation).where(
                SkosRelation.source_concept_id == child.id,
                SkosRelation.target_concept_id == parent.id,
                SkosRelation.relation_type == "broader",
            )
        )
        is not None
    )


def test_audit_when_near_duplicates_then_consolidates_via_usefor_without_deleting(session) -> None:
    first = SkosConcept(id=120, facet="topic", pref_label="Duplicate A", scheme="fixture")
    second = SkosConcept(id=121, facet="topic", pref_label="Duplicate B", scheme="fixture")
    session.add_all([first, second])
    session.flush()
    _assignment(session, first.id, 21)
    _assignment(session, second.id, 22)
    session.commit()

    run_taxonomy_audit(
        session,
        llm_command=_AuditLlm(
            [
                {
                    "op_type": "consolidate",
                    "concept_ids": [120, 121],
                    "target_concept_id": 121,
                    "rationale": "The labels are semantic duplicates.",
                }
            ]
        ),
        provider=_DuplicateProvider(),
        crowded_item_threshold=100,
        sparse_item_threshold=-1,
        duplicate_similarity_threshold=0.99,
    )

    assert session.get(SkosConcept, first.id) is not None
    assert session.get(SkosConcept, second.id) is not None
    assert (
        session.scalar(
            select(SkosRelation).where(
                SkosRelation.source_concept_id == first.id,
                SkosRelation.target_concept_id == second.id,
                SkosRelation.relation_type == "useFor",
            )
        )
        is not None
    )
