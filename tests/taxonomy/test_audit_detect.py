"""T-18 AC 6: invalid LLM output skips one issue without aborting its audit batch."""

from __future__ import annotations

import json
import logging
from hashlib import sha256

import numpy as np
from sqlalchemy import update

from rumen.models import ItemConcept, SkosConcept
from rumen.taxonomy.audit import run_taxonomy_audit


class _ThreeIssueProvider:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            key = "duplicate" if text in {"Duplicate A", "Duplicate B"} else text
            vectors[index, int.from_bytes(sha256(key.encode()).digest()[:4], "big") % 1024] = 1.0
        return vectors


class _ThreeIssueLlm:
    def __init__(self) -> None:
        self.responses = iter(
            [
                {
                    "op_type": "deepen",
                    "concept_ids": [400],
                    "rationale": "Split the crowded leaf.",
                    "children": [{"pref_label": "New child"}],
                },
                "not valid JSON",
                {
                    "op_type": "consolidate",
                    "concept_ids": [402, 403],
                    "target_concept_id": 403,
                    "rationale": "Alias duplicates.",
                },
            ]
        )

    def run(self, _prompt: str) -> str:
        response = next(self.responses)
        return response if isinstance(response, str) else json.dumps(response)


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


def test_audit_when_middle_llm_proposal_is_malformed_then_logs_and_continues(
    caplog, session
) -> None:
    # Migrations seed the legacy topic spine; keep this fixture's issue count isolated.
    session.execute(update(SkosConcept).values(status="deprecated"))
    session.add_all(
        [
            SkosConcept(id=400, facet="topic", pref_label="Crowded", scheme="fixture"),
            SkosConcept(id=401, facet="topic", pref_label="Dead", scheme="fixture"),
            SkosConcept(id=402, facet="topic", pref_label="Duplicate A", scheme="fixture"),
            SkosConcept(id=403, facet="topic", pref_label="Duplicate B", scheme="fixture"),
        ]
    )
    session.flush()
    _assignment(session, 400, 1)
    _assignment(session, 400, 2)
    _assignment(session, 402, 3)
    _assignment(session, 403, 4)
    session.commit()

    with caplog.at_level(logging.WARNING):
        result = run_taxonomy_audit(
            session,
            llm_command=_ThreeIssueLlm(),
            provider=_ThreeIssueProvider(),
            crowded_item_threshold=1,
            sparse_item_threshold=0,
            duplicate_similarity_threshold=0.99,
        )

    assert result.detected == 3
    assert len(result.applied_revision_ids) == 2
    assert result.skipped == 1
    assert "Invalid taxonomy-audit proposal" in caplog.text
