"""T-18 ACs 5 and 7: linked concepts are never destructively rewritten."""

from __future__ import annotations

from sqlalchemy import select

from rumen.models import ItemConcept, SkosConcept
from rumen.taxonomy.audit.apply import apply_proposal
from rumen.taxonomy.audit.propose import AuditProposal


def _linked_concept(session, concept_id: int, label: str) -> SkosConcept:
    concept = SkosConcept(id=concept_id, facet="topic", pref_label=label, scheme="fixture")
    session.add(concept)
    session.flush()
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=concept_id,
            concept_id=concept_id,
            facet="topic",
            confidence=1.0,
            method="fixture",
        )
    )
    session.commit()
    return concept


def test_apply_when_linked_concept_is_consolidated_then_keeps_opaque_id_and_item_links(
    session,
) -> None:
    source = _linked_concept(session, 200, "Old label")
    target = SkosConcept(id=201, facet="topic", pref_label="Preferred label", scheme="fixture")
    session.add(target)
    session.commit()

    apply_proposal(
        session,
        AuditProposal(
            "consolidate",
            (source.id, target.id),
            "Alias the existing concept.",
            target_concept_id=target.id,
        ),
        audit_run_id="audit-ac5",
    )
    session.commit()

    assert session.get(SkosConcept, source.id) is not None
    assert (
        session.scalar(select(ItemConcept).where(ItemConcept.concept_id == source.id)) is not None
    )


def test_apply_when_prune_targets_linked_concept_then_deprecates_without_deleting(session) -> None:
    concept = _linked_concept(session, 210, "Still linked")

    apply_proposal(
        session,
        AuditProposal("prune", (concept.id,), "Sparse but still referenced."),
        audit_run_id="audit-ac7",
    )
    session.commit()

    stored = session.get(SkosConcept, concept.id)
    assert stored is not None
    assert stored.status == "deprecated"
    assert (
        session.scalar(select(ItemConcept).where(ItemConcept.concept_id == concept.id)) is not None
    )
