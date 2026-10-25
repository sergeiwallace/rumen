"""T-18 AC 4: applied maintenance edits round-trip exactly through rollback."""

from __future__ import annotations

from sqlalchemy import select

from rumen.models import ItemConcept, SkosConcept, SkosRelation, TaxonomyRevision
from rumen.taxonomy.audit.apply import apply_proposal
from rumen.taxonomy.audit.propose import AuditProposal
from rumen.taxonomy.audit.rollback import rollback_revision


def test_rollback_when_revision_is_requested_then_restores_exact_pre_edit_state_and_links_rows(
    session,
) -> None:
    source = SkosConcept(id=300, facet="topic", pref_label="Alias", scheme="fixture")
    target = SkosConcept(id=301, facet="topic", pref_label="Canonical", scheme="fixture")
    session.add_all([source, target])
    session.flush()
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=1,
            concept_id=source.id,
            facet="topic",
            confidence=1.0,
            method="fixture",
        )
    )
    session.commit()

    original = apply_proposal(
        session,
        AuditProposal(
            "consolidate",
            (source.id, target.id),
            "Alias duplicate labels.",
            target_concept_id=target.id,
        ),
        audit_run_id="audit-ac4",
    )
    session.commit()
    assert session.scalar(select(SkosRelation).where(SkosRelation.relation_type == "useFor"))

    rollback = rollback_revision(session, original.id, audit_run_id="rollback-ac4")
    session.commit()
    session.expire_all()

    restored_original = session.get(TaxonomyRevision, original.id)
    restored_rollback = session.get(TaxonomyRevision, rollback.id)
    assert restored_rollback is not None
    assert restored_rollback.op_type == "rollback"
    assert restored_rollback.reverts == original.id
    assert restored_rollback.after_json == restored_original.before_json
    assert restored_original.reverted_by == restored_rollback.id
    assert session.get(SkosConcept, source.id).status == "active"
    assert (
        session.scalars(select(SkosRelation).where(SkosRelation.relation_type == "useFor")).all()
        == []
    )
