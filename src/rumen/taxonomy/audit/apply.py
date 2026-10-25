"""Non-destructive application and state snapshots for D-23 audit edits."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.models import ItemConcept, SkosConcept, SkosRelation, TaxonomyRevision
from rumen.taxonomy.audit.propose import AuditProposal
from rumen.taxonomy.materialize import stable_concept_id

_KC = "https://rumen.dev/ns#"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def snapshot_state(session: Session, concept_ids: set[int] | list[int] | tuple[int, ...]) -> dict:
    """Serialize concepts and all touching relations into JSON-safe rollback state."""
    ids = sorted(set(concept_ids))
    if not ids:
        return {"concepts": [], "relations": []}
    concepts = session.scalars(select(SkosConcept).where(SkosConcept.id.in_(ids))).all()
    relations = session.scalars(
        select(SkosRelation).where(
            SkosRelation.source_concept_id.in_(ids) | SkosRelation.target_concept_id.in_(ids)
        )
    ).all()
    return {
        "concepts": sorted(
            (_concept_data(concept) for concept in concepts), key=lambda row: row["id"]
        ),
        "relations": sorted(
            (_relation_data(relation) for relation in relations), key=lambda row: row["id"]
        ),
    }


def apply_proposal(
    session: Session,
    proposal: AuditProposal,
    *,
    audit_run_id: str,
    actor: str = "system:taxonomy-audit",
) -> TaxonomyRevision:
    """Apply one validated proposal and write exactly one reversible audit revision."""
    affected_ids = set(proposal.concept_ids)
    before = snapshot_state(session, affected_ids)
    concepts = {
        concept.id: concept
        for concept in session.scalars(select(SkosConcept).where(SkosConcept.id.in_(affected_ids)))
    }
    if set(concepts) != affected_ids:
        raise ValueError("taxonomy audit proposal refers to a missing concept")

    if proposal.op_type == "deepen":
        affected_ids.update(_apply_deepen(session, concepts, proposal))
    elif proposal.op_type == "consolidate":
        _apply_consolidate(session, concepts, proposal)
    elif proposal.op_type == "relabel":
        _apply_relabel(session, concepts, proposal)
    elif proposal.op_type == "prune":
        _apply_prune(session, concepts)
    elif proposal.op_type == "add":
        affected_ids.update(_apply_deepen(session, concepts, proposal))
    else:  # validation keeps this unreachable; retain a guard at the mutation boundary.
        raise ValueError(f"unsupported taxonomy audit operation: {proposal.op_type}")

    session.flush()
    after = snapshot_state(session, affected_ids)
    revision = TaxonomyRevision(
        audit_run_id=audit_run_id,
        op_type=proposal.op_type,
        concept_ids=sorted(affected_ids),
        before_json=before,
        after_json=after,
        actor=actor,
        rationale=proposal.rationale,
        reversible=1,
    )
    session.add(revision)
    session.flush()
    return revision


def restore_state(session: Session, state: dict[str, Any], *, scope_ids: set[int]) -> None:
    """Restore a snapshot exactly, refusing any deletion that would break item links."""
    target_concepts = {row["id"]: row for row in state.get("concepts", [])}
    target_ids = set(target_concepts)
    existing = {
        concept.id: concept
        for concept in session.scalars(select(SkosConcept).where(SkosConcept.id.in_(scope_ids)))
    }
    obsolete_ids = set(existing) - target_ids
    linked_obsolete = (
        session.scalar(
            select(ItemConcept.id).where(ItemConcept.concept_id.in_(obsolete_ids)).limit(1)
        )
        if obsolete_ids
        else None
    )
    if linked_obsolete is not None:
        raise ValueError("rollback refuses to delete a concept with item_concepts links")

    # Relations are not link targets and can be restored exactly.  Remove every relation
    # touching this operation's scope before recreating the captured pre-edit relations.
    for relation in session.scalars(
        select(SkosRelation).where(
            SkosRelation.source_concept_id.in_(scope_ids)
            | SkosRelation.target_concept_id.in_(scope_ids)
        )
    ):
        session.delete(relation)
    session.flush()

    for concept_id in obsolete_ids:
        session.delete(existing[concept_id])
    for concept_id, row in target_concepts.items():
        concept = existing.get(concept_id)
        if concept is None:
            session.add(SkosConcept(**_concept_values(row)))
        else:
            for key, value in _concept_values(row).items():
                if key != "id":
                    setattr(concept, key, value)
    session.flush()

    for row in state.get("relations", []):
        session.add(SkosRelation(**_relation_values(row)))
    session.flush()


def _apply_deepen(
    session: Session, concepts: dict[int, SkosConcept], proposal: AuditProposal
) -> set[int]:
    if len(proposal.concept_ids) != 1:
        raise ValueError("deepen needs exactly one parent concept")
    parent = concepts[proposal.concept_ids[0]]
    created_ids: set[int] = set()
    for child in proposal.children:
        slug = _slug(child.get("slug") or child["pref_label"])
        source_uri = f"{_KC}audit/{parent.facet}/{parent.id}/{slug}"
        concept_id = stable_concept_id(source_uri)
        existing = session.get(SkosConcept, concept_id)
        if existing is None:
            session.add(
                SkosConcept(
                    id=concept_id,
                    facet=parent.facet,
                    pref_label=child["pref_label"],
                    scheme=parent.scheme,
                    scheme_version=parent.scheme_version,
                    source_uri=source_uri,
                    status="active",
                )
            )
        elif existing.source_uri != source_uri:
            raise ValueError("stable audit concept id collision")
        relation = session.scalar(
            select(SkosRelation).where(
                SkosRelation.source_concept_id == concept_id,
                SkosRelation.target_concept_id == parent.id,
                SkosRelation.relation_type == "broader",
            )
        )
        if relation is None:
            session.add(
                SkosRelation(
                    source_concept_id=concept_id,
                    target_concept_id=parent.id,
                    relation_type="broader",
                )
            )
        created_ids.add(concept_id)
    return created_ids


def _apply_consolidate(
    session: Session, concepts: dict[int, SkosConcept], proposal: AuditProposal
) -> None:
    target_id = proposal.target_concept_id
    assert target_id is not None  # proposal validation enforces this
    for concept_id in proposal.concept_ids:
        if concept_id == target_id:
            continue
        alias = session.scalar(
            select(SkosRelation).where(
                SkosRelation.source_concept_id == concept_id,
                SkosRelation.target_concept_id == target_id,
                SkosRelation.relation_type == "useFor",
            )
        )
        if alias is None:
            session.add(
                SkosRelation(
                    source_concept_id=concept_id,
                    target_concept_id=target_id,
                    relation_type="useFor",
                )
            )
        # Keep both opaque concept ids and their item assignments alive.  Marking the
        # alias deprecated means future classifiers can prefer the target without data loss.
        concepts[concept_id].status = "deprecated"


def _apply_relabel(
    session: Session, concepts: dict[int, SkosConcept], proposal: AuditProposal
) -> None:
    concept = concepts[proposal.concept_ids[0]]
    assert proposal.pref_label is not None  # proposal validation enforces this
    concept.pref_label = proposal.pref_label
    if proposal.target_concept_id is not None and proposal.target_concept_id != concept.id:
        if proposal.target_concept_id not in concepts:
            raise ValueError("relabel alias target must be an affected concept")
        alias = session.scalar(
            select(SkosRelation).where(
                SkosRelation.source_concept_id == concept.id,
                SkosRelation.target_concept_id == proposal.target_concept_id,
                SkosRelation.relation_type == "useFor",
            )
        )
        if alias is None:
            session.add(
                SkosRelation(
                    source_concept_id=concept.id,
                    target_concept_id=proposal.target_concept_id,
                    relation_type="useFor",
                )
            )


def _apply_prune(session: Session, concepts: dict[int, SkosConcept]) -> None:
    # D-23's non-destructive bias applies whether or not a concept happens to have
    # assignments today.  In particular, this is never a DELETE for linked concepts.
    for concept in concepts.values():
        if session.scalar(
            select(ItemConcept.id).where(ItemConcept.concept_id == concept.id).limit(1)
        ):
            concept.status = "deprecated"
        else:
            concept.status = "deprecated"


def _concept_data(concept: SkosConcept) -> dict[str, Any]:
    return {
        "id": concept.id,
        "facet": concept.facet,
        "pref_label": concept.pref_label,
        "notation": concept.notation,
        "scheme": concept.scheme,
        "scheme_version": concept.scheme_version,
        "source_uri": concept.source_uri,
        "status": concept.status,
        "created_at": concept.created_at.isoformat(),
        "updated_at": concept.updated_at.isoformat(),
    }


def _relation_data(relation: SkosRelation) -> dict[str, Any]:
    return {
        "id": relation.id,
        "source_concept_id": relation.source_concept_id,
        "target_concept_id": relation.target_concept_id,
        "relation_type": relation.relation_type,
        "source_scheme": relation.source_scheme,
        "source_version": relation.source_version,
        "target_scheme": relation.target_scheme,
        "target_version": relation.target_version,
        "reviewer": relation.reviewer,
        "review_date": relation.review_date.isoformat() if relation.review_date else None,
        "created_at": relation.created_at.isoformat(),
    }


def _concept_values(row: dict[str, Any]) -> dict[str, Any]:
    values = dict(row)
    values["created_at"] = _datetime(values["created_at"])
    values["updated_at"] = _datetime(values["updated_at"])
    return values


def _relation_values(row: dict[str, Any]) -> dict[str, Any]:
    values = dict(row)
    values["created_at"] = _datetime(values["created_at"])
    if values["review_date"] is not None:
        values["review_date"] = datetime.fromisoformat(values["review_date"]).date()
    return values


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    if not slug:
        raise ValueError("audit child label cannot produce an empty URI slug")
    return slug
