"""Revision-id rollback for the D-23 taxonomy audit."""

from __future__ import annotations

from sqlalchemy.orm import Session

from rumen.models import TaxonomyRevision
from rumen.taxonomy.audit.apply import restore_state, snapshot_state


def rollback_revision(
    session: Session,
    revision_id: int,
    *,
    audit_run_id: str,
    actor: str = "system:taxonomy-audit",
) -> TaxonomyRevision:
    """Restore an edit's exact pre-state and record the rollback as a new revision."""
    original = session.get(TaxonomyRevision, revision_id)
    if original is None:
        raise ValueError(f"taxonomy revision {revision_id} does not exist")
    if not original.reversible or original.before_json is None:
        raise ValueError(f"taxonomy revision {revision_id} is not reversible")
    if original.reverted_by is not None:
        raise ValueError(f"taxonomy revision {revision_id} was already rolled back")
    scope_ids = (
        set(original.concept_ids)
        | _state_ids(original.before_json)
        | _state_ids(original.after_json or {})
    )
    before = snapshot_state(session, scope_ids)
    restore_state(session, original.before_json, scope_ids=scope_ids)
    after = snapshot_state(session, scope_ids)
    rollback = TaxonomyRevision(
        audit_run_id=audit_run_id,
        op_type="rollback",
        concept_ids=sorted(scope_ids),
        before_json=before,
        after_json=after,
        actor=actor,
        rationale=f"Rollback of taxonomy revision {original.id}: {original.rationale}",
        reversible=1,
        reverts=original.id,
    )
    session.add(rollback)
    session.flush()
    original.reverted_by = rollback.id
    session.flush()
    return rollback


def _state_ids(state: dict) -> set[int]:
    return {row["id"] for row in state.get("concepts", [])}
