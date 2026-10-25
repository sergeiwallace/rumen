"""Scheduled, self-healing taxonomy maintenance audit (D-23)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from rumen.embeddings import EmbeddingProvider
from rumen.llm import LlmCommand
from rumen.taxonomy.audit.apply import apply_proposal
from rumen.taxonomy.audit.cluster import cluster_concepts
from rumen.taxonomy.audit.detect import AuditIssue, detect_issues
from rumen.taxonomy.audit.propose import propose_edit

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TaxonomyAuditResult:
    """Counts and audit identity for one independently schedulable audit run."""

    audit_run_id: str
    detected: int = 0
    applied_revision_ids: list[int] = field(default_factory=list)
    skipped: int = 0

    @property
    def exit_code(self) -> int:
        return 0


def run_taxonomy_audit(
    session: Session,
    *,
    llm_command: LlmCommand | None,
    provider: EmbeddingProvider | None = None,
    crowded_item_threshold: int = 100,
    sparse_item_threshold: int = 0,
    duplicate_similarity_threshold: float = 0.92,
    audit_run_id: str | None = None,
) -> TaxonomyAuditResult:
    """Cluster → detect → propose → apply, skipping each failed proposal independently."""
    run_id = audit_run_id or str(uuid.uuid4())
    result = TaxonomyAuditResult(run_id)
    _clusters, pairs = cluster_concepts(
        session, provider=provider, similarity_threshold=duplicate_similarity_threshold
    )
    issues = detect_issues(
        session,
        similar_pairs=pairs,
        crowded_item_threshold=crowded_item_threshold,
        sparse_item_threshold=sparse_item_threshold,
    )
    result.detected = len(issues)
    for issue in issues:
        proposal = propose_edit(issue, llm_command)
        if proposal is None:
            result.skipped += 1
            continue
        try:
            revision = apply_proposal(session, proposal, audit_run_id=run_id)
            session.commit()  # Incremental durability: one bad later edit cannot lose prior edits.
        except Exception as exc:  # noqa: BLE001 - a later detected issue must still be examined.
            session.rollback()
            result.skipped += 1
            logger.warning(
                "Taxonomy audit edit failed; skipping issue (%s): %s", issue.rationale, exc
            )
            continue
        result.applied_revision_ids.append(revision.id)
    return result


__all__ = ["AuditIssue", "TaxonomyAuditResult", "run_taxonomy_audit"]
