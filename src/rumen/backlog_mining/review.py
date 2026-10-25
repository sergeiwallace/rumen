"""Read-only reviewer presentation and explicit candidate dispositions."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.backlog_mining.repository import (
    CandidateNotFoundError,
    CandidateRepository,
    CandidateRepositoryError,
    CandidateTaskLinkRepository,
)
from rumen.digest import append_backlog_candidate_review_references
from rumen.models import (
    BacklogCandidate,
    BacklogCandidateDisposition,
    BacklogCandidateEvidence,
    BacklogCandidateRecommendation,
)


class CandidateRenderError(ValueError):
    """A candidate cannot be emitted as a reviewer-ready draft."""


@dataclass(frozen=True, slots=True)
class RenderedCandidate:
    """The only candidate handoff payload exposed by the review surface."""

    candidate_id: int
    fingerprint: str
    title: str
    body: str
    evidence_links: tuple[str, ...]
    template_revision: str | None


def candidate_for_identifier(session: Session, identifier: str) -> BacklogCandidate:
    """Resolve a stable candidate ID first, then a fingerprint family identifier."""
    candidate = None
    if identifier.isdecimal():
        candidate = session.get(BacklogCandidate, int(identifier))
    if candidate is None:
        candidate = session.scalar(
            select(BacklogCandidate)
            .where(BacklogCandidate.fingerprint == identifier)
            .order_by(BacklogCandidate.id.desc())
        )
    if candidate is None:
        raise CandidateNotFoundError(f"candidate {identifier!r} does not exist")
    return candidate


def candidate_review(session: Session, identifier: str) -> dict[str, Any]:
    """Return separately-labelled reviewer evidence for one candidate family member."""
    candidate = candidate_for_identifier(session, identifier)
    evidence_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_rows = session.scalars(
        select(BacklogCandidateEvidence)
        .where(BacklogCandidateEvidence.candidate_id == candidate.id)
        .order_by(BacklogCandidateEvidence.created_at, BacklogCandidateEvidence.id)
    )
    for evidence in evidence_rows:
        evidence_by_kind[evidence.kind].append(_evidence_dict(evidence))
    links = CandidateTaskLinkRepository(session).links_for_candidate(candidate.id)
    history = session.scalars(
        select(BacklogCandidateDisposition)
        .where(BacklogCandidateDisposition.fingerprint == candidate.fingerprint)
        .order_by(BacklogCandidateDisposition.sequence_no, BacklogCandidateDisposition.id)
    )
    recommendations = session.scalars(
        select(BacklogCandidateRecommendation)
        .where(BacklogCandidateRecommendation.candidate_id == candidate.id)
        .order_by(BacklogCandidateRecommendation.id)
    )
    cross_candidate_evidence = []
    if candidate.topic_entity_id is not None:
        cross_candidate_evidence = [
            {
                "candidate_id": group.candidate.id,
                "fingerprint": group.candidate.fingerprint,
                "evidence": [_evidence_dict(evidence) for evidence in group.evidence],
            }
            for group in CandidateRepository(session).cross_candidate_evidence_for_entity(
                candidate.topic_entity_id
            )
        ]
    return {
        "candidate": {
            "id": candidate.id,
            "fingerprint": candidate.fingerprint,
            "state": candidate.state,
            "content_snapshot": candidate.content_snapshot,
            "assessment": candidate.assessment,
        },
        "source_claims": {
            kind: rows
            for kind, rows in evidence_by_kind.items()
            if kind in {"source", "source_claim"}
        },
        "internal_context": {
            "configuration": candidate.configuration_snapshot,
            "source_snapshot": candidate.source_snapshot,
            "evidence": evidence_by_kind.get("internal_context", []),
            "cross_candidate_evidence": cross_candidate_evidence,
        },
        "model_inferences": {
            kind: rows for kind, rows in evidence_by_kind.items() if kind == "inference"
        },
        "task_matches": [_task_link_dict(link) for link in links],
        "disposition_history": [_disposition_dict(frame) for frame in history],
        "recommendations": [_recommendation_dict(row, candidate) for row in recommendations],
    }


def review_candidate(
    session: Session,
    identifier: str,
    *,
    action: str,
    actor: str,
    rationale: str,
    payload: dict[str, Any] | None = None,
):
    """Record one selected lifecycle disposition through the existing repository."""
    candidate = candidate_for_identifier(session, identifier)
    return CandidateRepository(session).transition(
        candidate.id,
        action=action,
        actor=actor,
        rationale=rationale,
        payload=payload,
    )


def render_candidate(session: Session, candidate_id: int) -> RenderedCandidate:
    """Render only approved candidate drafts, without changing queue state."""
    candidate = session.get(BacklogCandidate, candidate_id)
    if candidate is None:
        raise CandidateNotFoundError(f"candidate {candidate_id} does not exist")
    if candidate.state not in {"accepted", "edited"}:
        raise CandidateRenderError("only accepted or edited candidates can be rendered")
    title = candidate.content_snapshot.get("title")
    if not isinstance(title, str) or not title.strip():
        raise CandidateRenderError("candidate render payload requires a non-empty title")
    evidence_links = _evidence_links(session, candidate)
    if not evidence_links:
        raise CandidateRenderError("candidate render payload requires evidence links")
    body = "\n\n".join(
        (
            title.strip(),
            "## Candidate draft",
            json.dumps(candidate.content_snapshot, indent=2, sort_keys=True),
            "## Assessment",
            json.dumps(candidate.assessment, indent=2, sort_keys=True),
        )
    )
    template_revision = candidate.configuration_snapshot.get("template_revision")
    return RenderedCandidate(
        candidate_id=candidate.id,
        fingerprint=candidate.fingerprint,
        title=title.strip(),
        body=body,
        evidence_links=tuple(evidence_links),
        template_revision=template_revision if isinstance(template_revision, str) else None,
    )


def append_candidate_review_references(markdown: str, candidates: list[dict[str, Any]]) -> str:
    """Append read-only review commands to a stored digest's Markdown."""
    return append_backlog_candidate_review_references(markdown, candidates)


def _evidence_links(session: Session, candidate: BacklogCandidate) -> list[str]:
    links: list[str] = []
    for evidence in session.scalars(
        select(BacklogCandidateEvidence)
        .where(BacklogCandidateEvidence.candidate_id == candidate.id)
        .order_by(BacklogCandidateEvidence.id)
    ):
        metadata = evidence.metadata_json or {}
        url = metadata.get("url")
        if isinstance(url, str) and url:
            links.append(url)
    for claim in candidate.source_snapshot.get("source_claims", []):
        if isinstance(claim, dict) and isinstance(claim.get("url"), str) and claim["url"]:
            links.append(claim["url"])
    return list(dict.fromkeys(links))


def _evidence_dict(evidence: BacklogCandidateEvidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "feed_item_id": evidence.feed_item_id,
        "excerpt": evidence.excerpt,
        "metadata": evidence.metadata_json,
    }


def _task_link_dict(link) -> dict[str, Any]:
    return {
        "task_store": link.task_store,
        "external_task_key": link.external_task_key,
        "title": link.last_observed_title,
        "state": link.last_observed_state,
        "closure_reason": link.last_observed_closure_reason,
        "closure_class": link.closure_class,
    }


def _disposition_dict(frame: BacklogCandidateDisposition) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "action": frame.action,
        "actor": frame.actor,
        "rationale": frame.rationale,
        "created_at": frame.created_at.isoformat(),
    }


def _recommendation_dict(
    row: BacklogCandidateRecommendation, candidate: BacklogCandidate
) -> dict[str, Any]:
    return {
        "recommended_repo": candidate.configuration_snapshot.get("beads_repo_identity"),
        "task_store": row.task_store,
        "recommended_cc_task": bool(row.recommended_cc_task),
        "rationale": row.rationale,
        "confidence": row.confidence,
    }


__all__ = [
    "CandidateRenderError",
    "CandidateRepositoryError",
    "RenderedCandidate",
    "append_candidate_review_references",
    "candidate_for_identifier",
    "candidate_review",
    "render_candidate",
    "review_candidate",
]
