"""Session-scoped draft projection with no task-store capability."""

from __future__ import annotations

from typing import Any

from rumen.backlog_mining.review import RenderedCandidate
from rumen.models import BacklogCandidate


class CcTaskCreationDisabledError(ValueError):
    """The explicit per-installation projection gate is not enabled."""


class NonCurrentSessionError(ValueError):
    """A projection may only describe work for the invoking session."""


class CandidateNotApprovedForProjectionError(ValueError):
    """Only an accepted or edited candidate can be projected."""


def build_cc_task_plan(
    candidate: BacklogCandidate,
    draft: RenderedCandidate,
    *,
    enabled: bool,
    requested_session: str,
    invoking_session: str,
) -> dict[str, Any]:
    """Return one session-local creation plan; execution belongs to the caller."""
    if not enabled:
        raise CcTaskCreationDisabledError("CC-task creation is disabled by configuration")
    if not invoking_session or requested_session != invoking_session:
        raise NonCurrentSessionError("CC-task plans may target only the current session")
    if candidate.state not in {"accepted", "edited"}:
        raise CandidateNotApprovedForProjectionError(
            "only accepted or edited candidates can be projected"
        )
    if candidate.id != draft.candidate_id:
        raise CandidateNotApprovedForProjectionError(
            "draft does not belong to the selected candidate"
        )
    return {
        "candidate_id": candidate.id,
        "fingerprint": candidate.fingerprint,
        "session": invoking_session,
        "title": draft.title,
        "body": draft.body,
        "template_revision": draft.template_revision,
        "evidence_links": list(draft.evidence_links),
    }


__all__ = [
    "CandidateNotApprovedForProjectionError",
    "CcTaskCreationDisabledError",
    "NonCurrentSessionError",
    "build_cc_task_plan",
]
