"""Backlog-mining model exports and lifecycle vocabulary."""

from rumen.models import (
    BacklogCandidate,
    BacklogCandidateDisposition,
    BacklogCandidateEvidence,
    BacklogCandidatePromotion,
    BacklogCandidateRecommendation,
    BacklogCandidateTaskLink,
    BacklogMappingReview,
    BacklogMiningRun,
    BacklogTaskMatch,
)

ACTIVE_CANDIDATE_STATES = frozenset({"draft", "accepted", "edited", "deferred"})
CANDIDATE_STATES = ACTIVE_CANDIDATE_STATES | frozenset(
    {"rejected", "knowledge_only", "merged", "superseded"}
)
DISPOSITION_ACTIONS = frozenset(
    {"accept", "edit", "defer", "reject", "knowledge_only", "merge", "reopen", "supersede"}
)

__all__ = [
    "ACTIVE_CANDIDATE_STATES",
    "CANDIDATE_STATES",
    "DISPOSITION_ACTIONS",
    "BacklogCandidate",
    "BacklogCandidateDisposition",
    "BacklogCandidateEvidence",
    "BacklogCandidatePromotion",
    "BacklogCandidateRecommendation",
    "BacklogCandidateTaskLink",
    "BacklogMiningRun",
    "BacklogMappingReview",
    "BacklogTaskMatch",
]
