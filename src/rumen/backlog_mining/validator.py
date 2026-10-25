"""Deterministic admission checks for structured backlog assessments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AssessmentValidationError(ValueError):
    """A model assessment is not safe to admit as a draft."""


def validate_candidate(assessment: Mapping[str, Any]) -> None:
    """Require labeled claims and supported inferences before any candidate write."""
    draft = assessment.get("candidate_draft")
    if not isinstance(draft, Mapping) or not draft:
        raise AssessmentValidationError("assessment lacks an immutable candidate draft")
    claims = assessment.get("claims")
    if not isinstance(claims, list) or not claims:
        raise AssessmentValidationError("claims must be a non-empty list")
    evidence_ids = assessment.get("evidence_ids")
    if not isinstance(evidence_ids, set) or not evidence_ids:
        raise AssessmentValidationError("assessment lacks admitted evidence identifiers")
    claim_ids: set[str] = set()
    directly_supported_claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, Mapping) or claim.get("kind") not in {
            "source",
            "internal_context",
            "inference",
        }:
            raise AssessmentValidationError("each claim must have a recognized kind")
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in claim_ids:
            raise AssessmentValidationError("claims require unique non-empty identifiers")
        claim_ids.add(claim_id)
        if claim.get("kind") == "inference":
            support = claim.get("support")
            if not isinstance(support, list) or not support:
                raise AssessmentValidationError("inferences require linked supporting evidence")
            if not all(
                isinstance(item, str) and item in directly_supported_claim_ids for item in support
            ):
                raise AssessmentValidationError(
                    "inference support must reference admitted evidence"
                )
        else:
            evidence_id = claim.get("evidence_id")
            if not isinstance(evidence_id, str) or evidence_id not in evidence_ids:
                raise AssessmentValidationError(
                    "claim must reference admitted source/context evidence"
                )
            directly_supported_claim_ids.add(claim_id)
    draft_claim_ids = draft.get("claim_ids")
    if (
        not isinstance(draft_claim_ids, list)
        or not draft_claim_ids
        or not all(isinstance(item, str) and item in claim_ids for item in draft_claim_ids)
    ):
        raise AssessmentValidationError("candidate draft must reference admitted claims")


def validate_evaluator_pair(
    generator_provenance: Mapping[str, Any],
    evaluator_provenance: Mapping[str, Any],
    evaluator_request: Mapping[str, Any],
) -> None:
    """Enforce the generator/evaluator context and prompt-content boundary."""
    allowed = {"candidate_draft", "source_context_evidence", "evaluator_prompt_template"}
    if set(evaluator_request) != allowed:
        raise AssessmentValidationError(
            "evaluator request must contain exactly its three allowed fields"
        )
    if not isinstance(evaluator_request["candidate_draft"], Mapping):
        raise AssessmentValidationError("evaluator request lacks candidate draft snapshot")
    for field in ("invocation_id", "provider_request_id", "context_id", "backing_context_id"):
        if not generator_provenance.get(field) or not evaluator_provenance.get(field):
            raise AssessmentValidationError(f"missing {field} provenance")
        if generator_provenance[field] == evaluator_provenance[field]:
            raise AssessmentValidationError(f"generator/evaluator reused {field}")
    generator_hash = generator_provenance.get("prompt_content_hash")
    evaluator_hash = evaluator_provenance.get("prompt_content_hash")
    if not isinstance(generator_hash, str) or not isinstance(evaluator_hash, str):
        raise AssessmentValidationError("canonical prompt content hashes are required")
    if generator_hash == evaluator_hash:
        raise AssessmentValidationError(
            "generator/evaluator request content hashes must differ: canonical prompt content hashes must differ"
        )
