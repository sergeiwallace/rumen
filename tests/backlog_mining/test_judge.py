from __future__ import annotations

import hashlib
import json
import sys

import pytest

from rumen.backlog_mining.judge import LlmActionabilityJudge
from rumen.backlog_mining.validator import (
    AssessmentValidationError,
    validate_candidate,
    validate_evaluator_pair,
)
from rumen.config import TargetTemplate
from rumen.llm import FallbackLlmCommand, LlmExecution, SubprocessLlmCommand


def _provenance(prefix: str, request_hash: str) -> dict[str, str]:
    return {
        "invocation_id": f"invocation-{prefix}",
        "provider_request_id": f"request-{prefix}",
        "context_id": f"context-{prefix}",
        "backing_context_id": f"backing-context-{prefix}",
        "prompt_content_hash": request_hash,
        "request_content_hash": request_hash,
    }


def _generator_response() -> str:
    return json.dumps(
        {
            "outcome": "actionable",
            "candidate_draft": {"title": "Draft", "claim_ids": ["claim-source"]},
            "claims": [
                {
                    "id": "claim-source",
                    "kind": "source",
                    "evidence_id": "source:1",
                }
            ],
        }
    )


def _packet() -> dict:
    return {
        "source_claims": [{"id": 1, "url": "https://example.test", "quote_or_excerpt": "quote"}],
        "entity_identity": {},
    }


def test_evaluator_pair_when_context_or_request_hash_is_reused_then_rejects_without_draft() -> None:
    request = {
        "candidate_draft": {"title": "Draft"},
        "source_context_evidence": {"source_claims": []},
        "evaluator_prompt_template": {"identity": "e", "instructions": "review"},
    }
    with pytest.raises(AssessmentValidationError, match="reused invocation_id"):
        validate_evaluator_pair(
            _provenance("same", "generator"), _provenance("same", "evaluator"), request
        )
    with pytest.raises(AssessmentValidationError, match="request content hashes must differ"):
        validate_evaluator_pair(_provenance("g", "same"), _provenance("e", "same"), request)


def test_evaluator_pair_when_generator_metadata_is_serialized_then_rejects() -> None:
    request = {
        "candidate_draft": {"title": "Draft"},
        "source_context_evidence": {},
        "evaluator_prompt_template": {},
        "generator_provenance": {},
    }
    with pytest.raises(AssessmentValidationError, match="exactly"):
        validate_evaluator_pair(_provenance("g", "one"), _provenance("e", "two"), request)


def test_evaluator_pair_when_only_canonical_prompt_content_is_reused_then_rejects() -> None:
    request = {
        "candidate_draft": {"title": "Draft"},
        "source_context_evidence": {"source_claims": []},
        "evaluator_prompt_template": {"identity": "e", "instructions": "review"},
    }
    generator = _provenance("g", "same-canonical-content")
    evaluator = _provenance("e", "same-canonical-content")
    generator["request_content_hash"] = "different-generator-request"
    evaluator["request_content_hash"] = "different-evaluator-request"

    with pytest.raises(
        AssessmentValidationError, match="canonical prompt content hashes must differ"
    ):
        validate_evaluator_pair(generator, evaluator, request)


def test_validator_when_claims_are_empty_or_inference_support_dangles_then_rejects() -> None:
    with pytest.raises(AssessmentValidationError, match="non-empty"):
        validate_candidate(
            {
                "candidate_draft": {"title": "Draft", "claim_ids": ["source"]},
                "claims": [],
                "evidence_ids": {"source:1"},
            }
        )
    with pytest.raises(AssessmentValidationError, match="inference support"):
        validate_candidate(
            {
                "candidate_draft": {"title": "Draft", "claim_ids": ["inference"]},
                "claims": [{"id": "inference", "kind": "inference", "support": ["not-evidence"]}],
                "evidence_ids": {"source:1"},
            }
        )


_ACTIONABILITY_SCRIPT = """
import json
import sys

request = json.load(sys.stdin)
if "candidate_draft" in request:
    print(json.dumps({"approved": True}))
else:
    print(json.dumps({
        "outcome": "actionable",
        "candidate_draft": {"title": "Draft", "claim_ids": ["claim-source"]},
        "claims": [{"id": "claim-source", "kind": "source", "evidence_id": "source:1"}],
    }))
"""


@pytest.mark.parametrize("use_fallback", [False, True])
def test_judge_when_trusted_adapter_executes_each_role_then_uses_distinct_processes(
    use_fallback: bool,
) -> None:
    target = TargetTemplate("owner/repo", "beads", "v1")

    def factory():
        command = SubprocessLlmCommand([sys.executable, "-c", _ACTIONABILITY_SCRIPT])
        return FallbackLlmCommand([command]) if use_fallback else command

    result = LlmActionabilityJudge(factory).evaluate(_packet(), target)

    assert result.outcome == "actionable"
    assert result.evaluator_provenance is not None
    assert set(result.evaluator_request or {}) == {
        "candidate_draft",
        "source_context_evidence",
        "evaluator_prompt_template",
    }
    generator_pid = int(
        result.generator_provenance["provider_request_id"].removeprefix("subprocess:")
    )
    evaluator_pid = int(
        result.evaluator_provenance["provider_request_id"].removeprefix("subprocess:")
    )
    assert generator_pid != evaluator_pid
    expected_generator_request = LlmActionabilityJudge._generator_prompt(
        _packet(),
        target,
        instructions=LlmActionabilityJudge._generator_prompt_content(),
    )
    expected_evaluator_request = json.dumps(
        result.evaluator_request, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert (
        result.generator_provenance["request_content_hash"]
        == hashlib.sha256(expected_generator_request.encode()).hexdigest()
    )
    assert (
        result.evaluator_provenance["request_content_hash"]
        == hashlib.sha256(expected_evaluator_request.encode()).hexdigest()
    )


def test_judge_when_distinct_wrappers_report_unique_ids_but_share_state_then_abstains() -> None:
    class Wrapper:
        def __init__(self, number: int, backing_state: dict[str, bool]) -> None:
            self.number = number
            self.backing_state = backing_state
            self.backing_context_id = f"reported-backing-{number}"

        def execute(self, request: str) -> LlmExecution:
            if "candidate_draft" not in json.loads(request):
                self.backing_state["generated"] = True
                response = _generator_response()
            else:
                response = json.dumps({"approved": self.backing_state["generated"]})
            return LlmExecution(
                response=response,
                invocation_id=f"invocation-{self.number}",
                provider_request_id=f"request-{self.number}",
                context_id=f"wrapper-context-{self.number}",
                backing_context_id=self.backing_context_id,
            )

    backing_state = {"generated": False}
    commands: list[Wrapper] = []

    def factory() -> Wrapper:
        command = Wrapper(len(commands) + 1, backing_state)
        commands.append(command)
        return command

    result = LlmActionabilityJudge(factory).evaluate(
        _packet(), TargetTemplate("owner/repo", "beads", "v1")
    )

    assert len(commands) == 1
    assert backing_state["generated"] is False
    assert result.outcome == "abstained"
    assert result.candidate_draft is None


def test_judge_when_terminal_backend_raises_or_budget_is_spent_then_abstains() -> None:
    failure = LlmActionabilityJudge(
        lambda: SubprocessLlmCommand([sys.executable, "-c", "raise SystemExit(1)"])
    ).evaluate(_packet(), TargetTemplate("owner/repo", "beads", "v1"))
    assert failure.outcome == "abstained"
    assert failure.assessment["terminal_tier"] == "rules/templates-only"
    assert (
        LlmActionabilityJudge(
            lambda: SubprocessLlmCommand([sys.executable, "-c", "raise SystemExit(1)"]),
            max_calls=0,
        )
        .evaluate(_packet(), TargetTemplate("owner/repo", "beads", "v1"))
        .outcome
        == "abstained"
    )
