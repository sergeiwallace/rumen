"""Config-driven LLM proposal generation and strict proposal validation."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field

from rumen.llm import LlmCommand, LlmUnavailable, parse_json_response
from rumen.taxonomy.audit.detect import AuditIssue

logger = logging.getLogger(__name__)

_OPS = frozenset(("add", "deepen", "consolidate", "prune", "relabel"))


@dataclass(frozen=True)
class AuditProposal:
    """One schema-validated LLM suggestion that is safe to hand to the applier."""

    op_type: str
    concept_ids: tuple[int, ...]
    rationale: str
    children: tuple[dict[str, str], ...] = ()
    target_concept_id: int | None = None
    pref_label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def build_prompt(issue: AuditIssue) -> str:
    """Constrain the model to one edit type and a compact JSON proposal schema."""
    return (
        "You are maintaining a controlled taxonomy. Return ONLY one JSON object. "
        "Do not delete concepts: consolidations must use a useFor alias and relabels retain IDs.\n"
        f"Detected issue: {json.dumps({'op_type': issue.op_type, 'concept_ids': issue.concept_ids, 'details': issue.details})}\n"
        "Required fields: op_type (must match), concept_ids (must exactly match), rationale "
        "(non-empty). deepen additionally requires children: [{pref_label, slug?}]. "
        "consolidate requires target_concept_id from concept_ids. relabel requires pref_label "
        "and may include target_concept_id. prune has no destructive action."
    )


def propose_edit(issue: AuditIssue, llm_command: LlmCommand | None) -> AuditProposal | None:
    """Ask the configured command for one proposal; malformed output is skipped safely."""
    if llm_command is None:
        logger.warning(
            "No configured LLM command; skipping taxonomy audit issue: %s", issue.rationale
        )
        return None
    try:
        raw = llm_command.run(build_prompt(issue))
        return validate_proposal(parse_json_response(raw), issue)
    except (
        subprocess.SubprocessError,
        OSError,
        LlmUnavailable,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        logger.warning(
            "Invalid taxonomy-audit proposal; skipping issue (%s): %s", issue.rationale, exc
        )
        return None


def validate_proposal(data: object, issue: AuditIssue) -> AuditProposal:
    """Validate untrusted LLM JSON before it can reach database mutation code."""
    if not isinstance(data, dict):
        raise TypeError("proposal must be a JSON object")
    op_type = data.get("op_type")
    if op_type not in _OPS or op_type != issue.op_type:
        raise ValueError("proposal op_type must match the detected issue")
    concept_ids = data.get("concept_ids")
    if not isinstance(concept_ids, list) or any(type(value) is not int for value in concept_ids):
        raise ValueError("proposal concept_ids must be an integer array")
    if tuple(concept_ids) != issue.concept_ids:
        raise ValueError("proposal concept_ids must exactly match the issue")
    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("proposal rationale must be non-empty")

    children: tuple[dict[str, str], ...] = ()
    if op_type == "deepen":
        raw_children = data.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise ValueError("deepen proposal needs at least one child")
        parsed_children: list[dict[str, str]] = []
        for child in raw_children:
            if not isinstance(child, dict) or not isinstance(child.get("pref_label"), str):
                raise TypeError("every child needs a pref_label")
            label = child["pref_label"].strip()
            if not label:
                raise ValueError("child pref_label must be non-empty")
            parsed_children.append(
                {
                    key: value
                    for key, value in child.items()
                    if key == "slug" and isinstance(value, str)
                }
                | {"pref_label": label}
            )
        children = tuple(parsed_children)

    target = data.get("target_concept_id")
    if op_type == "consolidate":
        if type(target) is not int or target not in concept_ids:
            raise ValueError("consolidate target_concept_id must name an affected concept")
    elif target is not None and type(target) is not int:
        raise ValueError("target_concept_id must be an integer")
    pref_label = data.get("pref_label")
    if op_type == "relabel" and (not isinstance(pref_label, str) or not pref_label.strip()):
        raise ValueError("relabel proposal needs a non-empty pref_label")
    return AuditProposal(
        op_type,
        tuple(concept_ids),
        rationale.strip(),
        children,
        target,
        pref_label.strip() if isinstance(pref_label, str) else None,
        {
            key: value
            for key, value in data.items()
            if key
            not in {
                "op_type",
                "concept_ids",
                "rationale",
                "children",
                "target_concept_id",
                "pref_label",
            }
        },
    )
