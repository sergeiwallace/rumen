"""Facet-constrained, multi-label Tier-2 LLM judge."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence

from rumen.llm import LlmCommand, LlmUnavailable, parse_json_response
from rumen.models import SkosConcept

logger = logging.getLogger(__name__)

_PROMPT = """You are a strict content classifier. Classify the item using one or more concepts
from the requested facet only. Respond ONLY with JSON:
{{"concept_ids": [<integer concept ids>], "confidence": <0.0-1.0>}}

Facet: {facet}
Allowed concepts: {concepts}
Item title: {title}
Item content: {content}
"""


def build_prompt(
    title: str, content: str | None, *, facet: str, concepts: Sequence[SkosConcept]
) -> str:
    allowed = ", ".join(f"{concept.id}:{concept.pref_label}" for concept in concepts)
    return _PROMPT.format(
        facet=facet, concepts=allowed, title=title, content=(content or "")[:2000]
    )


def classify_tier2(
    title: str,
    content: str | None,
    *,
    facet: str,
    concepts: Sequence[SkosConcept],
    llm_command: LlmCommand,
) -> dict[int, float] | None:
    """Return validated active concept ids for ``facet`` or ``None`` without mutation."""
    allowed = {
        concept.id for concept in concepts if concept.facet == facet and concept.status == "active"
    }
    if not allowed:
        return None
    try:
        raw = llm_command.run(build_prompt(title, content, facet=facet, concepts=concepts))
    except (subprocess.SubprocessError, OSError, LlmUnavailable) as exc:
        logger.warning("Tier-2 LLM judge unavailable for facet %s (%s)", facet, exc)
        return None
    try:
        parsed = parse_json_response(raw)
        concept_ids = parsed["concept_ids"]
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not isinstance(concept_ids, list)
        or not concept_ids
        or any(
            not isinstance(concept_id, int) or isinstance(concept_id, bool)
            for concept_id in concept_ids
        )
        or len(set(concept_ids)) != len(concept_ids)
        or not set(concept_ids).issubset(allowed)
    ):
        return None
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (AttributeError, TypeError, ValueError):
        return None
    return {concept_id: max(0.0, min(1.0, confidence)) for concept_id in concept_ids}
