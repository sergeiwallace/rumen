"""Tier-0 deterministic, facet-keyed source signals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Tier0Input:
    source: str
    url: str | None = None
    stable_id_kind: str | None = None
    content: str | None = None


# Values are symbolic taxonomy labels.  The cascade resolves them against active concepts; a
# missing or malformed taxonomy path deliberately leaves only that facet unresolved.
_SOURCE_SIGNALS: dict[str, dict[str, tuple[str, ...]]] = {
    "github_releases": {
        "topic": ("tool",),
        "content-type": ("tool-release",),
        "source": ("github-releases",),
    },
    "npm_registry": {
        "topic": ("tool",),
        "content-type": ("tool-release",),
        "source": ("npm-registry",),
    },
    "hf_daily": {"topic": ("research",), "source": ("hf-daily",)},
}


def classify_tier0(item: Tier0Input) -> dict[str, tuple[str, ...]]:
    """Return symbolic targets keyed by facet; an empty mapping means no rule applies."""
    source_type = item.source.split(":", 1)[0]
    result = dict(_SOURCE_SIGNALS.get(source_type, {}))
    if item.stable_id_kind in {"arxiv", "doi"} or "arxiv.org" in (item.url or "").lower():
        result["topic"] = ("paper",)
        result.setdefault("content-type", ("paper",))
    return result


def tier0_expected_facets(item: Tier0Input) -> frozenset[str]:
    """Return facets this source/identity is contractually expected to resolve at Tier-0.

    This deliberately remains a separate routing assertion from ``classify_tier0()``. If a
    source-type matching regression makes a known source miss its rule, the Tier-2 budget must
    still recognize that fallthrough as anomalous rather than treating it as normal LLM work.
    """
    source_type = item.source.split(":", 1)[0]
    facets = set(_SOURCE_SIGNALS.get(source_type, {}))
    if item.stable_id_kind in {"arxiv", "doi"} or "arxiv.org" in (item.url or "").lower():
        facets.update(("topic", "content-type"))
    return frozenset(facets)
