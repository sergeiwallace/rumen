"""Documented partial crosswalk from legacy classifier labels to the topic spine.

This deliberately contains only the labels established by the live-corpus
diagnosis.  Unknown labels must remain untouched by the migration until they are
reviewed and added here.
"""

from __future__ import annotations

from dataclasses import dataclass

from rumen.taxonomy.concept_id import stable_concept_id

_KC = "https://rumen.dev/ns#"
TOPIC_SPINE_SCHEME = f"{_KC}scheme/iptc-topic-spine"


@dataclass(frozen=True)
class TopicConcept:
    """A migration-seeded, top-level topic-spine concept."""

    slug: str
    pref_label: str

    @property
    def uri(self) -> str:
        # A plain string, not `rdflib.URIRef`: this module is imported by the core Alembic
        # migration (0008), which must run without the optional `taxonomy` extra installed
        # (D-24). Every consumer only ever calls `str()` on this value.
        return f"{_KC}topic/iptc-{self.slug}"

    @property
    def id(self) -> int:
        return stable_concept_id(self.uri)


ECONOMY = TopicConcept("economy-business-finance", "Economy, business and finance")
SCIENCE_TECHNOLOGY = TopicConcept("science-technology", "Science and technology")
POLITICS = TopicConcept("politics", "Politics")
LIFESTYLE_LEISURE = TopicConcept("lifestyle-leisure", "Lifestyle and leisure")
NEEDS_REVIEW = TopicConcept("unclassified-needs-review", "Unclassified / needs review")

TOPIC_CONCEPTS = (
    ECONOMY,
    SCIENCE_TECHNOLOGY,
    POLITICS,
    LIFESTYLE_LEISURE,
    NEEDS_REVIEW,
)

# This is intentionally a partial seed, not a claim to enumerate the live 41-label
# corpus.  It covers only the case/synonym/junk/singleton examples documented in that diagnosis.
LEGACY_CATEGORY_CROSSWALK: dict[str, TopicConcept] = {
    "Other": NEEDS_REVIEW,
    "other": NEEDS_REVIEW,
    "unclassified": NEEDS_REVIEW,
    "safe": NEEDS_REVIEW,
    "request clarification": NEEDS_REVIEW,
    "econ": ECONOMY,
    "economics": ECONOMY,
    "ai": SCIENCE_TECHNOLOGY,
    "ai-news": SCIENCE_TECHNOLOGY,
    "tech": SCIENCE_TECHNOLOGY,
    "science": SCIENCE_TECHNOLOGY,
    "war": POLITICS,
    "military": POLITICS,
    "geopolitics": POLITICS,
    "self-help": LIFESTYLE_LEISURE,
    "hobby": LIFESTYLE_LEISURE,
    "gaming": LIFESTYLE_LEISURE,
    "animals": LIFESTYLE_LEISURE,
}


def concept_for_legacy_label(label: str) -> TopicConcept | None:
    """Return the reviewed topic-spine target for ``label``, if one is known."""
    return LEGACY_CATEGORY_CROSSWALK.get(label)
