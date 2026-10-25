"""Embedding-based, per-facet concept clustering for the D-23 audit."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.embeddings import EmbeddingProvider, cosine_similarity, get_default_provider
from rumen.models import SkosConcept


@dataclass(frozen=True)
class ConceptCluster:
    """A connected component of semantically similar concepts in one facet."""

    facet: str
    concept_ids: tuple[int, ...]


@dataclass(frozen=True)
class SimilarConceptPair:
    """One pair whose label embeddings clear the configured duplicate threshold."""

    facet: str
    first_id: int
    second_id: int
    similarity: float


def cluster_concepts(
    session: Session,
    *,
    provider: EmbeddingProvider | None = None,
    similarity_threshold: float = 0.92,
) -> tuple[list[ConceptCluster], list[SimilarConceptPair]]:
    """Cluster active concepts within facets and return the duplicate-strength pairs.

    A threshold graph (rather than a fixed-k clustering algorithm) keeps the audit
    conservative: a cluster only connects labels for which the configured embedding
    provider actually reports a sufficiently high cosine similarity.
    """
    active_provider = provider or get_default_provider()
    concepts = session.scalars(
        select(SkosConcept)
        .where(SkosConcept.status == "active")
        .order_by(SkosConcept.facet, SkosConcept.id)
    ).all()
    by_facet: dict[str, list[SkosConcept]] = {}
    for concept in concepts:
        by_facet.setdefault(concept.facet, []).append(concept)

    clusters: list[ConceptCluster] = []
    pairs: list[SimilarConceptPair] = []
    for facet, facet_concepts in by_facet.items():
        vectors = active_provider.embed([concept.pref_label for concept in facet_concepts])
        neighbors: dict[int, set[int]] = {concept.id: set() for concept in facet_concepts}
        for first_index, first in enumerate(facet_concepts):
            for second_index in range(first_index + 1, len(facet_concepts)):
                second = facet_concepts[second_index]
                similarity = cosine_similarity(vectors[first_index], vectors[second_index])
                if similarity < similarity_threshold:
                    continue
                pairs.append(SimilarConceptPair(facet, first.id, second.id, similarity))
                neighbors[first.id].add(second.id)
                neighbors[second.id].add(first.id)
        visited: set[int] = set()
        for concept in facet_concepts:
            if concept.id in visited:
                continue
            component: set[int] = set()
            pending = [concept.id]
            while pending:
                concept_id = pending.pop()
                if concept_id in visited:
                    continue
                visited.add(concept_id)
                component.add(concept_id)
                pending.extend(neighbors[concept_id] - visited)
            clusters.append(ConceptCluster(facet, tuple(sorted(component))))
    return clusters, pairs
