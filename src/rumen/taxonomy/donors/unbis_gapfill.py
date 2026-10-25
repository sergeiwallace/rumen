"""Reviewed, per-concept UNBIS gap-fill imports (never a bulk donor)."""

from __future__ import annotations

from datetime import date

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

from rumen.taxonomy.donors.eurovoc import RUMEN
from rumen.taxonomy.donors.graft import MappingMetadata

UNBIS_SCHEME = URIRef("http://metadata.un.org/thesaurus/")


def add_unbis_gapfill(
    graph: Graph,
    *,
    rumen_leaf: URIRef,
    unbis_uri: URIRef,
    label: str,
    source_version: str,
    reviewer: str | None,
    review_date: date | None,
    relation_type: str = "closeMatch",
) -> tuple[URIRef, URIRef, MappingMetadata]:
    """Add one documented coverage gap and its mandatory reviewed mapping."""
    if not reviewer or review_date is None:
        raise ValueError("UNBIS gap-fill imports require an explicit reviewer id and review date")
    if relation_type not in {"closeMatch", "exactMatch"}:
        raise ValueError(
            "UNBIS gap-fill mappings must be closeMatch or reviewer-confirmed exactMatch"
        )
    graph.add((UNBIS_SCHEME, RDF.type, SKOS.ConceptScheme))
    graph.add((UNBIS_SCHEME, DCTERMS.identifier, Literal("unbis")))
    graph.add((UNBIS_SCHEME, URIRef(f"{RUMEN}facet"), Literal("topic")))
    graph.add((UNBIS_SCHEME, URIRef(f"{RUMEN}schemeVersion"), Literal(source_version)))
    graph.add((unbis_uri, RDF.type, SKOS.Concept))
    graph.add((unbis_uri, SKOS.inScheme, UNBIS_SCHEME))
    graph.add((unbis_uri, SKOS.prefLabel, Literal(label)))
    graph.add((unbis_uri, DCTERMS.source, unbis_uri))
    graph.add((rumen_leaf, getattr(SKOS, relation_type), unbis_uri))
    return (
        rumen_leaf,
        unbis_uri,
        MappingMetadata(
            "rumen-topic", "1", "unbis", source_version, relation_type, reviewer, review_date
        ),
    )
