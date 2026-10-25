"""OpenAlex Topics adapter; callers pass a cached API response or local export."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

from rumen.taxonomy.donors.eurovoc import RUMEN

OPENALEX_TOPICS_API = "https://api.openalex.org/topics"


def build_openalex_scheme(records: Iterable[Mapping[str, object]], *, version: str) -> Graph:
    """Convert a reviewed OpenAlex topic slice to SKOS without fetching in tests."""
    scheme = URIRef(f"{RUMEN}scheme/openalex/{version}")
    graph = _scheme(scheme, "openalex", version)
    for record in records:
        uri = URIRef(str(record["id"]))
        graph.add((uri, RDF.type, SKOS.Concept))
        graph.add((uri, SKOS.inScheme, scheme))
        graph.add((uri, SKOS.prefLabel, Literal(str(record["display_name"]))))
        graph.add((uri, DCTERMS.source, uri))
        if parent := record.get("parent_id"):
            graph.add((uri, SKOS.broader, URIRef(str(parent))))
    return graph


def _scheme(uri: URIRef, identifier: str, version: str) -> Graph:
    graph = Graph()
    graph.add((uri, RDF.type, SKOS.ConceptScheme))
    graph.add((uri, DCTERMS.identifier, Literal(identifier)))
    graph.add((uri, URIRef(f"{RUMEN}facet"), Literal("topic")))
    graph.add((uri, URIRef(f"{RUMEN}schemeVersion"), Literal(version)))
    return graph
