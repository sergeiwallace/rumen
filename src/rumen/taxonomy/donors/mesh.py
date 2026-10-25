"""MeSH descriptor XML adapter for reviewed health subtrees."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

from rumen.taxonomy.donors.eurovoc import RUMEN

MESH_DESCRIPTOR_XML_URL = (
    "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml"
)


def build_mesh_scheme(records: Iterable[Mapping[str, str]], *, version: str) -> Graph:
    """Create SKOS from a cached, pre-pruned MeSH descriptor extraction."""
    scheme = URIRef(f"{RUMEN}scheme/mesh/{version}")
    graph = Graph()
    graph.add((scheme, RDF.type, SKOS.ConceptScheme))
    graph.add((scheme, DCTERMS.identifier, Literal("mesh")))
    graph.add((scheme, URIRef(f"{RUMEN}facet"), Literal("topic")))
    graph.add((scheme, URIRef(f"{RUMEN}schemeVersion"), Literal(version)))
    for record in records:
        descriptor = record["descriptor_id"]
        concept = URIRef(f"https://id.nlm.nih.gov/mesh/{descriptor}")
        graph.add((concept, RDF.type, SKOS.Concept))
        graph.add((concept, SKOS.inScheme, scheme))
        graph.add((concept, SKOS.prefLabel, Literal(record["label"])))
        graph.add((concept, DCTERMS.source, concept))
        if parent := record.get("parent_id"):
            graph.add((concept, SKOS.broader, URIRef(f"https://id.nlm.nih.gov/mesh/{parent}")))
    return graph
