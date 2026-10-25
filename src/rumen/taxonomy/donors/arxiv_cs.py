"""arXiv ``cs.*`` donor adapter for cached category listings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

from rumen.taxonomy.donors.eurovoc import RUMEN

ARXIV_TAXONOMY_URL = "https://arxiv.org/category_taxonomy"


def build_arxiv_cs_scheme(records: Iterable[Mapping[str, str]], *, version: str) -> Graph:
    scheme = URIRef(f"{RUMEN}scheme/arxiv-cs/{version}")
    root = URIRef("https://arxiv.org/category_taxonomy/cs")
    graph = Graph()
    for triple in (
        (scheme, RDF.type, SKOS.ConceptScheme),
        (scheme, DCTERMS.identifier, Literal("arxiv-cs")),
        (scheme, URIRef(f"{RUMEN}facet"), Literal("topic")),
        (scheme, URIRef(f"{RUMEN}schemeVersion"), Literal(version)),
        (root, RDF.type, SKOS.Concept),
        (root, SKOS.inScheme, scheme),
        (root, SKOS.prefLabel, Literal("Computer Science")),
    ):
        graph.add(triple)
    for record in records:
        code = record["code"]
        if not code.startswith("cs."):
            raise ValueError(f"arXiv donor accepts cs.* codes only, got {code!r}")
        concept = URIRef(f"https://arxiv.org/category_taxonomy/{code}")
        graph.add((concept, RDF.type, SKOS.Concept))
        graph.add((concept, SKOS.inScheme, scheme))
        graph.add((concept, SKOS.prefLabel, Literal(record["label"])))
        graph.add((concept, SKOS.notation, Literal(code)))
        graph.add((concept, DCTERMS.source, concept))
        graph.add((concept, SKOS.broader, root))
    return graph
