"""JEL cached code-list adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

from rumen.taxonomy.donors.eurovoc import RUMEN

JEL_CODES_URL = "https://www.aeaweb.org/econlit/jelCodes.php"


def build_jel_scheme(records: Iterable[Mapping[str, str]], *, version: str) -> Graph:
    scheme = URIRef(f"{RUMEN}scheme/jel/{version}")
    graph = Graph()
    graph.add((scheme, RDF.type, SKOS.ConceptScheme))
    graph.add((scheme, DCTERMS.identifier, Literal("jel")))
    graph.add((scheme, URIRef(f"{RUMEN}facet"), Literal("topic")))
    graph.add((scheme, URIRef(f"{RUMEN}schemeVersion"), Literal(version)))
    for record in records:
        code = record["code"]
        concept = URIRef(f"https://www.aeaweb.org/econlit/jelCodes.php?view=jel#{code}")
        graph.add((concept, RDF.type, SKOS.Concept))
        graph.add((concept, SKOS.inScheme, scheme))
        graph.add((concept, SKOS.prefLabel, Literal(record["label"])))
        graph.add((concept, SKOS.notation, Literal(code)))
        graph.add((concept, DCTERMS.source, concept))
        if parent := record.get("parent_code"):
            graph.add(
                (
                    concept,
                    SKOS.broader,
                    URIRef(f"https://www.aeaweb.org/econlit/jelCodes.php?view=jel#{parent}"),
                )
            )
    return graph
