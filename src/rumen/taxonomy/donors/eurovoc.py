"""Pruned EuroVoc v4.24 politics/policy structural donor."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import IO

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

from rumen.taxonomy.donors.graft import MappingMetadata
from rumen.taxonomy.skos_store import load_skos

RUMEN = "https://rumen.dev/ns#"
EUROVOC_0431 = URIRef("http://eurovoc.europa.eu/100168")
EUROVOC_POLITICS = URIRef("http://eurovoc.europa.eu/4704")
EUROVOC_VERSION = "4.24"
RUMEN_POLITICS_POLICY = URIRef(f"{RUMEN}topic/politics-policy")
RUMEN_IPTC_POLITICS = URIRef(f"{RUMEN}topic/iptc-politics")
RUMEN_SCHEME = URIRef(f"{RUMEN}scheme/topic")
EUROVOC_SCHEME = URIRef("http://eurovoc.europa.eu/100141")
_FACET = URIRef(f"{RUMEN}facet")
_VERSION = URIRef(f"{RUMEN}schemeVersion")


def load_eurovoc_source(source: str | Path | IO[bytes] | IO[str]) -> Graph:
    """Load a real EuroVoc RDF/Turtle distribution through the shared SKOS loader."""
    return load_skos(source, format="turtle")


def build_politics_policy_graft(
    source: str | Path | IO[bytes] | IO[str],
    *,
    included_concepts: set[URIRef],
    reviewer: str,
    review_date: date,
) -> tuple[Graph, list[tuple[URIRef, URIRef, MappingMetadata]]]:
    """Build the bounded 0431 graft without altering EuroVoc's own hierarchy.

    ``included_concepts`` is the reviewed corpus slice.  Its parent chain must be
    included too; this prevents an operator from accidentally bulk-importing 0431.
    """
    donor = load_eurovoc_source(source)
    if EUROVOC_POLITICS not in included_concepts:
        raise ValueError("EuroVoc politics root must be included in the reviewed subset")
    graph = Graph()
    graph.bind("rumen", RUMEN)
    graph.bind("skos", SKOS)
    graph.bind("dcterms", DCTERMS)
    _add_scheme(graph, RUMEN_SCHEME, "rumen-topic", "topic", "1")
    _add_scheme(graph, EUROVOC_SCHEME, "eurovoc", "topic", EUROVOC_VERSION)
    graph.add((RUMEN_IPTC_POLITICS, RDF.type, SKOS.Concept))
    graph.add((RUMEN_IPTC_POLITICS, SKOS.inScheme, RUMEN_SCHEME))
    graph.add((RUMEN_IPTC_POLITICS, SKOS.prefLabel, Literal("Politics")))
    graph.add((RUMEN_POLITICS_POLICY, RDF.type, SKOS.Concept))
    graph.add((RUMEN_POLITICS_POLICY, SKOS.inScheme, RUMEN_SCHEME))
    graph.add((RUMEN_POLITICS_POLICY, SKOS.prefLabel, Literal("Politics and policy")))
    graph.add((RUMEN_POLITICS_POLICY, SKOS.broader, RUMEN_IPTC_POLITICS))
    # 0431 is a EuroVoc micro-thesaurus (a ConceptScheme in the source).  This
    # local source-anchor makes that required closeMatch materializable while
    # retaining its canonical EuroVoc URI and source attribution.
    graph.add((EUROVOC_0431, RDF.type, SKOS.Concept))
    graph.add((EUROVOC_0431, SKOS.inScheme, RUMEN_SCHEME))
    graph.add((EUROVOC_0431, SKOS.prefLabel, Literal("0431 politics and public safety", lang="en")))
    graph.add((EUROVOC_0431, DCTERMS.source, EUROVOC_0431))
    graph.add((RUMEN_POLITICS_POLICY, SKOS.closeMatch, EUROVOC_0431))
    for concept in included_concepts:
        if (concept, RDF.type, SKOS.Concept) not in donor:
            raise ValueError(f"EuroVoc subset does not contain concept {concept}")
        for predicate, obj in donor.predicate_objects(concept):
            if predicate == SKOS.inScheme:
                continue
            if predicate in {SKOS.broader, SKOS.narrower} and obj not in included_concepts:
                continue
            graph.add((concept, predicate, obj))
        graph.add((concept, RDF.type, SKOS.Concept))
        graph.add((concept, SKOS.inScheme, EUROVOC_SCHEME))
        graph.add((concept, DCTERMS.source, concept))
    graph.add((EUROVOC_POLITICS, SKOS.broadMatch, RUMEN_POLITICS_POLICY))
    metadata = [
        (
            RUMEN_POLITICS_POLICY,
            EUROVOC_0431,
            MappingMetadata(
                "rumen-topic", "1", "eurovoc", EUROVOC_VERSION, "closeMatch", reviewer, review_date
            ),
        ),
        (
            EUROVOC_POLITICS,
            RUMEN_POLITICS_POLICY,
            MappingMetadata(
                "eurovoc", EUROVOC_VERSION, "rumen-topic", "1", "broadMatch", reviewer, review_date
            ),
        ),
    ]
    return graph, metadata


def _add_scheme(graph: Graph, scheme: URIRef, identifier: str, facet: str, version: str) -> None:
    graph.add((scheme, RDF.type, SKOS.ConceptScheme))
    graph.add((scheme, DCTERMS.identifier, Literal(identifier)))
    graph.add((scheme, _FACET, Literal(facet)))
    graph.add((scheme, _VERSION, Literal(version)))
