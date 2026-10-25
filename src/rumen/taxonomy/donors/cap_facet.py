"""CAP policy-issue facet, kept independent from the topic hierarchy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS
from sqlalchemy.orm import Session

from rumen.models import ItemConcept
from rumen.taxonomy.donors.eurovoc import RUMEN
from rumen.taxonomy.materialize import stable_concept_id

CAP_SCHEME = URIRef(f"{RUMEN}scheme/cap-policy-issue")
CAP_FACET = URIRef(f"{RUMEN}facet/policy-issue")


def build_cap_facet(
    codes: Iterable[Mapping[str, str]], *, version: str, rights_confirmed: bool = False
) -> Graph:
    """Build a personal-use CAP facet; export policy is applied separately.

    ``rights_confirmed`` is accepted here so deployment configuration travels with
    the adapter, while the graph intentionally remains usable in a private install.
    """
    del rights_confirmed  # local use is independent of bundle redistribution rights
    graph = Graph()
    graph.add((CAP_SCHEME, RDF.type, SKOS.ConceptScheme))
    graph.add((CAP_SCHEME, DCTERMS.identifier, Literal("cap-policy-issue")))
    graph.add((CAP_SCHEME, URIRef(f"{RUMEN}facet"), Literal("policy-issue")))
    graph.add((CAP_SCHEME, URIRef(f"{RUMEN}schemeVersion"), Literal(version)))
    for code in codes:
        concept = URIRef(f"{RUMEN}cap/{code['code']}")
        graph.add((concept, RDF.type, SKOS.Concept))
        graph.add((concept, SKOS.inScheme, CAP_SCHEME))
        graph.add((concept, SKOS.prefLabel, Literal(code["label"])))
        graph.add((concept, SKOS.notation, Literal(code["code"])))
        graph.add((concept, DCTERMS.source, concept))
    return graph


def oss_default_bundle(graph: Graph, *, rights_confirmed: bool = False) -> Graph:
    """Return an OSS-safe graph, omitting CAP labels until rights are confirmed."""
    if rights_confirmed:
        return graph
    result = Graph()
    cap_concepts = set(graph.subjects(SKOS.inScheme, CAP_SCHEME))
    for triple in graph:
        if triple[0] != CAP_SCHEME and triple[0] not in cap_concepts:
            result.add(triple)
    return result


def assign_policy_issue(
    session: Session,
    *,
    item_table: str,
    item_id: int,
    concept_uri: URIRef,
    confidence: float,
    method: str,
) -> ItemConcept:
    """Assign a CAP code on its facet, without creating any topic parent relation."""
    assignment = ItemConcept(
        item_table=item_table,
        item_id=item_id,
        concept_id=stable_concept_id(concept_uri),
        facet="policy-issue",
        confidence=confidence,
        method=method,
    )
    session.add(assignment)
    return assignment
