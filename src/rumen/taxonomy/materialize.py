"""Validated, idempotent SKOS/RDF materialization into the D-24 runtime tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS
from sqlalchemy import Column, Date, Integer, MetaData, Table, Text, select
from sqlalchemy.orm import Session

from rumen.taxonomy.concept_id import stable_concept_id as _stable_concept_id
from rumen.taxonomy.validate import ensure_valid

_KC = "https://rumen.dev/ns#"
_FACET = URIRef(f"{_KC}facet")
_SCHEME_VERSION = URIRef(f"{_KC}schemeVersion")
_RELATION_TYPES = {
    SKOS.broader: "broader",
    SKOS.narrower: "narrower",
    SKOS.related: "related",
    SKOS.broadMatch: "broadMatch",
    SKOS.closeMatch: "closeMatch",
    SKOS.exactMatch: "exactMatch",
    URIRef(f"{_KC}useFor"): "useFor",
}

_metadata = MetaData()
_concepts = Table(
    "skos_concepts",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("facet", Text, nullable=False),
    Column("pref_label", Text, nullable=False),
    Column("notation", Text),
    Column("scheme", Text, nullable=False),
    Column("scheme_version", Text),
    Column("source_uri", Text),
    Column("status", Text, nullable=False),
)
_relations = Table(
    "skos_relations",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("source_concept_id", Integer, nullable=False),
    Column("target_concept_id", Integer, nullable=False),
    Column("relation_type", Text, nullable=False),
    Column("source_scheme", Text),
    Column("source_version", Text),
    Column("target_scheme", Text),
    Column("target_version", Text),
    Column("reviewer", Text),
    Column("review_date", Date),
)


@dataclass(frozen=True)
class MaterializationResult:
    """Counts written by one materialization pass."""

    concepts_written: int
    relations_written: int


def materialize_scheme(session: Session, graph: Graph) -> MaterializationResult:
    """Load a validated RDF graph into the SKOS runtime tables.

    Validation completes before the first database statement, which makes invalid source graphs
    fail closed.  Existing rows are inspected before insert so an unchanged source is a true
    no-op on both SQLite and Postgres without dialect-specific upsert syntax.
    """
    ensure_valid(graph)
    concept_rows, uri_ids = _concept_rows(graph)
    relation_rows = _relation_rows(graph, uri_ids)

    existing_concept_ids = set(session.execute(select(_concepts.c.id)).scalars())
    new_concepts = [row for row in concept_rows if row["id"] not in existing_concept_ids]

    existing_relations = {
        tuple(row)
        for row in session.execute(
            select(
                _relations.c.source_concept_id,
                _relations.c.target_concept_id,
                _relations.c.relation_type,
            )
        )
    }
    new_relations = [
        row
        for row in relation_rows
        if (row["source_concept_id"], row["target_concept_id"], row["relation_type"])
        not in existing_relations
    ]

    # A savepoint ensures any unexpected write failure rolls back this materialization only,
    # while callers retain their surrounding session transaction.
    with session.begin_nested():
        if new_concepts:
            session.execute(_concepts.insert(), new_concepts)
        if new_relations:
            session.execute(_relations.insert(), new_relations)
    return MaterializationResult(len(new_concepts), len(new_relations))


def _concept_rows(graph: Graph) -> tuple[list[dict[str, Any]], dict[URIRef, int]]:
    concepts = sorted(set(graph.subjects(RDF.type, SKOS.Concept)), key=str)
    uri_ids: dict[URIRef, int] = {}
    for concept in concepts:
        if not isinstance(concept, URIRef):
            raise TypeError("SKOS concepts must have URI identifiers for stable materialization")
        concept_id = stable_concept_id(concept)
        if concept_id in uri_ids.values():
            raise ValueError("stable materialization id collision; choose distinct concept URIs")
        uri_ids[concept] = concept_id

    rows = []
    for concept in concepts:
        scheme = _one(graph.objects(concept, SKOS.inScheme), "inScheme", concept)
        label = _one(graph.objects(concept, SKOS.prefLabel), "prefLabel", concept)
        facet = _one(graph.objects(scheme, _FACET), "facet", scheme)
        rows.append(
            {
                "id": uri_ids[concept],
                "facet": str(facet),
                "pref_label": str(label),
                "notation": _optional(graph.objects(concept, SKOS.notation)),
                "scheme": _optional(graph.objects(scheme, DCTERMS.identifier)) or str(scheme),
                "scheme_version": _optional(graph.objects(scheme, _SCHEME_VERSION)),
                "source_uri": str(concept),
                "status": "active",
            }
        )
    return rows, uri_ids


def _relation_rows(graph: Graph, uri_ids: dict[URIRef, int]) -> list[dict[str, Any]]:
    rows = []
    for predicate, relation_type in _RELATION_TYPES.items():
        for source, target in graph.subject_objects(predicate):
            if source not in uri_ids or target not in uri_ids:
                continue
            rows.append(
                {
                    "source_concept_id": uri_ids[source],
                    "target_concept_id": uri_ids[target],
                    "relation_type": relation_type,
                }
            )
    return rows


# Re-exported for existing callers (`from rumen.taxonomy.materialize import
# stable_concept_id`); the implementation lives in ``concept_id`` so the migration-time legacy
# crosswalk can use it without importing rdflib.
stable_concept_id = _stable_concept_id


def _one(values: Any, name: str, subject: URIRef) -> Any:
    value = next(iter(values), None)
    if value is None:
        raise ValueError(f"{subject} is missing required SKOS {name}")
    return value


def _optional(values: Any) -> str | None:
    value = next(iter(values), None)
    return None if value is None else str(value)
