"""T-12: validated RDF materialization acceptance coverage."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

pytest.importorskip("rdflib", reason="taxonomy extra is not installed")
pytest.importorskip("skosify", reason="taxonomy extra is not installed")

from rumen.taxonomy.materialize import materialize_scheme
from rumen.taxonomy.skos_store import load_skos
from rumen.taxonomy.validate import TaxonomyValidationError

_PREFIX = """
@prefix ex: <https://example.test/taxonomy/> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix rumen: <https://rumen.dev/ns#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
"""

_VALID = (
    _PREFIX
    + """
ex:scheme a skos:ConceptScheme ; dcterms:identifier "example-topic" ; rumen:facet "topic" ;
    rumen:schemeVersion "2026.07" .
ex:root a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Root" ; skos:notation "ROOT" .
ex:child a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Child" ;
    skos:broader ex:root ; skos:related ex:root ; skos:broadMatch ex:root ;
    skos:closeMatch ex:root ; skos:exactMatch ex:root ; rumen:useFor ex:root .
ex:root skos:narrower ex:child .
"""
)


def test_materialize_scheme_when_valid_then_writes_concepts_and_supported_relations(
    session,
) -> None:
    # Migration 0008 seeds 5 baseline topic-spine concepts (T-14) into every migrated
    # DB, independent of this test's own scheme -- scope the query to this scheme only
    # so that seed doesn't leak into a set-equality assertion.
    result = materialize_scheme(session, load_skos(_VALID, format="turtle"))
    session.commit()

    concepts = session.execute(
        text(
            "SELECT facet, pref_label, notation, scheme, scheme_version, status "
            "FROM skos_concepts WHERE scheme=:scheme"
        ),
        {"scheme": "example-topic"},
    ).all()
    relations = session.execute(text("SELECT relation_type FROM skos_relations")).scalars().all()
    assert result.concepts_written == 2
    assert {tuple(row) for row in concepts} == {
        ("topic", "Root", "ROOT", "example-topic", "2026.07", "active"),
        ("topic", "Child", None, "example-topic", "2026.07", "active"),
    }
    assert set(relations) == {
        "broader",
        "narrower",
        "related",
        "broadMatch",
        "closeMatch",
        "exactMatch",
        "useFor",
    }


def test_materialize_scheme_when_unchanged_then_is_idempotent(session) -> None:
    # Baseline captured after migration (which seeds 5 T-14 topic-spine concepts) so
    # the assertion is a delta, not an absolute count -- robust to future seed changes.
    baseline_concepts = session.execute(text("SELECT count(*) FROM skos_concepts")).scalar()
    graph = load_skos(_VALID, format="turtle")
    first = materialize_scheme(session, graph)
    session.commit()
    second = materialize_scheme(session, graph)
    session.commit()

    assert first.concepts_written == 2
    assert second.concepts_written == 0
    assert (
        session.execute(text("SELECT count(*) FROM skos_concepts")).scalar() - baseline_concepts
        == 2
    )
    assert session.execute(text("SELECT count(*) FROM skos_relations")).scalar() == 7


def test_materialize_scheme_when_validation_fails_then_writes_nothing(session) -> None:
    baseline_concepts = session.execute(text("SELECT count(*) FROM skos_concepts")).scalar()
    invalid = load_skos(
        _VALID + 'ex:orphan a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Orphan" .',
        format="turtle",
    )

    with pytest.raises(TaxonomyValidationError):
        materialize_scheme(session, invalid)

    assert session.execute(text("SELECT count(*) FROM skos_concepts")).scalar() == baseline_concepts
    assert session.execute(text("SELECT count(*) FROM skos_relations")).scalar() == 0


def test_skos_relation_when_concept_is_missing_then_foreign_key_rejects_it(session) -> None:
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                "INSERT INTO skos_relations "
                "(source_concept_id, target_concept_id, relation_type, created_at) "
                "VALUES (999991, 999992, 'related', CURRENT_TIMESTAMP)"
            )
        )
        session.commit()
    session.rollback()
