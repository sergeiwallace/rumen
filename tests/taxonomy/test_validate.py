"""T-12: qSKOS/Skosify and required custom validation coverage."""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib", reason="taxonomy extra is not installed")
pytest.importorskip("skosify", reason="taxonomy extra is not installed")

from rumen.taxonomy.skos_store import load_skos
from rumen.taxonomy.validate import (
    TaxonomyValidationError,
    ensure_valid,
    validate_scheme,
)

_PREFIX = """
@prefix ex: <https://example.test/taxonomy/> .
@prefix rumen: <https://rumen.dev/ns#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
"""

_VALID = (
    _PREFIX
    + """
ex:scheme a skos:ConceptScheme ; rumen:facet "topic" .
ex:root a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Root" .
ex:child a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Child" ; skos:broader ex:root .
"""
)


def test_validate_scheme_when_well_formed_then_reports_no_issues() -> None:
    assert validate_scheme(load_skos(_VALID, format="turtle")) == []


@pytest.mark.parametrize(
    ("body", "code", "expected_ids"),
    [
        (
            """ex:orphan a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Orphan" .""",
            "orphan_concept",
            {"https://example.test/taxonomy/orphan"},
        ),
        (
            """ex:a a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "A" ; skos:broader ex:b .
               ex:b a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "B" ; skos:broader ex:a .""",
            "hierarchy_cycle",
            {"https://example.test/taxonomy/a", "https://example.test/taxonomy/b"},
        ),
        (
            """ex:a a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Same" ; skos:broader ex:root .
               ex:b a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "same" ; skos:broader ex:root .""",
            "duplicate_pref_label",
            {"https://example.test/taxonomy/a", "https://example.test/taxonomy/b"},
        ),
    ],
)
def test_validate_scheme_when_required_qskos_violation_then_reports_offending_concepts(
    body: str, code: str, expected_ids: set[str]
) -> None:
    issues = validate_scheme(load_skos(_VALID + body, format="turtle"))

    issue = next(issue for issue in issues if issue.code == code)
    assert set(issue.concept_ids) == expected_ids


def test_validate_scheme_when_components_disconnected_then_reports_concepts() -> None:
    graph = load_skos(
        _PREFIX
        + """
        ex:scheme a skos:ConceptScheme ; rumen:facet "topic" .
        ex:a a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "A" ; skos:related ex:b .
        ex:b a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "B" ; skos:related ex:a .
        ex:c a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "C" ; skos:related ex:d .
        ex:d a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "D" ; skos:related ex:c .
        """,
        format="turtle",
    )

    issue = next(
        issue for issue in validate_scheme(graph) if issue.code == "disconnected_components"
    )
    assert set(issue.concept_ids) == {
        "https://example.test/taxonomy/a",
        "https://example.test/taxonomy/b",
        "https://example.test/taxonomy/c",
        "https://example.test/taxonomy/d",
    }


def test_ensure_valid_when_invalid_then_raises_with_all_issues() -> None:
    with pytest.raises(TaxonomyValidationError, match="orphan_concept"):
        ensure_valid(
            load_skos(
                _VALID
                + 'ex:orphan a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Orphan" .',
                format="turtle",
            )
        )
