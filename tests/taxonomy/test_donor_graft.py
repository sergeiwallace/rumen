"""T-13 donor graft and crosswalk acceptance coverage."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from rdflib import URIRef
from rdflib.namespace import RDF, SKOS
from sqlalchemy import text

pytest.importorskip("rdflib", reason="taxonomy extra is not installed")
pytest.importorskip("skosify", reason="taxonomy extra is not installed")

from rumen.taxonomy.donors.arxiv_cs import build_arxiv_cs_scheme
from rumen.taxonomy.donors.eurovoc import (
    EUROVOC_POLITICS,
    RUMEN_POLITICS_POLICY,
    build_politics_policy_graft,
)
from rumen.taxonomy.donors.graft import (
    DonorConfig,
    DonorConfigurationError,
    MappingMetadata,
    graft_donor,
)
from rumen.taxonomy.donors.openalex import build_openalex_scheme

_FIXTURES = Path(__file__).parent / "fixtures"
_VERIFIED = DonorConfig("EuroVoc", True, "https://op.europa.eu/reuse")


def _eurovoc_graph():
    return build_politics_policy_graft(
        _FIXTURES / "eurovoc_politics_subset.ttl",
        included_concepts={
            URIRef("http://eurovoc.europa.eu/4704"),
            URIRef("http://eurovoc.europa.eu/3237"),
            URIRef("http://eurovoc.europa.eu/3517"),
        },
        reviewer="taxonomy-reviewer",
        review_date=date(2026, 7, 19),
    )


def test_donor_graft_when_spliced_then_preserves_hierarchy_and_records_broad_match(session) -> None:
    graph, mappings = _eurovoc_graph()
    result = graft_donor(session, graph, config=_VERIFIED, mappings=mappings)
    session.commit()

    assert result.skipped is False
    assert (URIRef("http://eurovoc.europa.eu/3237"), SKOS.broader, EUROVOC_POLITICS) in graph
    assert (EUROVOC_POLITICS, SKOS.broadMatch, RUMEN_POLITICS_POLICY) in graph
    assert (
        session.execute(
            text("SELECT count(*) FROM skos_relations WHERE relation_type = 'broadMatch'")
        ).scalar_one()
        == 1
    )


def test_donor_graft_when_schemes_overlap_then_keeps_two_concepts_and_crosswalk(session) -> None:
    # Baseline after migration (which seeds 5 T-14 topic-spine concepts) so the
    # assertion is a delta, not an absolute count -- robust to future seed changes.
    baseline_concepts = session.execute(text("SELECT count(*) FROM skos_concepts")).scalar_one()
    openalex_record = json.loads((_FIXTURES / "openalex_machine_learning_subset.json").read_text())
    arxiv_code, arxiv_label = (
        (_FIXTURES / "arxiv_cs_lg_subset.txt").read_text().splitlines()[1].split("\\t")
    )
    openalex = build_openalex_scheme([openalex_record], version="2026-07")
    arxiv = build_arxiv_cs_scheme([{"code": arxiv_code, "label": arxiv_label}], version="2026")
    arxiv_concept = URIRef("https://arxiv.org/category_taxonomy/cs.LG")
    openalex_concept = URIRef(openalex_record["id"])
    openalex.add((openalex_concept, SKOS.closeMatch, arxiv_concept))
    result = graft_donor(
        session,
        openalex + arxiv,
        config=DonorConfig("OpenAlex/arXiv", True, "https://docs.openalex.org"),
        mappings=[
            (
                openalex_concept,
                arxiv_concept,
                MappingMetadata(
                    "openalex",
                    "2026-07",
                    "arxiv-cs",
                    "2026",
                    "closeMatch",
                    "reviewer",
                    datetime.now(UTC).date(),
                ),
            )
        ],
    )
    session.commit()

    assert result.materialization is not None
    assert (
        session.execute(text("SELECT count(*) FROM skos_concepts")).scalar_one() - baseline_concepts
        == 3
    )
    assert (
        session.execute(
            text("SELECT count(*) FROM skos_relations WHERE relation_type = 'closeMatch'")
        ).scalar_one()
        == 1
    )


def test_donor_graft_when_validation_fails_then_skips_without_partial_branch(session) -> None:
    baseline_concepts = session.execute(text("SELECT count(*) FROM skos_concepts")).scalar_one()
    graph, mappings = _eurovoc_graph()
    orphan = URIRef("https://example.test/orphan")
    graph.add((orphan, RDF.type, SKOS.Concept))
    graph.add((orphan, SKOS.inScheme, URIRef("http://eurovoc.europa.eu/100141")))
    graph.add((orphan, SKOS.prefLabel, URIRef("https://example.test/not-a-label")))

    result = graft_donor(session, graph, config=_VERIFIED, mappings=mappings)

    assert result.skipped is True
    assert (
        session.execute(text("SELECT count(*) FROM skos_concepts")).scalar_one()
        == baseline_concepts
    )


def test_donor_graft_when_reuse_verification_is_missing_then_refuses_donor(session) -> None:
    graph, mappings = _eurovoc_graph()

    with pytest.raises(
        DonorConfigurationError, match="CRS.*missing documented reuse-terms verification"
    ):
        graft_donor(session, graph, config=DonorConfig("CRS", False), mappings=mappings)
