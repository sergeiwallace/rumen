"""T-13 bounded EuroVoc/CAP/UNBIS politics-policy hybrid coverage."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from rdflib import URIRef
from rdflib.namespace import DCTERMS, SKOS
from sqlalchemy import select, text

pytest.importorskip("rdflib", reason="taxonomy extra is not installed")
pytest.importorskip("skosify", reason="taxonomy extra is not installed")

from rumen.models import FeedItem, ItemConcept
from rumen.taxonomy.donors.cap_facet import (
    CAP_SCHEME,
    assign_policy_issue,
    build_cap_facet,
    oss_default_bundle,
)
from rumen.taxonomy.donors.eurovoc import (
    EUROVOC_0431,
    EUROVOC_POLITICS,
    RUMEN_POLITICS_POLICY,
    build_politics_policy_graft,
)
from rumen.taxonomy.donors.graft import DonorConfig, graft_donor
from rumen.taxonomy.donors.unbis_gapfill import add_unbis_gapfill
from rumen.taxonomy.materialize import materialize_scheme

_FIXTURES = Path(__file__).parent / "fixtures"


def _politics_graph():
    return build_politics_policy_graft(
        _FIXTURES / "eurovoc_politics_subset.ttl",
        included_concepts={
            EUROVOC_POLITICS,
            URIRef("http://eurovoc.europa.eu/3237"),
            URIRef("http://eurovoc.europa.eu/3517"),
        },
        reviewer="taxonomy-reviewer",
        review_date=date(2026, 7, 19),
    )


def test_eurovoc_when_imported_then_uses_0431_close_match_and_preserves_provenance(session) -> None:
    graph, mappings = _politics_graph()
    graft_donor(
        session,
        graph,
        config=DonorConfig("EuroVoc", True, "https://op.europa.eu/reuse"),
        mappings=mappings,
    )
    session.commit()

    assert (RUMEN_POLITICS_POLICY, SKOS.closeMatch, EUROVOC_0431) in graph
    assert (EUROVOC_POLITICS, DCTERMS.source, EUROVOC_POLITICS) in graph
    assert session.execute(
        text("SELECT source_uri, scheme_version FROM skos_concepts WHERE source_uri=:uri"),
        {"uri": str(EUROVOC_POLITICS)},
    ).one() == (str(EUROVOC_POLITICS), "4.24")
    row = session.execute(
        text(
            "SELECT source_scheme, source_version, target_scheme, target_version, "
            "reviewer, review_date FROM skos_relations WHERE relation_type='closeMatch'"
        )
    ).one()
    # SQLite has no native DATE type and round-trips it as text; Postgres returns a
    # real `date` object. Compare via str() so the assertion holds on both dialects.
    assert row[:5] == ("rumen-topic", "1", "eurovoc", "4.24", "taxonomy-reviewer")
    assert str(row[5]) == "2026-07-19"


def test_cap_when_rights_unconfirmed_then_private_assignment_remains_independent_and_export_omits_labels(
    session,
) -> None:
    cap = build_cap_facet(
        [{"code": "3", "label": "Health"}], version="2025.1", rights_confirmed=False
    )
    materialize_scheme(session, cap)
    item = FeedItem(
        source="test",
        source_id="cap-item",
        url="https://example.test/cap",
        title="Health policy",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    session.add(item)
    session.flush()
    assign_policy_issue(
        session,
        item_table="feed_items",
        item_id=item.id,
        concept_uri=URIRef("https://rumen.dev/ns#cap/3"),
        confidence=0.9,
        method="rule",
    )
    session.commit()

    assignment = session.scalars(select(ItemConcept).where(ItemConcept.item_id == item.id)).one()
    assert assignment.facet == "policy-issue"
    assert list(oss_default_bundle(cap, rights_confirmed=False).subjects(SKOS.prefLabel)) == []
    assert list(oss_default_bundle(cap, rights_confirmed=True).subjects(SKOS.prefLabel))
    assert not session.execute(
        text("SELECT 1 FROM skos_relations WHERE relation_type='broadMatch'")
    ).all()


def test_unbis_when_gapfill_unreviewed_then_rejects_and_reviewed_mapping_is_allowed() -> None:
    graph, _ = _politics_graph()
    with pytest.raises(ValueError, match="reviewer id and review date"):
        add_unbis_gapfill(
            graph,
            rumen_leaf=RUMEN_POLITICS_POLICY,
            unbis_uri=URIRef("http://metadata.un.org/thesaurus/1004779"),
            label="PEACEKEEPING OPERATIONS",
            source_version="2026-05-06",
            reviewer=None,
            review_date=None,
        )

    mapping = add_unbis_gapfill(
        graph,
        rumen_leaf=RUMEN_POLITICS_POLICY,
        unbis_uri=URIRef("http://metadata.un.org/thesaurus/1004779"),
        label="PEACEKEEPING OPERATIONS",
        source_version="2026-05-06",
        reviewer="taxonomy-reviewer",
        review_date=date(2026, 7, 19),
    )
    assert mapping[2].relation_type == "closeMatch"
    assert (RUMEN_POLITICS_POLICY, SKOS.closeMatch, mapping[1]) in graph
    assert (CAP_SCHEME, SKOS.broader, RUMEN_POLITICS_POLICY) not in graph
