"""T-13 Wikidata P279 graft enrichment coverage."""

from __future__ import annotations

from datetime import UTC, datetime

from rdflib import URIRef
from rdflib.namespace import SKOS

from rumen.models import FeedItem
from rumen.taxonomy.donors.wikidata_link import WikidataEntity, link_wikidata_entity


def _item() -> FeedItem:
    return FeedItem(
        source="test",
        source_id="wikidata",
        url="https://example.test/wikidata",
        title="Machine learning",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
    )


def test_wikidata_link_when_entity_resolves_then_walks_p279_to_depth_cap_and_uses_stable_id() -> (
    None
):
    entities = {
        "Q1": WikidataEntity("Q1", "Machine learning", ("Q2",)),
        "Q2": WikidataEntity("Q2", "Artificial intelligence", ("Q3",)),
        "Q3": WikidataEntity("Q3", "Computer science", ()),
    }
    item = _item()
    graph = link_wikidata_entity(
        item,
        qid="Q1",
        grafted_branch=URIRef("https://rumen.dev/ns#topic/science"),
        resolver=entities.get,
        depth_cap=1,
    )

    assert item.stable_id_kind == "wikidata"
    assert item.stable_id_value == "Q1"
    assert graph is not None
    assert (
        URIRef("https://www.wikidata.org/entity/Q1"),
        SKOS.broadMatch,
        URIRef("https://rumen.dev/ns#topic/science"),
    ) in graph
    assert (
        URIRef("https://www.wikidata.org/entity/Q1"),
        SKOS.broadMatch,
        URIRef("https://www.wikidata.org/entity/Q2"),
    ) in graph
    assert URIRef("https://www.wikidata.org/entity/Q3") not in graph.subjects()


def test_wikidata_link_when_lookup_is_absent_then_leaves_item_unlinked() -> None:
    item = _item()

    assert (
        link_wikidata_entity(
            item,
            qid="Q404",
            grafted_branch=URIRef("https://rumen.dev/ns#topic/science"),
            resolver=lambda _qid: None,
        )
        is None
    )
    assert item.stable_id_kind is None
    assert item.stable_id_value is None
