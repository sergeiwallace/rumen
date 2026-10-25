"""Best-effort Wikidata P279 enrichment for already-classified corpus items."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import URLError

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

from rumen.models import FeedItem, TopicEntity
from rumen.net_safety import UnsafeTarget, open_public_request
from rumen.taxonomy.donors.eurovoc import RUMEN

WIKIDATA_SCHEME = URIRef(f"{RUMEN}scheme/wikidata")


@dataclass(frozen=True)
class WikidataEntity:
    qid: str
    label: str
    parents: tuple[str, ...]


def fetch_wikidata_entity(qid: str, *, timeout: float = 5.0) -> WikidataEntity | None:
    """Read one entity from Wikidata; network failures are deliberately non-fatal."""
    try:
        with open_public_request(
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json", timeout=timeout
        ) as response:
            entity = json.load(response)["entities"][qid]
    except (KeyError, OSError, TimeoutError, URLError, UnsafeTarget, json.JSONDecodeError):
        return None
    label = entity.get("labels", {}).get("en", {}).get("value", qid)
    parents = tuple(
        claim["mainsnak"]["datavalue"]["value"]["id"]
        for claim in entity.get("claims", {}).get("P279", [])
        if claim.get("mainsnak", {}).get("snaktype") == "value"
        and claim["mainsnak"].get("datavalue", {}).get("type") == "wikibase-entityid"
    )
    return WikidataEntity(qid, label, parents)


def link_wikidata_entity(
    item: FeedItem | TopicEntity,
    *,
    qid: str,
    grafted_branch: URIRef,
    resolver: Callable[[str], WikidataEntity | None] = fetch_wikidata_entity,
    depth_cap: int = 3,
) -> Graph | None:
    """Walk P279 to a depth cap and produce broadMatch links; absent is a no-op."""
    first = resolver(qid)
    if first is None:
        return None
    item.stable_id_kind = "wikidata"
    item.stable_id_value = qid
    graph = Graph()
    graph.add((WIKIDATA_SCHEME, RDF.type, SKOS.ConceptScheme))
    graph.add((WIKIDATA_SCHEME, DCTERMS.identifier, Literal("wikidata")))
    graph.add((WIKIDATA_SCHEME, URIRef(f"{RUMEN}facet"), Literal("topic")))
    graph.add((WIKIDATA_SCHEME, URIRef(f"{RUMEN}schemeVersion"), Literal("live")))
    seen: set[str] = set()
    current = [first]
    for depth in range(depth_cap + 1):
        next_level: list[WikidataEntity] = []
        for entity in current:
            if entity.qid in seen:
                continue
            seen.add(entity.qid)
            concept = URIRef(f"https://www.wikidata.org/entity/{entity.qid}")
            graph.add((concept, RDF.type, SKOS.Concept))
            graph.add((concept, SKOS.inScheme, WIKIDATA_SCHEME))
            graph.add((concept, SKOS.prefLabel, Literal(entity.label)))
            graph.add((concept, DCTERMS.source, concept))
            if depth == 0:
                graph.add((concept, SKOS.broadMatch, grafted_branch))
            if depth < depth_cap:
                for parent_qid in entity.parents:
                    parent = resolver(parent_qid)
                    if parent is not None:
                        parent_uri = URIRef(f"https://www.wikidata.org/entity/{parent.qid}")
                        graph.add((concept, SKOS.broadMatch, parent_uri))
                        next_level.append(parent)
        current = next_level
    return graph
