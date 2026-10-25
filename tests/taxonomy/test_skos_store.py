"""T-12: RDF/SKOS load and dump acceptance coverage."""

from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest

pytest.importorskip("rdflib", reason="taxonomy extra is not installed")
pytest.importorskip("skosify", reason="taxonomy extra is not installed")

from rdflib import URIRef
from rdflib.namespace import RDF, SKOS

from rumen.taxonomy.skos_store import dump_skos, load_skos


@pytest.mark.parametrize(
    "url", ["http://169.254.169.254/latest/meta-data/", "https://localhost/context"]
)
@pytest.mark.parametrize("kind", ["context", "nested", "import", "source"])
def test_given_remote_taxonomy_reference_when_loaded_then_fails_without_network(
    monkeypatch, url, kind
):
    opener = Mock(side_effect=AssertionError("unexpected network request"))
    monkeypatch.setattr("urllib.request.OpenerDirector.open", opener)
    if kind == "source":
        source = url
    else:
        context = {"@import": url} if kind == "import" else url
        document = {"@context": context}
        if kind == "nested":
            document = {"@context": {"term": {"@id": "https://example.test/term", **document}}}
        source = io.StringIO(json.dumps(document))
    with pytest.raises(ValueError, match="(?i)(remote|network).*(disabled|not supported)"):
        load_skos(source, format="json-ld")
    opener.assert_not_called()


@pytest.mark.parametrize("kind", ["inline", "stream", "path", "string_path"])
def test_given_embedded_context_when_loaded_then_imports_without_network(
    monkeypatch, tmp_path, kind
):
    document = json.dumps(
        {
            "@context": {"skos": str(SKOS)},
            "@id": "https://example.test/topic",
            "@type": "skos:Concept",
            "skos:prefLabel": "Topic",
        }
    )
    source = document
    if kind == "stream":
        source = io.StringIO(document)
    elif kind in {"path", "string_path"}:
        source = tmp_path / "taxonomy.jsonld"
        source.write_text(document)
        if kind == "string_path":
            source = str(source)
    opener = Mock(side_effect=AssertionError("unexpected network request"))
    monkeypatch.setattr("urllib.request.OpenerDirector.open", opener)
    graph = load_skos(source, format="json-ld" if kind in {"inline", "stream"} else None)
    assert (URIRef("https://example.test/topic"), RDF.type, SKOS.Concept) in graph
    opener.assert_not_called()


_TURTLE = """
@prefix ex: <https://example.test/taxonomy/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

ex:scheme a skos:ConceptScheme ; skos:prefLabel "Example scheme" .
ex:parent a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Parent" .
ex:child a skos:Concept ; skos:inScheme ex:scheme ; skos:prefLabel "Child" ;
    skos:broader ex:parent ; skos:related ex:parent .
ex:parent skos:narrower ex:child .
"""


def test_load_skos_when_well_formed_then_preserves_concepts_labels_and_relations() -> None:
    graph = load_skos(_TURTLE, format="turtle")

    parent = URIRef("https://example.test/taxonomy/parent")
    child = URIRef("https://example.test/taxonomy/child")
    assert (None, RDF.type, SKOS.Concept) in graph
    assert (None, SKOS.prefLabel, None) in graph
    assert len(list(graph.triples((None, RDF.type, SKOS.Concept)))) == 2
    assert (child, SKOS.broader, parent) in graph
    assert (parent, SKOS.narrower, child) in graph
    assert (child, SKOS.related, parent) in graph


def test_dump_skos_when_loaded_then_roundtrips_rdf() -> None:
    original = load_skos(_TURTLE, format="turtle")
    reloaded = load_skos(dump_skos(original), format="turtle")

    assert set(original) == set(reloaded)
