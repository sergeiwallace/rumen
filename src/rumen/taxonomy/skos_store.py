"""RDF/SKOS source-of-truth loading and dumping helpers (D-24)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit

from rdflib import Graph
from rdflib.util import guess_format

_LOCAL_FORMATS = {
    "turtle",
    "ttl",
    "n3",
    "nt",
    "ntriples",
    "trig",
    "nquads",
    "xml",
    "trix",
    "json-ld",
}


def _check_contexts(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "@import" or (key == "@context" and _has_remote_context(child)):
                raise ValueError("Remote JSON-LD contexts are disabled; embed the context instead")
            _check_contexts(child)
    elif isinstance(value, list):
        for child in value:
            _check_contexts(child)


def _has_remote_context(value: object) -> bool:
    return isinstance(value, str) or (
        isinstance(value, list) and any(_has_remote_context(child) for child in value)
    )


def load_skos(source: str | Path | IO[bytes] | IO[str], *, format: str | None = None) -> Graph:
    """Load an RDF/SKOS document into an in-memory :class:`rdflib.Graph`.

    ``source`` may be RDF text, a local filesystem path, or a file-like object.
    Network sources and remote JSON-LD contexts are disabled. Embed contexts in
    JSON-LD documents and download remote taxonomies separately before importing.
    """
    location = None
    if hasattr(source, "read"):
        data = source.read()
    elif isinstance(source, str) and (
        "\n" in source or source.lstrip().startswith(("@prefix", "PREFIX", "<", "{", "["))
    ):
        data = source
    else:
        if isinstance(source, str) and (urlsplit(source).scheme or source.startswith("//")):
            raise ValueError("Remote taxonomy sources are disabled; use a local file")
        location = Path(source).resolve()
        data = location.read_bytes()
    format = format or (guess_format(str(location)) if location else None) or "turtle"
    format = {
        "application/ld+json": "json-ld",
        "application/rdf+xml": "xml",
        "text/turtle": "turtle",
    }.get(format, format)
    if format not in _LOCAL_FORMATS:
        raise ValueError("Network-capable or unknown RDF formats are not supported")
    if format == "json-ld":
        _check_contexts(json.loads(data))
    graph = Graph()
    graph.parse(data=data, format=format, publicID=location.as_uri() if location else None)
    return graph


def dump_skos(
    graph: Graph,
    destination: str | Path | IO[bytes] | IO[str] | None = None,
    *,
    format: str = "turtle",
) -> str:
    """Serialize a SKOS graph, returning RDF text when no destination is supplied."""
    serialized: Any = graph.serialize(
        destination=str(destination) if isinstance(destination, Path) else destination,
        format=format,
    )
    if serialized is None:
        return ""
    return serialized.decode() if isinstance(serialized, bytes) else str(serialized)
