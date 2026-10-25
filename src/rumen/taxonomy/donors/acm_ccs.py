"""ACM CCS is already SKOS/RDF: load it without a lossy conversion."""

from __future__ import annotations

from pathlib import Path
from typing import IO

from rdflib import Graph

from rumen.taxonomy.skos_store import load_skos

ACM_CCS_SOURCE_URL = "https://dl.acm.org/ccs"


def load_acm_ccs(source: str | Path | IO[bytes] | IO[str]) -> Graph:
    return load_skos(source)
