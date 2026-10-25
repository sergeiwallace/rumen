"""Deterministic opaque concept identifiers, independent of any RDF library.

Extracted from ``materialize.py`` so migration-time code (the legacy category
crosswalk, revision 0008) can compute concept ids without requiring the optional
``taxonomy`` extra's ``rdflib`` dependency — the base engine and OSS on-ramp stay
light (D-24).
"""

from __future__ import annotations

from hashlib import sha256


def stable_concept_id(uri: str) -> int:
    # The schema's INTEGER primary key is Postgres int4, so keep the deterministic opaque ID
    # in its signed range.  RDF URIs remain the authoring identifiers; this id is runtime-only.
    return int.from_bytes(sha256(str(uri).encode()).digest()[:4], "big") & 0x7FFFFFFF
