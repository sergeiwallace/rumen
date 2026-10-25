"""Validated donor grafting and reviewed SKOS crosswalk persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from rdflib import Graph, URIRef
from sqlalchemy import text
from sqlalchemy.orm import Session

from rumen.taxonomy.materialize import (
    MaterializationResult,
    materialize_scheme,
    stable_concept_id,
)
from rumen.taxonomy.validate import TaxonomyValidationError

logger = logging.getLogger(__name__)


class DonorConfigurationError(ValueError):
    """Raised when a configured donor has no documented permission to reuse."""


@dataclass(frozen=True)
class DonorConfig:
    name: str
    reuse_terms_verified: bool
    reuse_terms_url: str | None = None


@dataclass(frozen=True)
class MappingMetadata:
    source_scheme: str
    source_version: str
    target_scheme: str
    target_version: str
    relation_type: str
    reviewer: str
    review_date: date


@dataclass(frozen=True)
class GraftResult:
    materialization: MaterializationResult | None
    skipped: bool = False


def validate_donor_config(config: DonorConfig) -> None:
    """Fail closed before an unverified donor can reach the taxonomy store."""
    if not config.reuse_terms_verified or not config.reuse_terms_url:
        raise DonorConfigurationError(
            f"donor {config.name!r} is missing documented reuse-terms verification"
        )


def graft_donor(
    session: Session,
    graph: Graph,
    *,
    config: DonorConfig,
    mappings: list[tuple[URIRef, URIRef, MappingMetadata]],
) -> GraftResult:
    """Materialize one whole validated donor graph, then retain mapping provenance.

    Validation happens inside ``materialize_scheme`` before its first write.  A bad
    source therefore leaves an existing, coarser spine untouched instead of producing
    a partially grafted branch.
    """
    validate_donor_config(config)
    try:
        result = materialize_scheme(session, graph)
    except TaxonomyValidationError as exc:
        logger.warning("skipping donor %s after SKOS validation failure: %s", config.name, exc)
        return GraftResult(None, skipped=True)
    for source, target, metadata in mappings:
        _record_mapping(session, source, target, metadata)
    return GraftResult(result)


def _record_mapping(
    session: Session,
    source: URIRef,
    target: URIRef,
    metadata: MappingMetadata,
) -> None:
    if metadata.relation_type not in {"broadMatch", "closeMatch", "exactMatch"}:
        raise ValueError(f"unsupported crosswalk relation type: {metadata.relation_type}")
    updated = session.execute(
        text(
            "UPDATE skos_relations SET source_scheme=:source_scheme, "
            "source_version=:source_version, target_scheme=:target_scheme, "
            "target_version=:target_version, reviewer=:reviewer, review_date=:review_date "
            "WHERE source_concept_id=:source_id AND target_concept_id=:target_id "
            "AND relation_type=:relation_type"
        ),
        {
            "source_scheme": metadata.source_scheme,
            "source_version": metadata.source_version,
            "target_scheme": metadata.target_scheme,
            "target_version": metadata.target_version,
            "reviewer": metadata.reviewer,
            "review_date": metadata.review_date,
            "source_id": stable_concept_id(source),
            "target_id": stable_concept_id(target),
            "relation_type": metadata.relation_type,
        },
    )
    if updated.rowcount != 1:
        raise ValueError("crosswalk relation must be present in the validated donor graph")
