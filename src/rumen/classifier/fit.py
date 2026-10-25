"""Fit and load the per-facet HMC classifier bundle used by the live cascade."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from rumen.embeddings import EmbeddingProvider
from rumen.taxonomy.hmc.fit import (
    FacetFitReport,
    fit_all_facets_from_session,
    load_report_bundle,
    save_report_bundle,
)


def fit_from_session(
    session: Session,
    *,
    provider: EmbeddingProvider | None = None,
    path_consistency_floor: float = 0.75,
) -> dict[str, FacetFitReport]:
    """Fit bootstrap-present facets; flat feed-item categories are intentionally ignored."""
    reports = fit_all_facets_from_session(
        session, provider=provider, path_consistency_floor=path_consistency_floor
    )
    if not reports:
        raise ValueError("no bootstrap-present facets to fit")
    return reports


def save_classifier(reports: dict[str, FacetFitReport], path: str | Path) -> None:
    """Compatibility-named configuration seam; persists an HMC directory bundle."""
    save_report_bundle(reports, path)


def load_classifier(
    path: str | Path, provider: EmbeddingProvider | None = None
) -> dict[str, FacetFitReport] | None:
    """Load the HMC report bundle; no flat artifact fallback exists."""
    return load_report_bundle(path, provider=provider)
