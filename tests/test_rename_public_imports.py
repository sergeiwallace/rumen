"""Supported imports remain available only from the ratified package path."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "symbols"),
    [
        ("rumen", ("__version__",)),
        ("rumen.cli", ("main", "run")),
        (
            "rumen.adapters",
            (
                "ChangeSourceAdapter",
                "ChromeReadingListAdapter",
                "ChromeReadingListError",
                "RawItem",
                "SourceAdapter",
                "SourceChange",
                "SpotifyPodcastAdapter",
                "SpotifyPodcastCacheError",
            ),
        ),
        (
            "rumen.backlog_mining.models",
            (
                "ACTIVE_CANDIDATE_STATES",
                "CANDIDATE_STATES",
                "DISPOSITION_ACTIONS",
                "BacklogCandidate",
                "BacklogCandidateDisposition",
                "BacklogCandidateEvidence",
                "BacklogCandidatePromotion",
                "BacklogCandidateRecommendation",
                "BacklogCandidateTaskLink",
                "BacklogMiningRun",
                "BacklogMappingReview",
                "BacklogTaskMatch",
            ),
        ),
        (
            "rumen.backlog_mining.projection",
            (
                "CandidateNotApprovedForProjectionError",
                "CcTaskCreationDisabledError",
                "NonCurrentSessionError",
                "build_cc_task_plan",
            ),
        ),
        (
            "rumen.backlog_mining.review",
            (
                "CandidateRenderError",
                "CandidateRepositoryError",
                "RenderedCandidate",
                "append_candidate_review_references",
                "candidate_for_identifier",
                "candidate_review",
                "render_candidate",
                "review_candidate",
            ),
        ),
        ("rumen.classifier", ("ClassificationResult", "classify_item", "classify_pending")),
        ("rumen.db", ("db_url", "get_engine", "reset_engine")),
        ("rumen.grouping", ("group_pending", "unmerge_item")),
        ("rumen.paperpile", ("PaperpileSyncError", "SyncConfig", "synchronize")),
        ("rumen.podcasts", ("CacheError", "CacheSchemaError")),
        (
            "rumen.ranking",
            (
                "InterestProfile",
                "RankInput",
                "RankedEntity",
                "half_life_for",
                "rank",
                "rank_topic_entities",
            ),
        ),
        ("rumen.taxonomy.audit", ("AuditIssue", "TaxonomyAuditResult", "run_taxonomy_audit")),
    ],
)
def test_rename_public_imports(module_name: str, symbols: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)
    assert all(hasattr(module, symbol) for symbol in symbols)
