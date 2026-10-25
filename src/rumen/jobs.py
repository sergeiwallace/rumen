"""Schedulable entrypoints — the full pipeline chain + the period digest (T-10, AD-2).

Scheduler-agnostic orchestration with exit-code and idempotency semantics.
Configure scheduling outside this package to suit your sources and digest needs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rumen.adapters.base import SourceAdapter
from rumen.adapters.youtube import YouTubePlaylistAdapter
from rumen.backlog_mining.judge import ActionabilityJudge
from rumen.backlog_mining.notifications import NotificationChannel
from rumen.backlog_mining.task_store import TaskStoreReader
from rumen.classifier.cascade import classify_pending
from rumen.classifier.metrics import fail_pass, finalize_pass, start_or_resume_pass
from rumen.classifier.tier1_embedding import DEFAULT_TAU_CLASS
from rumen.config import PipelineConfig
from rumen.digest import (
    polish_extractive_digest,
    synthesize_digest,
    synthesize_digest_extractive,
)
from rumen.grouping import group_pending
from rumen.grouping.embedding_residue import DEFAULT_TAU_HIGH, DEFAULT_TAU_LOW
from rumen.ingest import run_ingestion
from rumen.keyword_filter import apply_keyword_filter
from rumen.llm import BudgetedLlmCommand, LlmCommand
from rumen.models import (
    BacklogMiningRun,
    Digest,
    DroppedItem,
    FeedRun,
    TopicEntity,
    topic_entity_not_fully_source_deleted,
)
from rumen.ranking.interest_profile import InterestProfile, load_or_seed_profile
from rumen.ranking.rank import (
    DEFAULT_EXPLORATION,
    DEFAULT_LAMBDA,
    rank_topic_entities,
)
from rumen.ranking.rank import (
    MEDIUM_HALF_LIFE as DEFAULT_HALF_LIFE,
)
from rumen.source_maintenance import SourceMaintainer

if TYPE_CHECKING:
    # The taxonomy audit needs the optional ``taxonomy`` extra (rdflib), so it is imported only
    # where it runs; ``rumen run`` on a base install must not import it.
    from rumen.taxonomy.audit import TaxonomyAuditResult
    from rumen.taxonomy.hmc.fit import FacetFitReport

logger = logging.getLogger(__name__)


@contextmanager
def _phase(name: str):
    """Log a pipeline phase's start + duration so a stall is attributable in real time
    (the run is otherwise a black box between progress samples). INFO-level; the CLI wires
    logging to stderr."""
    logger.info("phase %s: start", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("phase %s: done in %.1fs", name, time.perf_counter() - start)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class PipelineResult:
    ingest_runs: list[FeedRun] = field(default_factory=list)
    classified: dict = field(default_factory=dict)
    filtered: dict = field(default_factory=dict)
    grouped: dict = field(default_factory=dict)
    ranked_count: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    llm_budget_tripped: bool = False
    classification_pass_id: str | None = None
    classifier_alarms: list[dict] = field(default_factory=list)
    youtube: dict[str, dict] = field(default_factory=dict)
    ok: bool = True

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1


@dataclass(slots=True)
class BacklogMiningJobResult:
    result: object | None = None
    error: BaseException | None = None
    alert_status: str | None = None


@dataclass(slots=True)
class BacklogRefreshResult:
    pipeline: PipelineResult
    digest: Digest | None
    mining: BacklogMiningJobResult


def _record_mining_alert(
    session: Session,
    error: BaseException,
    *,
    run_id: int | None,
    channel: NotificationChannel | None,
    channel_setup_error: BaseException | None,
) -> str:
    """Persist a contained operational alert outcome; never re-raise delivery errors."""
    if channel_setup_error is not None:
        status = "alert_error"
        alert_error = channel_setup_error
    elif channel is None:
        status = "not_configured"
        alert_error = None
    else:
        try:
            status = (
                "sent"
                if channel.send("Rumen backlog mining failed", type(error).__name__)
                else "delivery_failed"
            )
            alert_error = None
        except BaseException as exc:  # notification failure must not break curation
            status = "alert_error"
            alert_error = exc
    run = session.get(BacklogMiningRun, run_id) if run_id is not None else None
    if run is not None:
        payload = dict(run.error_payload or {})
        payload.update(
            {
                "alert_status": status,
                "channel_setup_error": repr(channel_setup_error) if channel_setup_error else None,
                "alert_error": repr(alert_error) if alert_error else None,
            }
        )
        run.error_payload = payload
        session.commit()
    return status


def run_backlog_mining(
    session: Session,
    *,
    config: PipelineConfig,
    judge: ActionabilityJudge | None = None,
    channel: NotificationChannel | None = None,
    channel_setup_error: BaseException | None = None,
    task_store_reader: TaskStoreReader | None = None,
    digest_period_days: int = 7,
    now: datetime | None = None,
) -> BacklogMiningJobResult:
    """Run standalone mining and contain its operational alert boundary."""
    from rumen.backlog_mining.service import BacklogMiningService

    try:
        result = BacklogMiningService(
            session, config, judge, task_store_reader=task_store_reader, channel=channel
        ).mine(now=now, digest_period_days=digest_period_days)
        return BacklogMiningJobResult(result=result)
    except BaseException as exc:
        run_id = getattr(exc, "run_id", None)
        return BacklogMiningJobResult(
            error=exc,
            alert_status=_record_mining_alert(
                session,
                exc,
                run_id=run_id,
                channel=channel,
                channel_setup_error=channel_setup_error,
            ),
        )


def run_backlog_refresh(
    session: Session,
    *,
    config: PipelineConfig,
    digest_enabled: bool = False,
    judge: ActionabilityJudge | None = None,
    channel: NotificationChannel | None = None,
    channel_setup_error: BaseException | None = None,
    task_store_reader: TaskStoreReader | None = None,
    digest_period_days: int = 7,
    local_polish: bool = False,
    local_model_id: str | None = None,
    now: datetime | None = None,
    **pipeline_kwargs,
) -> BacklogRefreshResult:
    """The one combined route: pipeline, optional digest, then mining exactly once."""
    # Job callers supply an already-open session; reject an unavailable backend before any
    # pipeline/store work. The CLI performs the matching preflight before opening that session.
    from rumen.storage import preflight_storage

    preflight_storage(config)
    pipeline_kwargs.setdefault("adapters", config.adapters)
    pipeline_kwargs.setdefault("source_maintainers", config.source_maintainers)
    pipeline_kwargs.setdefault("llm_command", config.llm_command)
    pipeline = run_pipeline(session, now=now, **pipeline_kwargs)
    digest = None
    if digest_enabled:
        end = now or _utcnow()
        digest = run_digest(
            session,
            llm_command=config.llm_command,
            period_start=end - timedelta(days=digest_period_days),
            period_end=end,
            top_n=config.digest_top_n,
            section_map=config.digest_section_map,
            local_polish=local_polish,
            local_model_id=local_model_id,
        )
    mining = run_backlog_mining(
        session,
        config=config,
        judge=judge,
        channel=channel,
        channel_setup_error=channel_setup_error,
        task_store_reader=task_store_reader,
        digest_period_days=digest_period_days,
        now=now,
    )
    return BacklogRefreshResult(pipeline=pipeline, digest=digest, mining=mining)


def run_pipeline(
    session: Session,
    *,
    adapters: list[SourceAdapter],
    source_maintainers: list[SourceMaintainer] | None = None,
    reports: dict[str, FacetFitReport] | None = None,
    llm_command: LlmCommand | None = None,
    profile: InterestProfile | None = None,
    now: datetime | None = None,
    tau_class: float = DEFAULT_TAU_CLASS,
    tau_high: float = DEFAULT_TAU_HIGH,
    tau_low: float = DEFAULT_TAU_LOW,
    mmr_lambda: float = DEFAULT_LAMBDA,
    exploration_fraction: float = DEFAULT_EXPLORATION,
    half_life_days: dict[str, float] | None = None,
    default_half_life_days: float = DEFAULT_HALF_LIFE,
    out_of_scope: frozenset[str] = frozenset(),
    max_llm_calls: int = 200,
    commit_batch_size: int = 500,
    classify_write_max_attempts: int = 3,
    classify_write_retry_backoff_seconds: float = 1.0,
    resolve_redirect: Callable[[str], str] | None = None,
    residue_window_days: int = 30,
    classifier_path: str | None = None,
    classifier_escalation_alarm_threshold: float | None = None,
    transcript_provider=None,
) -> PipelineResult:
    """Run sync → classify → keyword-filter → group → rank. Idempotent (T-03/T-04 dedup).

    A failed adapter records a ``failed`` feed_run without advancing its high-water mark,
    and the result reports ``ok=False`` (``exit_code`` 1) so a scheduler surfaces it — but
    a retry never double-ingests (dedup + last-completed-cursor hold). All tuning thresholds
    (tau_*, MMR, exploration, half-lives) are caller-supplied (config-driven, no magic numbers).
    """
    now = now or _utcnow()
    result = PipelineResult()
    if profile is None:
        profile = load_or_seed_profile(session)

    maintainers = {maintainer.source_name(): maintainer for maintainer in source_maintainers or []}
    with _phase("ingest"):
        for adapter in adapters:
            maintainer = maintainers.get(adapter.source_name())
            if maintainer is not None:
                with _phase(f"maintain:{adapter.source_name()}"):
                    maintenance = maintainer.maintain(session)
                if maintenance.status != "completed":
                    result.ok = False
                    continue
            with _phase(f"ingest:{adapter.source_name()}"):
                run = run_ingestion(session, adapter, commit_batch_size=commit_batch_size)
            result.ingest_runs.append(run)
            if run.status == "failed":
                result.ok = False
                continue
            if isinstance(adapter, YouTubePlaylistAdapter):
                from rumen.youtube_enrichment import (
                    YouTubeTranscriptApiProvider,
                    reconcile_youtube_discovery,
                    run_transcript_enrichment,
                    run_youtube_maintenance,
                )

                lifecycle = reconcile_youtube_discovery(session, adapter, now=now)
                refresh = run_youtube_maintenance(session, adapter, now=now)
                lifecycle.metadata_refreshed += refresh.metadata_refreshed
                lifecycle.metadata_failures += refresh.metadata_failures
                lifecycle.warnings.extend(refresh.warnings)
                if adapter.transcript_mode == "unofficial_captions":
                    provider = transcript_provider or YouTubeTranscriptApiProvider()
                    enrichment = run_transcript_enrichment(session, adapter, provider, now=now)
                    lifecycle.transcript_created += enrichment.transcript_created
                    lifecycle.transcript_not_found += enrichment.transcript_not_found
                    lifecycle.transcript_retries += enrichment.transcript_retries
                    lifecycle.transcript_blocked += enrichment.transcript_blocked
                    lifecycle.warnings.extend(enrichment.warnings)
                result.youtube[adapter.source_name()] = {
                    "states_created": lifecycle.states_created,
                    "metadata_refreshed": lifecycle.metadata_refreshed,
                    "metadata_failures": lifecycle.metadata_failures,
                    "removals": lifecycle.removals,
                    "transcript_created": lifecycle.transcript_created,
                    "transcript_not_found": lifecycle.transcript_not_found,
                    "transcript_retries": lifecycle.transcript_retries,
                    "transcript_blocked": lifecycle.transcript_blocked,
                    "warnings": lifecycle.warnings,
                }

    # Safeguard: Tier-2 work with no Tier-0 route can scale with a legitimate catch-up batch,
    # but a fallthrough from a deterministic Tier-0 route is capped + repeated failures open
    # the breaker. This catches the source-type regression that sent too many items to the LLM.
    budgeted = BudgetedLlmCommand(llm_command, max_calls=max_llm_calls) if llm_command else None
    snapshot = start_or_resume_pass(
        session, reports=reports, tau_class=tau_class, classifier_path=classifier_path
    )
    result.classification_pass_id = snapshot.classification_pass_id
    session.commit()
    try:
        with _phase("classify"):
            result.classified = classify_pending(
                session,
                reports=reports,
                llm_command=budgeted,
                tau_class=tau_class,
                out_of_scope=out_of_scope,
                commit_batch_size=commit_batch_size,
                classify_write_max_attempts=classify_write_max_attempts,
                classify_write_retry_backoff_seconds=classify_write_retry_backoff_seconds,
                metric_snapshot=snapshot,
            )
        finalize_pass(session, snapshot)
        session.commit()
    except Exception as exc:  # noqa: BLE001 - a failed pass must be rolled back and recorded.
        session.rollback()
        persisted = session.get(type(snapshot), snapshot.id)
        if persisted is not None:
            fail_pass(session, persisted, exc)
        result.ok = False
        return result
    if classifier_escalation_alarm_threshold is not None:
        from rumen.classifier.metrics import snapshot_payload

        result.classifier_alarms = snapshot_payload(
            snapshot, threshold=classifier_escalation_alarm_threshold
        )["alarms"]
    if budgeted is not None:
        result.llm_calls = budgeted.calls
        result.llm_failures = budgeted.failures
        result.llm_budget_tripped = budgeted.breaker_open or (
            budgeted.unexpected_calls >= budgeted.max_calls
        )
    with _phase("keyword_filter"):
        result.filtered = apply_keyword_filter(session, commit_batch_size=commit_batch_size)
    with _phase("group"):
        result.grouped = group_pending(
            session,
            tau_high=tau_high,
            tau_low=tau_low,
            commit_batch_size=commit_batch_size,
            resolve_redirect=resolve_redirect,
            residue_window_days=residue_window_days,
            now=now,
        )
    with _phase("rank"):
        ranked = rank_topic_entities(
            session,
            profile,
            now=now,
            lambda_mmr=mmr_lambda,
            exploration_fraction=exploration_fraction,
            half_life_days=half_life_days,
            default_half_life_days=default_half_life_days,
            commit_batch_size=commit_batch_size,
        )
    result.ranked_count = len(ranked)
    return result


def _existing_digest(
    session: Session, period_start: datetime, period_end: datetime
) -> Digest | None:
    return session.scalars(
        select(Digest).where(Digest.period_start == period_start, Digest.period_end == period_end)
    ).first()


def run_digest(
    session: Session,
    *,
    period_start: datetime,
    period_end: datetime,
    llm_command: LlmCommand | None = None,
    top_n: int = 20,
    force: bool = False,
    extractive: bool = True,
    section_map: dict[str, str] | None = None,
    model_used: str = "unspecified",
    local_polish: bool = False,
    local_model_id: str | None = None,
) -> Digest:
    """Produce exactly one digest per period. Re-invocation is a no-op unless ``force``.

    Defaults to the **extractive** zero-LLM path (D-17) — the digest no longer hard-depends
    on a hosted LLM. ``extractive=False`` uses the LLM-synthesis path and requires
    ``llm_command`` (raises ``ValueError`` otherwise)."""
    existing = _existing_digest(session, period_start, period_end)
    if existing is not None and not force:
        return existing

    entity_ids = list(
        session.scalars(
            select(TopicEntity.id)
            .where(topic_entity_not_fully_source_deleted())
            # Portable NULLS-LAST: ranked entities (is_(None) → False) sort first.
            .order_by(TopicEntity.rank_position.is_(None), TopicEntity.rank_position.asc())
            .limit(top_n)
        )
    )
    suppressed_count = session.scalar(
        select(func.count(DroppedItem.id)).where(
            DroppedItem.dropped_at >= period_start, DroppedItem.dropped_at < period_end
        )
    )
    if extractive:
        digest = synthesize_digest_extractive(
            session,
            entity_ids=entity_ids,
            period_start=period_start,
            period_end=period_end,
            llm_command=llm_command,
            section_map=section_map,
            suppressed_count=suppressed_count or 0,
        )
        if local_polish and llm_command is not None and local_model_id is not None:
            return polish_extractive_digest(
                session, digest=digest, llm_command=llm_command, model_used=local_model_id
            )
        return digest
    if llm_command is None:
        raise ValueError("run_digest(extractive=False) requires an llm_command")
    return synthesize_digest(
        session,
        llm_command=llm_command,
        entity_ids=entity_ids,
        period_start=period_start,
        period_end=period_end,
        suppressed_count=suppressed_count or 0,
        model_used=model_used,
    )


def run_taxonomy_audit(
    session: Session,
    *,
    llm_command: LlmCommand | None,
    crowded_item_threshold: int = 100,
    sparse_item_threshold: int = 0,
    duplicate_similarity_threshold: float = 0.92,
    audit_run_id: str | None = None,
) -> TaxonomyAuditResult:
    """Run the separate D-23 scheduled maintenance job (never part of ``rumen run``)."""
    from rumen.taxonomy.audit import run_taxonomy_audit as _run_taxonomy_audit

    with _phase("taxonomy-audit"):
        return _run_taxonomy_audit(
            session,
            llm_command=llm_command,
            crowded_item_threshold=crowded_item_threshold,
            sparse_item_threshold=sparse_item_threshold,
            duplicate_similarity_threshold=duplicate_similarity_threshold,
            audit_run_id=audit_run_id,
        )
