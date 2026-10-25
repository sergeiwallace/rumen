"""``rumen`` console entrypoint (design Stage 7 / T-09, AD-2).

Scheduler- and host-agnostic engine CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from rumen.backlog_mining.notifications import NotificationChannel
from rumen.mcp_tools import (
    get_feed_digest,
    get_feed_items,
    rate_feed_entity,
)
from rumen.taxonomy.corrections import record_classification_review
from rumen.views import UnknownViewError, stale_facet_concept_ids

NotificationChannelResolver = Callable[[], tuple[NotificationChannel | None, BaseException | None]]


def _resolve_notification_channel(
    resolver: NotificationChannelResolver | None,
) -> tuple[NotificationChannel | None, BaseException | None]:
    """Receive an already-resolved deployment channel without importing deployment code."""
    if resolver is None:
        return None, None
    try:
        return resolver()
    except BaseException as exc:
        return None, exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rumen", description="rumen engine CLI")
    sub = parser.add_subparsers(dest="group", required=True)
    feed = sub.add_parser("feed", help="feed views + digest").add_subparsers(
        dest="command", required=True
    )

    items = feed.add_parser("items", help="list items for a view or ad-hoc filter")
    items.add_argument("--view")
    items.add_argument("--category")
    items.add_argument("--source")
    items.add_argument("--since")
    items.add_argument("--limit", type=int)

    feed.add_parser("digest", help="show the latest digest (JSON)")
    feed.add_parser("status", help="ingestion run status")

    reddit = sub.add_parser("reddit", help="Reddit source operations").add_subparsers(
        dest="reddit_command", required=True
    )
    purge = reddit.add_parser("purge", help="permanently remove one Reddit profile")
    purge.add_argument("--profile", required=True)
    purge.add_argument("--confirm", action="store_true")
    purge.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))

    rate = feed.add_parser("rate", help="rate a topic entity (-1/0/1)")
    rate.add_argument("entity_id", type=int)
    rate.add_argument("rating", type=int)

    classify = sub.add_parser(
        "classify", help="taxonomy classification review tools"
    ).add_subparsers(dest="classify_command", required=True)
    review = classify.add_parser("review", help="accept a complete human item/facet review")
    review.add_argument("item_table", choices=("feed_items", "topic_entities"))
    review.add_argument("item_id", type=int)
    review.add_argument("facet")
    review.add_argument("--concept", dest="concept_ids", action="append", type=int, required=True)
    review.add_argument("--actor", required=True)
    review.add_argument("--rationale", required=True)
    review.add_argument("--route", choices=("route", "keep"))
    review.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    metrics = classify.add_parser("metrics", help="show completed classifier telemetry snapshots")
    metrics.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    metrics.add_argument("--since", help="inclusive ISO-8601 snapshot start")
    metrics.add_argument("--until", help="inclusive ISO-8601 snapshot start")

    run_p = sub.add_parser("run", help="run the full sync→classify→ingest→group→rank chain")
    run_p.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    run_p.add_argument(
        "--digest",
        action="store_true",
        help="also synthesize the period digest after the run (end-to-end refresh + digest)",
    )

    backlog = sub.add_parser("backlog", help="backlog-mining operations").add_subparsers(
        dest="backlog_command", required=True
    )
    mine = backlog.add_parser("mine", help="run standalone backlog mining")
    mine.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    list_candidates = backlog.add_parser("list", help="list candidate drafts and approved handoffs")
    list_candidates.add_argument("--state")
    show_candidate = backlog.add_parser("show", help="show one candidate's review evidence")
    show_candidate.add_argument("candidate")
    review_candidate = backlog.add_parser("review", help="record one candidate disposition")
    review_candidate.add_argument("candidate")
    review_candidate.add_argument("--action", required=True)
    review_candidate.add_argument("--actor", required=True)
    review_candidate.add_argument("--rationale", required=True)
    review_candidate.add_argument("--create-cc-task", action="store_true")
    review_candidate.add_argument("--cc-session")
    review_candidate.add_argument(
        "--promote-as", choices=("new", "related", "shared-epic", "update", "status")
    )
    review_candidate.add_argument("--existing-task")
    review_candidate.add_argument("--operation-id")
    review_candidate.add_argument("--body-file")
    review_candidate.add_argument(
        "--task-status", choices=("open", "in_progress", "blocked", "deferred", "closed")
    )
    review_candidate.add_argument("--reason")
    review_candidate.add_argument("--new-task-title")
    review_candidate.add_argument("--new-task-description-file")
    review_candidate.add_argument("--new-task-priority", type=int, choices=range(5))
    review_candidate.add_argument("--new-task-status")
    review_candidate.add_argument("--epic-title")
    review_candidate.add_argument("--epic-description-file")
    review_candidate.add_argument("--epic-priority", type=int, choices=range(5))
    review_candidate.add_argument("--epic-status")
    review_candidate.add_argument("--approved-plan")
    review_candidate.add_argument("--confirm-plan")
    review_candidate.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    run_p.add_argument(
        "--local-polish",
        action="store_true",
        help="after --digest, locally polish extractive prose when this AI_HOST has a local model",
    )
    run_p.add_argument(
        "--digest-period-days",
        type=int,
        default=7,
        help="digest look-back window in days when --digest is set (default 7)",
    )

    dig_p = sub.add_parser("digest", help="synthesize the period digest")
    dig_p.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    dig_p.add_argument("--period-start")
    dig_p.add_argument("--period-end")
    dig_p.add_argument("--force", action="store_true")
    dig_p.add_argument(
        "--abstractive",
        action="store_true",
        help="use LLM synthesis instead of the default zero-LLM extractive path (needs llm_command)",
    )
    dig_p.add_argument(
        "--local-polish",
        action="store_true",
        help="locally polish extractive prose when this AI_HOST has a local model",
    )

    fit_p = sub.add_parser(
        "fit-classifier", help="fit + persist the per-facet HMC classifier bundle"
    )
    fit_p.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    fit_p.add_argument("--out", help="artifact path (overrides config classifier_path)")
    fit_p.add_argument("--holdout", type=float, default=0.2, help="held-out fraction (default 0.2)")
    fit_p.add_argument(
        "--target-precision",
        type=float,
        default=0.90,
        help="tau_class calibration target precision (default 0.90, per D-15)",
    )

    audit_p = sub.add_parser(
        "taxonomy-audit", help="run the separate D-23 taxonomy self-healing maintenance audit"
    )
    audit_p.add_argument("--config", default=os.environ.get("RUMEN_CONFIG"))
    audit_p.add_argument(
        "--rollback-revision",
        type=int,
        help="restore one audit revision by id instead of running a new audit",
    )
    return parser


def run(
    argv: list[str],
    session: Session,
    *,
    notification_channel_resolver: NotificationChannelResolver | None = None,
) -> int:
    """Execute a CLI invocation against ``session``. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    return _run_parsed(args, session, notification_channel_resolver=notification_channel_resolver)


def _run_parsed(
    args: argparse.Namespace,
    session: Session,
    *,
    notification_channel_resolver: NotificationChannelResolver | None = None,
) -> int:
    if args.group == "run":
        return _cmd_run(args, session, notification_channel_resolver=notification_channel_resolver)
    if args.group == "digest":
        return _cmd_digest(args, session)
    if args.group == "fit-classifier":
        return _cmd_fit_classifier(args, session)
    if args.group == "taxonomy-audit":
        return _cmd_taxonomy_audit(args, session)
    if args.group == "backlog":
        return _cmd_backlog(
            args, session, notification_channel_resolver=notification_channel_resolver
        )
    if args.group == "reddit":
        return _cmd_reddit(args, session)
    if args.group == "classify":
        return _cmd_classify(args, session)
    if args.command == "items":
        try:
            items = get_feed_items(
                session,
                view=args.view,
                category=args.category,
                source=args.source,
                since=datetime.fromisoformat(args.since) if args.since else None,
                limit=args.limit,
            )
        except UnknownViewError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(items, indent=2))
        return 0
    if args.command == "digest":
        digest = get_feed_digest(session)
        if digest is None:
            print("error: no digest available", file=sys.stderr)
            return 1
        print(json.dumps(digest, indent=2))
        return 0
    if args.command == "status":
        from sqlalchemy import select

        from rumen.models import FeedRun, FeedView, SourceMaintenanceRun

        runs = session.scalars(select(FeedRun).order_by(FeedRun.started_at.desc()).limit(20))
        for run_row in runs:
            warn = f"  [{run_row.error_message}]" if run_row.error_message else ""
            print(f"{run_row.source}\t{run_row.status}\thwm={run_row.high_water_mark}{warn}")
        maintenance = session.scalars(
            select(SourceMaintenanceRun).order_by(SourceMaintenanceRun.started_at.desc()).limit(20)
        )
        for row in maintenance:
            print(
                f"{row.source}\tmaintenance={row.status}\tchecked={row.items_checked}\tpurged={row.items_purged}"
            )
        views = session.scalars(select(FeedView).order_by(FeedView.name))
        for view_row in views:
            stale_ids = sorted(stale_facet_concept_ids(session, view_row.filter_json))
            if stale_ids:
                warn = f"  [stale filter: concept ids {', '.join(map(str, stale_ids))}]"
                print(f"view:{view_row.name}{warn}")
        return 0
    if args.command == "rate":
        try:
            result = rate_feed_entity(session, args.entity_id, args.rating)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result))
        return 0
    return 2


def _cmd_reddit(args, session: Session) -> int:
    if args.reddit_command != "purge":
        return 2
    if not args.confirm:
        print("error: rumen reddit purge requires --confirm", file=sys.stderr)
        return 1
    if not args.config:
        print(
            "error: --config (or RUMEN_CONFIG) is required for `rumen reddit purge`",
            file=sys.stderr,
        )
        return 1
    from rumen.config import load_config

    cfg = load_config(args.config)
    source = f"reddit:{args.profile}"
    maintainer = next(
        (item for item in cfg.source_maintainers if item.source_name() == source), None
    )
    if maintainer is None:
        print("error: Reddit profile is not enabled in configuration", file=sys.stderr)
        return 1
    removed = maintainer.terminate(session)
    print(json.dumps({"source": source, "items_purged": removed}))
    return 0


def _cmd_classify(args, session: Session) -> int:
    if args.classify_command == "metrics":
        if not args.config:
            print(
                "error: --config (or RUMEN_CONFIG) is required for `rumen classify metrics`",
                file=sys.stderr,
            )
            return 1
        from rumen.classifier.metrics import snapshots_in_range
        from rumen.config import load_config

        cfg = load_config(args.config)
        try:
            since = datetime.fromisoformat(args.since) if args.since else None
            until = datetime.fromisoformat(args.until) if args.until else None
        except ValueError as exc:
            print(f"error: invalid ISO-8601 timestamp: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                snapshots_in_range(
                    session,
                    since=since,
                    until=until,
                    threshold=cfg.classifier_escalation_alarm_threshold,
                ),
                indent=2,
            )
        )
        return 0
    if args.classify_command != "review":
        return 2
    try:
        out_of_scope = frozenset()
        if args.config:
            from rumen.config import load_config

            out_of_scope = load_config(args.config).out_of_scope
        correction = record_classification_review(
            session,
            item_table=args.item_table,
            item_id=args.item_id,
            facet=args.facet,
            corrected_concept_ids=args.concept_ids,
            actor=args.actor,
            rationale=args.rationale,
            source="cli",
            out_of_scope=out_of_scope,
            route=args.route,
        )
        session.commit()
    except ValueError as exc:
        session.rollback()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        session.rollback()
        raise
    print(
        json.dumps(
            {
                "id": correction.id,
                "item_table": correction.item_table,
                "item_id": correction.item_id,
                "facet": correction.facet,
                "action": correction.action,
                "status": correction.status,
            }
        )
    )
    return 0


def _cmd_run(
    args,
    session: Session,
    *,
    notification_channel_resolver: NotificationChannelResolver | None = None,
) -> int:
    from rumen.config import load_config
    from rumen.jobs import run_backlog_refresh, run_pipeline

    if not args.config:
        print("error: --config (or RUMEN_CONFIG) is required for `rumen run`", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    reports = None
    if cfg.classifier_path:
        from rumen.classifier.fit import load_classifier

        try:
            reports = load_classifier(cfg.classifier_path)
        except ValueError as exc:
            print(
                f"warning: HMC classifier bundle {cfg.classifier_path!r} is unusable ({exc}) — "
                "running Tier-0 plus facet-constrained Tier-2 only.",
                file=sys.stderr,
            )
        if reports is None:
            print(
                f"warning: HMC classifier bundle {cfg.classifier_path!r} not found — running "
                "Tier-0 plus facet-constrained Tier-2 only. Run `rumen fit-classifier` first.",
                file=sys.stderr,
            )
    pipeline_kwargs = dict(
        adapters=cfg.adapters,
        source_maintainers=cfg.source_maintainers,
        reports=reports,
        llm_command=cfg.llm_command,
        out_of_scope=cfg.out_of_scope,
        max_llm_calls=cfg.max_llm_calls,
        commit_batch_size=cfg.commit_batch_size,
        classify_write_max_attempts=cfg.classify_write_max_attempts,
        classify_write_retry_backoff_seconds=cfg.classify_write_retry_backoff_seconds,
        tau_class=cfg.tau_class,
        tau_high=cfg.tau_high,
        tau_low=cfg.tau_low,
        mmr_lambda=cfg.mmr_lambda,
        exploration_fraction=cfg.exploration_fraction,
        half_life_days=cfg.half_life_days,
        default_half_life_days=cfg.default_half_life_days,
        residue_window_days=cfg.residue_window_days,
        classifier_path=cfg.classifier_path,
        classifier_escalation_alarm_threshold=cfg.classifier_escalation_alarm_threshold,
    )
    mining_enabled = cfg.backlog_mining_enabled
    channel, channel_setup_error = (
        _resolve_notification_channel(notification_channel_resolver)
        if mining_enabled
        else (None, None)
    )
    refresh = (
        run_backlog_refresh(
            session,
            config=cfg,
            digest_enabled=bool(args.digest),
            digest_period_days=args.digest_period_days,
            local_polish=args.local_polish,
            local_model_id=(cfg.local_llm_by_host.get(cfg.ai_host) or {}).get("model_id"),
            channel=channel,
            channel_setup_error=channel_setup_error,
            **pipeline_kwargs,
        )
        if mining_enabled
        else None
    )
    result = refresh.pipeline if refresh is not None else run_pipeline(session, **pipeline_kwargs)
    print(
        json.dumps(
            {
                "ingested": {r.source: r.items_passed_filter for r in result.ingest_runs},
                "classified": result.classified,
                "grouped": result.grouped,
                "ranked": result.ranked_count,
                "llm_calls": result.llm_calls,
                "llm_failures": result.llm_failures,
                "classification_pass_id": result.classification_pass_id,
                "classifier_alarms": result.classifier_alarms,
                "youtube": result.youtube,
                "ok": result.ok,
            }
        )
    )
    if result.llm_budget_tripped:
        print(
            f"warning: Tier-2 LLM budget/breaker tripped ({result.llm_calls} calls, "
            f"{result.llm_failures} failures) — items left unclassified; check classifier "
            "routing or LLM quota.",
            file=sys.stderr,
        )
    for alarm in result.classifier_alarms:
        print(
            "alarm: classifier escalation rate "
            f"pass={result.classification_pass_id} facet={alarm['facet']} "
            f"rate={alarm['rate']:.3f} threshold={alarm['threshold']:.3f}",
            file=sys.stderr,
        )
    if refresh is not None and refresh.digest is not None:
        print(
            json.dumps(
                {
                    "digest_id": refresh.digest.id,
                    "period_end": refresh.digest.period_end.isoformat(),
                    "mining": refresh.mining.result.counts if refresh.mining.result else None,
                    "mining_error": type(refresh.mining.error).__name__
                    if refresh.mining.error
                    else None,
                    "alert_status": refresh.mining.alert_status,
                }
            )
        )
    elif getattr(args, "digest", False):
        # End-to-end: synthesize the period digest right after the refresh. Extractive
        # (zero-LLM, D-17) so the refresh always produces a digest regardless of LLM health.
        from rumen.jobs import run_digest

        now = datetime.now(UTC)
        digest = run_digest(
            session,
            llm_command=cfg.llm_command,
            period_start=now - timedelta(days=args.digest_period_days),
            period_end=now,
            top_n=cfg.digest_top_n,
            section_map=cfg.digest_section_map,
            local_polish=args.local_polish,
            local_model_id=(cfg.local_llm_by_host.get(cfg.ai_host) or {}).get("model_id"),
        )
        _append_backlog_candidate_references(session, digest)
        print(json.dumps({"digest_id": digest.id, "period_end": digest.period_end.isoformat()}))
    return result.exit_code


def _cmd_backlog(
    args,
    session: Session,
    *,
    notification_channel_resolver: NotificationChannelResolver | None = None,
) -> int:
    if args.backlog_command == "mine":
        if not args.config:
            print(
                "error: --config (or RUMEN_CONFIG) is required for `rumen backlog mine`",
                file=sys.stderr,
            )
            return 1
        from rumen.config import load_config
        from rumen.jobs import run_backlog_mining

        config = load_config(args.config)
        channel, channel_setup_error = _resolve_notification_channel(notification_channel_resolver)
        result = run_backlog_mining(
            session,
            config=config,
            channel=channel,
            channel_setup_error=channel_setup_error,
        )
        print(
            json.dumps(
                {
                    "mining": result.result.counts if result.result else None,
                    "error": type(result.error).__name__ if result.error else None,
                    "alert_status": result.alert_status,
                }
            )
        )
        return 0 if result.error is None else 1

    from sqlalchemy import select

    from rumen.backlog_mining.projection import build_cc_task_plan
    from rumen.backlog_mining.repository import CandidateRepositoryError
    from rumen.backlog_mining.review import (
        CandidateRenderError,
        candidate_for_identifier,
        candidate_review,
        render_candidate,
        review_candidate,
    )
    from rumen.models import BacklogCandidate

    try:
        if args.backlog_command == "show":
            print(json.dumps(candidate_review(session, args.candidate), indent=2, default=str))
            return 0
        if args.backlog_command == "list":
            statement = select(BacklogCandidate).order_by(
                BacklogCandidate.created_at, BacklogCandidate.id
            )
            if args.state:
                statement = statement.where(BacklogCandidate.state == args.state)
            candidates = []
            for candidate in session.scalars(statement):
                entry = {
                    "id": candidate.id,
                    "fingerprint": candidate.fingerprint,
                    "state": candidate.state,
                }
                if candidate.state in {"accepted", "edited"}:
                    draft = render_candidate(session, candidate.id)
                    entry["draft"] = {
                        "title": draft.title,
                        "body": draft.body,
                        "evidence_links": draft.evidence_links,
                    }
                candidates.append(entry)
            print(json.dumps(candidates, indent=2, default=str))
            return 0
        if args.backlog_command != "review":
            return 2

        candidate = candidate_for_identifier(session, args.candidate)
        creation_requested = args.create_cc_task
        config = None
        if creation_requested:
            if not args.config:
                raise ValueError(
                    "--config (or RUMEN_CONFIG) is required for `rumen backlog review --create-cc-task`"
                )
            from rumen.config import load_config

            config = load_config(args.config)
            if not args.cc_session:
                raise ValueError("--cc-session is required when requesting a CC-task plan")
        frame = review_candidate(
            session,
            args.candidate,
            action=args.action,
            actor=args.actor,
            rationale=args.rationale,
            payload={
                "cc_task_creation_requested": creation_requested,
                "cc_session": args.cc_session if creation_requested else None,
            },
        )
        output: dict[str, object] = {
            "frame_id": frame.frame_id,
            "candidate_id": candidate.id,
            "action": frame.action,
        }
        if args.promote_as:
            from rumen.backlog_mining.promotion import (
                PromotionExecutor,
                PromotionRequest,
            )

            output["promotion"] = PromotionExecutor(session).execute(
                candidate,
                PromotionRequest(
                    primitive=args.promote_as,
                    actor=args.actor,
                    existing_task=args.existing_task,
                    operation_id=args.operation_id,
                    body_file=args.body_file,
                    task_status=args.task_status,
                    reason=args.reason,
                    new_task_title=args.new_task_title,
                    new_task_description_file=args.new_task_description_file,
                    new_task_priority=args.new_task_priority,
                    new_task_status=args.new_task_status,
                    epic_title=args.epic_title,
                    epic_description_file=args.epic_description_file,
                    epic_priority=args.epic_priority,
                    epic_status=args.epic_status,
                    approved_plan=args.approved_plan,
                    confirm_plan=args.confirm_plan,
                ),
            )
        if creation_requested:
            draft = render_candidate(session, candidate.id)
            output["cc_task_plan"] = build_cc_task_plan(
                candidate,
                draft,
                enabled=config.cc_task_creation.enabled,
                requested_session=args.cc_session,
                invoking_session=os.environ.get("RUMEN_CC_SESSION", ""),
            )
        session.commit()
        print(json.dumps(output, indent=2))
        return 0
    except (CandidateRepositoryError, CandidateRenderError, ValueError) as exc:
        # A promotion failure is itself a durable safety receipt.  Its preceding
        # human disposition and the failed journal row must survive this CLI error.
        from rumen.backlog_mining.promotion import PromotionError

        if isinstance(exc, PromotionError):
            session.commit()
        else:
            session.rollback()
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_digest(args, session: Session) -> int:
    from rumen.config import load_config
    from rumen.jobs import run_digest

    if not args.config:
        print("error: --config (or RUMEN_CONFIG) is required for `rumen digest`", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    abstractive = getattr(args, "abstractive", False)
    if abstractive and cfg.llm_command is None:
        print("error: --abstractive requested but config has no llm_command", file=sys.stderr)
        return 1
    now = datetime.now(UTC)
    period_end = datetime.fromisoformat(args.period_end) if args.period_end else now
    period_start = (
        datetime.fromisoformat(args.period_start)
        if args.period_start
        else period_end - timedelta(days=7)
    )
    digest = run_digest(
        session,
        llm_command=cfg.llm_command,
        period_start=period_start,
        period_end=period_end,
        top_n=cfg.digest_top_n,
        force=args.force,
        extractive=not abstractive,
        section_map=cfg.digest_section_map,
        local_polish=args.local_polish,
        local_model_id=(cfg.local_llm_by_host.get(cfg.ai_host) or {}).get("model_id"),
    )
    _append_backlog_candidate_references(session, digest)
    print(json.dumps({"digest_id": digest.id, "period_end": digest.period_end.isoformat()}))
    return 0


def _append_backlog_candidate_references(session: Session, digest) -> None:
    """Store read-only review commands for candidates already associated with a digest."""
    from sqlalchemy import select

    from rumen.digest import append_backlog_candidate_review_references
    from rumen.models import BacklogCandidate

    candidates = list(
        session.scalars(
            select(BacklogCandidate)
            .where(BacklogCandidate.digest_id == digest.id)
            .order_by(BacklogCandidate.id)
        )
    )
    digest.content_markdown = append_backlog_candidate_review_references(
        digest.content_markdown,
        [{"id": candidate.id, "fingerprint": candidate.fingerprint} for candidate in candidates],
    )
    session.commit()


def _cmd_fit_classifier(args, session: Session) -> int:
    from rumen.classifier.fit import fit_from_session, save_classifier
    from rumen.config import load_config

    if not args.config:
        print(
            "error: --config (or RUMEN_CONFIG) is required for `rumen fit-classifier`",
            file=sys.stderr,
        )
        return 1
    cfg = load_config(args.config)
    out = args.out or cfg.classifier_path
    if not out:
        print(
            "error: no artifact path — set classifier_path in config or pass --out", file=sys.stderr
        )
        return 1
    try:
        reports = fit_from_session(session, path_consistency_floor=cfg.tau_class)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save_classifier(reports, out)
    print(
        json.dumps(
            {
                "path": out,
                "facets": {
                    facet: {
                        "n_bootstrap": report.n_bootstrap,
                        "n_after_gate": report.n_after_gate,
                        "n_excluded": report.n_excluded,
                        "tier1_eligible": report.tier1_eligible,
                        "gold_precision": report.gold_evaluation.precision,
                        "gold_recall": report.gold_evaluation.recall,
                    }
                    for facet, report in reports.items()
                },
            },
            indent=2,
        )
    )
    return 0


def _cmd_taxonomy_audit(args, session: Session) -> int:
    import uuid

    from rumen.config import load_config
    from rumen.jobs import run_taxonomy_audit
    from rumen.taxonomy.audit.rollback import rollback_revision

    if not args.config:
        print(
            "error: --config (or RUMEN_CONFIG) is required for `rumen taxonomy-audit`",
            file=sys.stderr,
        )
        return 1
    cfg = load_config(args.config)
    if args.rollback_revision is not None:
        try:
            revision = rollback_revision(
                session,
                args.rollback_revision,
                audit_run_id=f"rollback-{uuid.uuid4()}",
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"revision_id": revision.id, "reverts": revision.reverts}))
        return 0
    result = run_taxonomy_audit(
        session,
        llm_command=cfg.llm_command,
        crowded_item_threshold=cfg.taxonomy_audit_crowded_item_threshold,
        sparse_item_threshold=cfg.taxonomy_audit_sparse_item_threshold,
        duplicate_similarity_threshold=cfg.taxonomy_audit_duplicate_similarity_threshold,
    )
    print(
        json.dumps(
            {
                "audit_run_id": result.audit_run_id,
                "detected": result.detected,
                "applied_revision_ids": result.applied_revision_ids,
                "skipped": result.skipped,
            }
        )
    )
    return result.exit_code


def main(
    argv: list[str] | None = None,
    *,
    notification_channel_resolver: NotificationChannelResolver | None = None,
) -> int:
    import logging

    from rumen.config import (
        PipelineConfig,
        load_config,
        migration_target_identity,
    )
    from rumen.db.migrate import upgrade_to_head
    from rumen.storage import open_storage

    args = _build_parser().parse_args(argv)
    # Phase timings + warnings go to stderr so a long/stalled run is observable live
    # (RUMEN_LOG_LEVEL overrides; default INFO).
    logging.basicConfig(
        level=os.environ.get("RUMEN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config_path = getattr(args, "config", os.environ.get("RUMEN_CONFIG"))
    config = load_config(config_path) if config_path else PipelineConfig()
    target_identity = migration_target_identity(config_path) if config_path else None
    engine = open_storage(config)
    upgrade_to_head(engine, backlog_target_identity=target_identity)
    with Session(engine) as session:
        return _run_parsed(
            args,
            session,
            notification_channel_resolver=notification_channel_resolver,
        )


if __name__ == "__main__":
    raise SystemExit(main())
