"""Phase-2 mining: evidence packets, trusted routing, and durable review records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from rumen.backlog_mining.judge import (
    ActionabilityJudge,
    JudgeResult,
    LlmActionabilityJudge,
)
from rumen.backlog_mining.notifications import NotificationChannel
from rumen.backlog_mining.repository import CandidateRepository
from rumen.backlog_mining.task_store import (
    BeadsTaskStoreReader,
    TaskInventory,
    TaskInventoryItem,
    TaskStoreReader,
)
from rumen.config import (
    PipelineConfig,
    TargetTemplate,
    create_category_mapping_result,
    mapping_confidence,
)
from rumen.models import (
    BacklogCandidate,
    BacklogCandidateTaskLink,
    BacklogMappingReview,
    BacklogMiningRun,
    BacklogTaskMatch,
    Digest,
    DigestEntity,
    FeedItem,
    ItemConcept,
    SkosConcept,
    TopicEntity,
)


@dataclass(slots=True)
class MiningResult:
    run_id: int
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TargetResolution:
    target: TargetTemplate | None
    event: str | None
    category: str | None
    confidence: float
    trusted_category: bool

    def __iter__(self):
        yield self.target
        yield self.event

    def __getitem__(self, index: int) -> TargetTemplate | None | str:
        return (self.target, self.event)[index]


class BacklogMiningFailure(RuntimeError):
    """A failed run with the precise durable row that must receive alert telemetry."""

    def __init__(self, run_id: int, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.run_id = run_id
        self.cause = cause


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _stable_identity(entity: TopicEntity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "stable_id_kind": entity.stable_id_kind,
        "stable_id_value": entity.stable_id_value,
        "canonical_url": entity.canonical_url,
    }


def _canonical_inventory_match_identity(
    entity: TopicEntity, target: TargetTemplate, item: TaskInventoryItem
) -> str:
    return json.dumps(
        {
            "topic_entity_id": entity.id,
            "target_identity": target.target_identity,
            "task_store": item.task_store,
            "external_task_key": item.external_task_key,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _excerpt(item: FeedItem) -> str | None:
    content = (item.content or "").strip()
    return content[:1000] if content else None


class BacklogMiningService:
    def __init__(
        self,
        session: Session,
        config: PipelineConfig,
        judge: ActionabilityJudge | None = None,
        *,
        task_store_reader: TaskStoreReader | None = None,
        channel: NotificationChannel | None = None,
    ) -> None:
        self.session = session
        self.config = config
        self.judge = judge or LlmActionabilityJudge(
            config.actionability_command_factory, max_calls=config.actionability_max_calls
        )
        self.task_store_reader = task_store_reader
        self.channel = channel
        self.repository = CandidateRepository(session)
        self._candidate_phase_started = False

    def mine(self, *, now: datetime | None = None, digest_period_days: int = 7) -> MiningResult:
        now = now or _utcnow()
        run = BacklogMiningRun(status="running", config_snapshot=self._config_snapshot(), counts={})
        self.session.add(run)
        self.session.commit()
        self._candidate_phase_started = False
        counts: dict[str, int] = {
            "eligible": 0,
            "drafted": 0,
            "abstained": 0,
            "skipped": 0,
            "suppressed": 0,
            "existing_family": 0,
            "family_reviews": 0,
            "mapping_reviews": 0,
            "mapping_created": 0,
            "mapping_conflicts": 0,
        }
        try:
            resolved = self._resolve_all_targets_before_candidate_writes(
                run, now=now, digest_period_days=digest_period_days, counts=counts
            )
            self._candidate_phase_started = True
            for entity, packet, resolution in resolved:
                if resolution.event == "review":
                    continue
                target = resolution.target
                if target is None:
                    self._count_skip(counts, "no_target")
                    continue
                inventory = self._inventory_for(target)
                if not inventory.available:
                    self._count_skip(counts, "task_inventory_unavailable")
                    continue
                family = self.repository.find_by_family_key(
                    topic_entity_id=entity.id, target_identity=target.target_identity
                )
                disposition = self._route_existing_family(family, packet, inventory, run)
                if disposition is None:
                    disposition = self._route_discovered_inventory_match(
                        entity, target, packet, inventory, run
                    )
                if disposition is not None:
                    counts["existing_family"] += 1
                    if disposition == "closed_review":
                        counts["family_reviews"] += 1
                        self._notify(
                            "Rumen backlog task requires review",
                            f"Closed task observed for candidate family of entity {entity.id}.",
                        )
                    continue
                result = self.judge.evaluate(packet, target)
                if result.outcome != "actionable" or result.candidate_draft is None:
                    counts["abstained"] += 1
                    continue
                fingerprint = self._fingerprint(entity.id, target.target_identity, result)
                existing = self.repository._active_candidate(fingerprint)
                self.repository.create_candidate(
                    fingerprint=fingerprint,
                    target_identity=target.target_identity,
                    topic_entity_id=entity.id,
                    digest_id=packet["digest"]["id"] if packet["digest"] else None,
                    source_snapshot=packet,
                    content_snapshot=result.candidate_draft,
                    assessment={
                        **result.assessment,
                        "generator_provenance": result.generator_provenance,
                        "evaluator_provenance": result.evaluator_provenance,
                    },
                    configuration_snapshot={
                        "target_identity": target.target_identity,
                        "beads_repo_identity": target.beads_repo_identity,
                        "task_store": target.task_store,
                        "template_revision": target.template_revision,
                    },
                    evidence=self._candidate_evidence(packet),
                )
                if existing is None:
                    counts["drafted"] += 1
                else:
                    counts["suppressed"] += 1
            run.status = "completed"
            run.counts = counts
            run.completed_at = _utcnow()
            self.session.commit()
            return MiningResult(run.id, counts)
        except Exception as exc:
            self.session.rollback()
            failed = self.session.get(BacklogMiningRun, run.id)
            if failed is not None:
                failed.status = "failed"
                failed.counts = counts
                failed.error_payload = {"type": type(exc).__name__, "message": str(exc)}
                failed.completed_at = _utcnow()
                self.session.commit()
            raise BacklogMiningFailure(run.id, exc) from exc
        finally:
            self._candidate_phase_started = False

    def _resolve_all_targets_before_candidate_writes(
        self,
        run: BacklogMiningRun,
        *,
        now: datetime,
        digest_period_days: int,
        counts: dict[str, int],
    ) -> list[tuple[TopicEntity, dict[str, Any], TargetResolution]]:
        """Persist durable mapping receipts before the run accumulates candidate work.

        A mapping may already be durable in configuration, so its database receipt must
        survive later candidate rollback.  Resolving every mapping in this preflight
        phase makes that narrow commit occur before this session contains any candidate
        write; candidate creation then remains one rollback unit through the run boundary.
        """
        resolved: list[tuple[TopicEntity, dict[str, Any], TargetResolution]] = []
        for entity in self._ranked_entities():
            packet, reason = self.build_source_packet(
                entity, now=now, digest_period_days=digest_period_days
            )
            if reason is not None:
                self._count_skip(counts, reason)
                continue
            assert packet is not None
            counts["eligible"] += 1
            resolution = self._resolve_target(entity)
            if resolution.event in {"created", "resolved_conflict", "review"}:
                self._record_mapping_decision(run, entity, resolution)
            if resolution.event == "created":
                counts["mapping_created"] += 1
            if resolution.event == "resolved_conflict":
                counts["mapping_conflicts"] += 1
            if resolution.event == "review":
                counts["mapping_reviews"] += 1
            resolved.append((entity, packet, resolution))
        return resolved

    @staticmethod
    def _count_skip(counts: dict[str, int], reason: str) -> None:
        counts["skipped"] += 1
        key = f"skipped:{reason}"
        counts[key] = counts.get(key, 0) + 1

    def build_source_packet(
        self, entity: TopicEntity, *, now: datetime, digest_period_days: int = 7
    ) -> tuple[dict[str, Any] | None, str | None]:
        if (
            entity.rank_position is None
            or entity.rank_position > self.config.backlog_max_rank_position
        ):
            return None, "outside_rank_window"
        cutoff = now - timedelta(days=self.config.backlog_max_age_days)
        items = list(
            self.session.scalars(
                select(FeedItem).where(FeedItem.topic_entity_id == entity.id).order_by(FeedItem.id)
            )
        )
        if not items or any(item.published_at < cutoff for item in items):
            return None, "stale_source_claim"
        claims = [
            {"id": item.id, "url": item.url, "quote_or_excerpt": _excerpt(item)} for item in items
        ]
        if any(claim["quote_or_excerpt"] is None for claim in claims):
            return None, "no_usable_fresh_source_claim"
        digest = self._current_digest(
            entity.id,
            period_start=now - timedelta(days=digest_period_days),
            period_end=now,
        )
        return {
            "entity_identity": _stable_identity(entity),
            "source_claims": claims,
            "rank_metadata": {
                "relevance_score": entity.relevance_score,
                "rank_position": entity.rank_position,
                "is_exploration": entity.is_exploration,
                "updated_at": entity.updated_at.isoformat(),
            },
            "digest": digest,
        }, None

    def _ranked_entities(self) -> list[TopicEntity]:
        return list(
            self.session.scalars(
                select(TopicEntity)
                .where(TopicEntity.rank_position.is_not(None))
                .order_by(TopicEntity.rank_position, TopicEntity.id)
            )
        )

    def _current_digest(
        self, entity_id: int, *, period_start: datetime, period_end: datetime
    ) -> dict[str, Any] | None:
        digest = self.session.scalar(
            select(Digest)
            .join(DigestEntity, DigestEntity.digest_id == Digest.id)
            .where(
                DigestEntity.topic_entity_id == entity_id,
                Digest.period_start >= period_start,
                Digest.period_end <= period_end,
            )
            .order_by(Digest.period_end.desc(), Digest.id.desc())
        )
        if digest is None:
            return None
        return {"id": digest.id, "excerpt": digest.content_markdown[:1000]}

    def _resolve_target(self, entity: TopicEntity) -> TargetResolution:
        category = entity.category
        target = self.config.category_to_target_template.get(category or "")
        if target is not None:
            return TargetResolution(
                target, None, category, 0.0, self._trusted_category(entity, category)
            )
        confidence = mapping_confidence(self.config, category)
        trusted_category = self._trusted_category(entity, category)
        if (
            category is not None
            and category in self.config.auto_mappable_categories
            and trusted_category
            and self.config.auto_create_mapping
            and confidence == 1.0
            and confidence >= self.config.auto_create_mapping_confidence_threshold
        ):
            write = create_category_mapping_result(self.config, category)
            return TargetResolution(
                write.target,
                "created" if write.created else "resolved_conflict",
                category,
                confidence,
                trusted_category,
            )
        return TargetResolution(None, "review", category, confidence, trusted_category)

    def _trusted_category(self, entity: TopicEntity, category: str | None) -> bool:
        """Only human/rule-owned taxonomy decisions may create a new routing entry."""
        if category is None:
            return False
        human_entity_category = self.session.scalar(
            select(ItemConcept.id)
            .join(SkosConcept, SkosConcept.id == ItemConcept.concept_id)
            .where(
                ItemConcept.item_table == "topic_entities",
                ItemConcept.item_id == entity.id,
                ItemConcept.facet == "topic",
                ItemConcept.method.startswith("human:"),
                SkosConcept.pref_label == category,
            )
            .limit(1)
        )
        if human_entity_category is not None:
            return True
        return (
            self.session.scalar(
                select(FeedItem.id)
                .where(
                    FeedItem.topic_entity_id == entity.id,
                    FeedItem.category == category,
                    or_(
                        FeedItem.classification_method == "rule",
                        FeedItem.classification_method.startswith("human:"),
                    ),
                )
                .limit(1)
            )
            is not None
        )

    def _inventory_for(self, target: TargetTemplate) -> TaskInventory:
        reader = self.task_store_reader or BeadsTaskStoreReader(target.task_store)
        try:
            return reader.list_inventory()
        except Exception as exc:  # a third-party reader is also fail-closed
            return TaskInventory.unavailable(str(exc))

    def _route_existing_family(
        self,
        family: list[BacklogCandidate],
        packet: dict[str, Any],
        inventory: TaskInventory,
        run: BacklogMiningRun,
    ) -> str | None:
        observed = {(item.task_store, item.external_task_key): item for item in inventory.items}
        for candidate in family:
            links = list(
                self.session.scalars(
                    select(BacklogCandidateTaskLink).where(
                        BacklogCandidateTaskLink.fingerprint == candidate.fingerprint
                    )
                )
            )
            for link in links:
                item = observed.get((link.task_store, link.external_task_key))
                if item is not None and item.state.casefold() in {
                    "closed",
                    "done",
                    "resolved",
                    "archived",
                }:
                    self._record_link_observation(candidate, link, item, run, packet)
                    return "closed_review"
            if links:
                self.repository.append_evidence(candidate.id, self._candidate_evidence(packet))
                return "linked"
            match = self.session.scalar(
                select(BacklogTaskMatch)
                .where(BacklogTaskMatch.candidate_id == candidate.id)
                .order_by(BacklogTaskMatch.id.desc())
            )
            if match is None:
                continue
            item = observed.get((match.task_store, match.external_task_key))
            if item is None:
                self.repository.append_evidence(candidate.id, self._candidate_evidence(packet))
                return "missing_from_complete_inventory"
            self._record_match_observation(candidate, match, item, run, packet)
            if item.state.casefold() in {"closed", "done", "resolved", "archived"}:
                return "closed_review"
            return "open_unlinked"
        return None

    def _route_discovered_inventory_match(
        self,
        entity: TopicEntity,
        target: TargetTemplate,
        packet: dict[str, Any],
        inventory: TaskInventory,
        run: BacklogMiningRun,
    ) -> str | None:
        """Attach one deterministic fresh-inventory match before model assessment.

        A title match is deliberately exact after whitespace/case normalization.  It
        avoids inventing a semantic similarity score at the task-store boundary; an
        ambiguous inventory result is left for the ordinary human-review path.
        """
        expected_title = self._normalized_inventory_title(entity.canonical_title)
        matches = [
            item
            for item in inventory.items
            if item.task_store == target.task_store
            and self._normalized_inventory_title(item.title) == expected_title
        ]
        if len(matches) != 1:
            return None
        item = matches[0]
        fingerprint = hashlib.sha256(
            _canonical_inventory_match_identity(entity, target, item).encode("utf-8")
        ).hexdigest()
        candidate = self.session.scalar(
            select(BacklogCandidate)
            .where(BacklogCandidate.fingerprint == fingerprint)
            .order_by(BacklogCandidate.id)
        )
        is_closed = item.state.casefold() in {"closed", "done", "resolved", "archived"}
        if candidate is None:
            candidate = self.repository.create_candidate(
                fingerprint=fingerprint,
                topic_entity_id=entity.id,
                target_identity=target.target_identity,
                digest_id=packet["digest"]["id"] if packet["digest"] else None,
                source_snapshot=packet,
                content_snapshot={"inventory_match_title": item.title},
                assessment={"outcome": "inventory_match", "match_method": "canonical_title_exact"},
                configuration_snapshot={
                    "target_identity": target.target_identity,
                    "beads_repo_identity": target.beads_repo_identity,
                    "task_store": target.task_store,
                    "template_revision": target.template_revision,
                },
                evidence=self._candidate_evidence(packet),
            )
            if not is_closed:
                candidate.state = "knowledge_only"
        self.session.add(
            BacklogTaskMatch(
                candidate_id=candidate.id,
                mining_run_id=run.id,
                task_store=item.task_store,
                external_task_key=item.external_task_key,
                title=item.title,
                state=item.state,
                closure_reason_text=item.closure_reason_text,
                match_evidence={
                    "match_method": "canonical_title_exact",
                    "inventory_checked_at": str(inventory.checked_at),
                },
            )
        )
        return "closed_review" if is_closed else "open_unlinked"

    @staticmethod
    def _normalized_inventory_title(title: str) -> str:
        return " ".join(title.casefold().split())

    def _record_link_observation(
        self,
        candidate: BacklogCandidate,
        link: BacklogCandidateTaskLink,
        item: TaskInventoryItem,
        run: BacklogMiningRun,
        packet: dict[str, Any],
    ) -> None:
        """Persist the closed linked-task observation before surfacing its review."""
        link.last_observed_title = item.title
        link.last_observed_state = item.state
        link.last_observed_closure_reason = item.closure_reason_text
        link.checked_at = _utcnow()
        self.session.add(
            BacklogTaskMatch(
                candidate_id=candidate.id,
                mining_run_id=run.id,
                task_store=item.task_store,
                external_task_key=item.external_task_key,
                title=item.title,
                state=item.state,
                closure_reason_text=item.closure_reason_text,
                match_evidence={
                    "task_link_id": link.id,
                    "inventory_checked_at": str(item.updated_at),
                },
            )
        )
        evidence = self._candidate_evidence(packet)
        evidence.append(
            {
                "kind": "task_inventory_match",
                "excerpt": item.title,
                "metadata": {
                    "task_store": item.task_store,
                    "external_task_key": item.external_task_key,
                    "state": item.state,
                    "closure_reason_text": item.closure_reason_text,
                },
            }
        )
        self.repository.append_evidence(candidate.id, evidence)

    def _record_match_observation(
        self,
        candidate: BacklogCandidate,
        prior: BacklogTaskMatch,
        item: TaskInventoryItem,
        run: BacklogMiningRun,
        packet: dict[str, Any],
    ) -> None:
        self.session.add(
            BacklogTaskMatch(
                candidate_id=candidate.id,
                mining_run_id=run.id,
                task_store=item.task_store,
                external_task_key=item.external_task_key,
                title=item.title,
                state=item.state,
                closure_reason_text=item.closure_reason_text,
                match_evidence={
                    "prior_match_id": prior.id,
                    "inventory_checked_at": str(item.updated_at),
                },
            )
        )
        evidence = self._candidate_evidence(packet)
        evidence.append(
            {
                "kind": "task_inventory_match",
                "excerpt": item.title,
                "metadata": {
                    "task_store": item.task_store,
                    "external_task_key": item.external_task_key,
                    "state": item.state,
                    "closure_reason_text": item.closure_reason_text,
                },
            }
        )
        self.repository.append_evidence(candidate.id, evidence)

    def _record_mapping_decision(
        self, run: BacklogMiningRun, entity: TopicEntity, resolution: TargetResolution
    ) -> None:
        if self._candidate_phase_started:
            raise RuntimeError("mapping decisions must be recorded before candidate writes begin")
        status = {
            "created": "auto_created",
            "resolved_conflict": "resolved_conflict",
            "review": "review_required",
        }[resolution.event or "review"]
        review = BacklogMappingReview(
            mining_run_id=run.id,
            topic_entity_id=entity.id,
            category=resolution.category,
            status=status,
            confidence=resolution.confidence,
            threshold=self.config.auto_create_mapping_confidence_threshold,
            target_identity=resolution.target.target_identity if resolution.target else None,
            decision_inputs={
                "auto_create_mapping": self.config.auto_create_mapping,
                "auto_mappable_categories": list(self.config.auto_mappable_categories),
                "trusted_category": resolution.trusted_category,
                "default_target_template": self.config.default_target_template.target_identity
                if self.config.default_target_template
                else None,
            },
        )
        self.session.add(review)
        self.session.flush()
        if resolution.event == "review":
            review.notification_status, review.notification_error = self._notify(
                "Rumen backlog mapping requires review",
                f"Category {resolution.category!r} was not auto-mapped for entity {entity.id}.",
            )
        # A category mapping is an external, already-durable config mutation. This
        # preflight happens before candidate writes, so the receipt's commit cannot
        # make any candidate work durable before the run boundary.
        self.session.commit()

    def _notify(self, subject: str, body: str) -> tuple[str, str | None]:
        if self.channel is None:
            return "not_configured", None
        try:
            return ("sent" if self.channel.send(subject, body) else "delivery_failed"), None
        except Exception as exc:  # a notification cannot create a draft or mask its review record
            return "alert_error", repr(exc)

    @staticmethod
    def _fingerprint(entity_id: int, target_identity: str, result: JudgeResult) -> str:
        normalized = json.dumps(result.candidate_draft, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{entity_id}|{target_identity}|{normalized}".encode()).hexdigest()

    @staticmethod
    def _candidate_evidence(packet: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "kind": "source_claim",
                "feed_item_id": claim["id"],
                "excerpt": claim["quote_or_excerpt"],
                "metadata": {"url": claim["url"]},
            }
            for claim in packet["source_claims"]
        ]

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "category_to_target_template": {
                key: value.target_identity
                for key, value in self.config.category_to_target_template.items()
            },
            "auto_create_mapping": self.config.auto_create_mapping,
            "auto_create_mapping_confidence_threshold": self.config.auto_create_mapping_confidence_threshold,
            "auto_mappable_categories": list(self.config.auto_mappable_categories),
            "default_target_template": self.config.default_target_template.target_identity
            if self.config.default_target_template
            else None,
        }
