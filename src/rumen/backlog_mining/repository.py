"""Transactional draft-queue repositories.

The candidate repository is the sole writer for candidate lifecycle projections.
Task-store mutations are intentionally absent from this Phase 1 module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rumen.backlog_mining.models import (
    ACTIVE_CANDIDATE_STATES,
    DISPOSITION_ACTIONS,
    BacklogCandidate,
    BacklogCandidateDisposition,
    BacklogCandidateEvidence,
    BacklogCandidatePromotion,
    BacklogCandidateTaskLink,
)
from rumen.backlog_mining.task_store import TaskInventoryItem, TaskStoreReader


class CandidateRepositoryError(ValueError):
    """Base class for typed no-change lifecycle failures."""


class UnsupportedActionError(CandidateRepositoryError):
    pass


class MissingRationaleError(CandidateRepositoryError):
    pass


class CallerPostimageError(CandidateRepositoryError):
    pass


class StaleHeadError(CandidateRepositoryError):
    pass


class LifecycleConflictError(CandidateRepositoryError):
    pass


class CandidateNotFoundError(CandidateRepositoryError):
    pass


@dataclass(frozen=True, slots=True)
class CandidateEvidenceGroup:
    """Evidence for one candidate in an entity-wide reviewer view."""

    candidate: BacklogCandidate
    evidence: tuple[BacklogCandidateEvidence, ...]


_TARGET_STATES = {
    "accept": "accepted",
    "edit": "edited",
    "defer": "deferred",
    "reject": "rejected",
    "knowledge_only": "knowledge_only",
    "merge": "merged",
    "reopen": "draft",
}


class CandidateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_candidate(
        self,
        *,
        fingerprint: str,
        source_snapshot: Mapping[str, Any],
        content_snapshot: Mapping[str, Any],
        assessment: Mapping[str, Any],
        configuration_snapshot: Mapping[str, Any],
        evidence: Iterable[Mapping[str, Any]] = (),
        topic_entity_id: int | None = None,
        digest_id: int | None = None,
        target_identity: str | None = None,
        active: bool = True,
    ) -> BacklogCandidate:
        """Store a durable draft, or return the existing active fingerprint holder.

        ``active=False`` is only used for a supersession replacement's immutable
        bootstrap record; it starts ``superseded`` until its family frame activates it.
        """
        if not fingerprint.strip():
            raise CandidateRepositoryError("candidate fingerprint must not be blank")
        if active:
            existing = self._active_candidate(fingerprint)
            if existing is not None:
                return existing
        candidate = BacklogCandidate(
            fingerprint=fingerprint,
            source_snapshot=dict(source_snapshot),
            content_snapshot=dict(content_snapshot),
            assessment=dict(assessment),
            configuration_snapshot=dict(configuration_snapshot),
            topic_entity_id=topic_entity_id,
            digest_id=digest_id,
            target_identity=target_identity,
            state="draft" if active else "superseded",
        )
        try:
            with self.session.begin_nested():
                self.session.add(candidate)
                self.session.flush()
        except IntegrityError:
            existing = self._active_candidate(fingerprint)
            if existing is not None:
                return existing
            raise LifecycleConflictError(
                "active fingerprint uniqueness race could not be reconciled"
            )
        self._add_evidence(candidate.id, evidence)
        self.session.flush()
        return candidate

    def find_by_family_key(
        self, *, topic_entity_id: int, target_identity: str
    ) -> list[BacklogCandidate]:
        """Return every candidate in a pre-assessment family.

        A NULL target is deliberately not a member of a family.  Fingerprint lookups
        remain the later, phase-one identity gate.
        """
        if not target_identity:
            return []
        return list(
            self.session.scalars(
                select(BacklogCandidate)
                .where(
                    BacklogCandidate.topic_entity_id == topic_entity_id,
                    BacklogCandidate.target_identity == target_identity,
                )
                .order_by(BacklogCandidate.id)
            )
        )

    def cross_candidate_evidence_for_entity(
        self, topic_entity_id: int
    ) -> list[CandidateEvidenceGroup]:
        """Return every cited row for candidates sharing one topic entity.

        This intentionally includes all candidate states.  Candidate and evidence
        chronology provide a stable grouping for reviewer rendering without
        inferring whether any two claims contradict each other.
        """
        rows = self.session.execute(
            select(BacklogCandidate, BacklogCandidateEvidence)
            .join(
                BacklogCandidateEvidence,
                BacklogCandidateEvidence.candidate_id == BacklogCandidate.id,
            )
            .where(BacklogCandidate.topic_entity_id == topic_entity_id)
            .order_by(
                BacklogCandidate.created_at,
                BacklogCandidate.id,
                BacklogCandidateEvidence.created_at,
                BacklogCandidateEvidence.id,
            )
        )
        grouped: dict[int, tuple[BacklogCandidate, list[BacklogCandidateEvidence]]] = {}
        for candidate, evidence in rows:
            if candidate.id not in grouped:
                grouped[candidate.id] = (candidate, [])
            grouped[candidate.id][1].append(evidence)
        return [
            CandidateEvidenceGroup(candidate=candidate, evidence=tuple(evidence))
            for candidate, evidence in grouped.values()
        ]

    def append_evidence(
        self, candidate_id: int, evidence: Iterable[Mapping[str, Any]]
    ) -> list[BacklogCandidateEvidence]:
        """Accumulate source/context evidence without a new candidate or history frame."""
        candidate = self._candidate(candidate_id)
        if candidate.state not in ACTIVE_CANDIDATE_STATES:
            raise LifecycleConflictError("evidence can be appended only to the active candidate")
        rows = self._add_evidence(candidate.id, evidence)
        self.session.flush()
        return rows

    def transition(
        self,
        candidate_id: int,
        *,
        action: str,
        actor: str,
        rationale: str,
        expected_head: str | None = None,
        parent_frame_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        postimage: Mapping[str, Any] | None = None,
        replacement_candidate_id: int | None = None,
        merge_target_candidate_id: int | None = None,
    ) -> BacklogCandidateDisposition:
        """Append one whole-family disposition frame and materialize its projection."""
        if action not in DISPOSITION_ACTIONS:
            raise UnsupportedActionError(f"unsupported candidate action: {action!r}")
        if not actor.strip() or not rationale.strip():
            raise MissingRationaleError("candidate dispositions require actor and rationale")
        if postimage is not None:
            raise CallerPostimageError("candidate repository derives postimages")
        if parent_frame_id is not None:
            if expected_head is not None and expected_head != parent_frame_id:
                raise StaleHeadError("expected head and parent frame disagree")
            expected_head = parent_frame_id

        candidate = self._candidate(candidate_id)
        current_head = self._applied_head(candidate.fingerprint)
        current_head_id = current_head.frame_id if current_head else None
        if expected_head is not None and expected_head != current_head_id:
            raise StaleHeadError("candidate lifecycle head is stale")
        family = self._family(candidate.fingerprint)
        preimage = self._snapshot(family)
        updates = self._derive_updates(
            candidate,
            action,
            replacement_candidate_id=replacement_candidate_id,
            merge_target_candidate_id=merge_target_candidate_id,
        )
        postimage = self._updated_image(preimage, updates)
        abandoned = self._abandon_redo_future(candidate.fingerprint)
        self._apply_image(postimage)
        frame = BacklogCandidateDisposition(
            frame_id=str(uuid4()),
            fingerprint=candidate.fingerprint,
            sequence_no=self._next_sequence(candidate.fingerprint),
            parent_frame_id=current_head_id,
            record_kind="disposition",
            action=action,
            target_candidate_id=candidate.id,
            affected_candidate_ids=sorted(int(key) for key in postimage),
            preimage=preimage,
            postimage=postimage,
            action_payload=dict(payload or {}),
            abandoned_frame_ids=abandoned,
            actor=actor,
            rationale=rationale,
            stack_state="applied",
        )
        self.session.add(frame)
        self.session.flush()
        return frame

    def undo(self, fingerprint: str, *, actor: str, rationale: str) -> BacklogCandidateDisposition:
        if not actor.strip() or not rationale.strip():
            raise MissingRationaleError("candidate controls require actor and rationale")
        frame = self._applied_head(fingerprint)
        if frame is None:
            raise StaleHeadError("there is no applied disposition to undo")
        before = self._snapshot(self._family(fingerprint))
        desired = dict(frame.preimage)
        effective = self._collision_reconciled_image(desired, before)
        self._apply_image(effective)
        frame.stack_state = "reversible"
        control = self._control_record(
            fingerprint,
            action="undo",
            moved=frame,
            preimage=before,
            postimage=effective,
            actor=actor,
            rationale=rationale,
        )
        self.session.add(control)
        self.session.flush()
        return control

    def redo(self, fingerprint: str, *, actor: str, rationale: str) -> BacklogCandidateDisposition:
        if not actor.strip() or not rationale.strip():
            raise MissingRationaleError("candidate controls require actor and rationale")
        undo_records = self.session.scalars(
            select(BacklogCandidateDisposition)
            .where(
                BacklogCandidateDisposition.fingerprint == fingerprint,
                BacklogCandidateDisposition.record_kind == "control",
                BacklogCandidateDisposition.action == "undo",
            )
            .order_by(BacklogCandidateDisposition.sequence_no.desc())
        )
        undo_record = next(
            (
                record
                for record in undo_records
                if (
                    moved := self.session.scalar(
                        select(BacklogCandidateDisposition).where(
                            BacklogCandidateDisposition.frame_id == record.moved_frame_id
                        )
                    )
                )
                is not None
                and moved.stack_state == "reversible"
            ),
            None,
        )
        if undo_record is None:
            raise StaleHeadError("there is no reversible disposition to redo")
        frame = self.session.scalar(
            select(BacklogCandidateDisposition).where(
                BacklogCandidateDisposition.frame_id == undo_record.moved_frame_id
            )
        )
        if frame is None:  # pragma: no cover - foreign key is intentionally external frame id
            raise LifecycleConflictError("reversible frame has no immutable undo control record")
        before = self._snapshot(self._family(fingerprint))
        effective = self._collision_reconciled_image(dict(undo_record.preimage), before)
        self._apply_image(effective)
        frame.stack_state = "applied"
        control = self._control_record(
            fingerprint,
            action="redo",
            moved=frame,
            preimage=before,
            postimage=effective,
            actor=actor,
            rationale=rationale,
        )
        self.session.add(control)
        self.session.flush()
        return control

    def classify_revision(
        self,
        candidate_id: int,
        *,
        classification: str,
        proposed_content_snapshot: Mapping[str, Any],
        actor: str,
        rationale: str,
        classification_metadata: Mapping[str, Any] | None = None,
    ) -> BacklogCandidate | BacklogCandidateDisposition:
        """Apply D-8's cosmetic/shape-change/abstain boundary.

        An abstained proposal is retained as evidence metadata for explicit reviewer
        classification; it does not create a candidate or disposition frame.
        """
        candidate = self._candidate(candidate_id)
        metadata = {"classification": classification, **dict(classification_metadata or {})}
        if classification == "abstain":
            self._add_evidence(
                candidate.id,
                [
                    {
                        "kind": "proposed_revision",
                        "metadata": {**metadata, "content": dict(proposed_content_snapshot)},
                    }
                ],
            )
            self.session.flush()
            return candidate
        if classification == "cosmetic":
            old_content = dict(candidate.content_snapshot)
            frame = self.transition(
                candidate.id,
                action="edit",
                actor=actor,
                rationale=rationale,
                expected_head=self._applied_head(candidate.fingerprint).frame_id
                if self._applied_head(candidate.fingerprint)
                else None,
                payload={
                    "classification": metadata,
                    "old_content": old_content,
                    "new_content": dict(proposed_content_snapshot),
                },
            )
            candidate.content_snapshot = dict(proposed_content_snapshot)
            updated_postimage = {key: dict(value) for key, value in frame.postimage.items()}
            updated_postimage[str(candidate.id)]["content_snapshot"] = dict(
                proposed_content_snapshot
            )
            frame.postimage = updated_postimage
            self.session.flush()
            return frame
        if classification == "shape_change":
            replacement = self.create_candidate(
                fingerprint=candidate.fingerprint,
                source_snapshot=candidate.source_snapshot,
                content_snapshot=proposed_content_snapshot,
                assessment=candidate.assessment,
                configuration_snapshot=candidate.configuration_snapshot,
                topic_entity_id=candidate.topic_entity_id,
                digest_id=candidate.digest_id,
                active=False,
            )
            return self.transition(
                candidate.id,
                action="supersede",
                actor=actor,
                rationale=rationale,
                expected_head=self._applied_head(candidate.fingerprint).frame_id
                if self._applied_head(candidate.fingerprint)
                else None,
                replacement_candidate_id=replacement.id,
                payload={"classification": metadata},
            )
        raise CandidateRepositoryError(
            "revision classification must be cosmetic, shape_change, or abstain"
        )

    def _add_evidence(
        self, candidate_id: int, evidence: Iterable[Mapping[str, Any]]
    ) -> list[BacklogCandidateEvidence]:
        rows = []
        for item in evidence:
            kind = item.get("kind")
            if not isinstance(kind, str) or not kind:
                raise CandidateRepositoryError("evidence kind must be a non-empty string")
            row = BacklogCandidateEvidence(
                candidate_id=candidate_id,
                kind=kind,
                feed_item_id=item.get("feed_item_id"),
                excerpt=item.get("excerpt"),
                metadata_json=dict(item["metadata"]) if item.get("metadata") is not None else None,
            )
            self.session.add(row)
            rows.append(row)
        return rows

    def _candidate(self, candidate_id: int) -> BacklogCandidate:
        candidate = self.session.get(BacklogCandidate, candidate_id)
        if candidate is None:
            raise CandidateNotFoundError(f"candidate {candidate_id} does not exist")
        return candidate

    def _family(self, fingerprint: str) -> list[BacklogCandidate]:
        return list(
            self.session.scalars(
                select(BacklogCandidate)
                .where(BacklogCandidate.fingerprint == fingerprint)
                .order_by(BacklogCandidate.id)
            )
        )

    def _active_candidate(self, fingerprint: str) -> BacklogCandidate | None:
        return self.session.scalar(
            select(BacklogCandidate)
            .where(
                BacklogCandidate.fingerprint == fingerprint,
                BacklogCandidate.state.in_(ACTIVE_CANDIDATE_STATES),
            )
            .order_by(BacklogCandidate.id)
        )

    def _applied_head(self, fingerprint: str) -> BacklogCandidateDisposition | None:
        return self.session.scalar(
            select(BacklogCandidateDisposition)
            .where(
                BacklogCandidateDisposition.fingerprint == fingerprint,
                BacklogCandidateDisposition.record_kind == "disposition",
                BacklogCandidateDisposition.stack_state == "applied",
            )
            .order_by(BacklogCandidateDisposition.sequence_no.desc())
        )

    def _next_sequence(self, fingerprint: str) -> int:
        return (
            int(
                self.session.scalar(
                    select(
                        func.coalesce(func.max(BacklogCandidateDisposition.sequence_no), 0)
                    ).where(BacklogCandidateDisposition.fingerprint == fingerprint)
                )
                or 0
            )
            + 1
        )

    @staticmethod
    def _snapshot(family: Iterable[BacklogCandidate]) -> dict[str, dict[str, Any]]:
        return {
            str(candidate.id): {
                "state": candidate.state,
                "superseded_by_candidate_id": candidate.superseded_by_candidate_id,
                "merge_target_candidate_id": candidate.merge_target_candidate_id,
                "content_snapshot": dict(candidate.content_snapshot),
            }
            for candidate in family
        }

    def _derive_updates(
        self,
        candidate: BacklogCandidate,
        action: str,
        *,
        replacement_candidate_id: int | None,
        merge_target_candidate_id: int | None,
    ) -> dict[int, dict[str, Any]]:
        if action == "supersede":
            if replacement_candidate_id is None:
                raise CandidateRepositoryError("supersede requires a replacement candidate")
            replacement = self._candidate(replacement_candidate_id)
            if replacement.fingerprint != candidate.fingerprint or replacement.id == candidate.id:
                raise CandidateRepositoryError(
                    "supersession replacement must be another family member"
                )
            updates = {
                candidate.id: {"state": "superseded", "superseded_by_candidate_id": replacement.id},
                replacement.id: {"state": "draft", "superseded_by_candidate_id": None},
            }
        else:
            updates = {candidate.id: {"state": _TARGET_STATES[action]}}
            if action == "merge":
                updates[candidate.id]["merge_target_candidate_id"] = merge_target_candidate_id
        target_state = next(iter(updates.values()))["state"]
        if target_state in ACTIVE_CANDIDATE_STATES:
            active = self._active_candidate(candidate.fingerprint)
            if active is not None and active.id not in updates:
                updates[active.id] = {
                    "state": "superseded",
                    "superseded_by_candidate_id": candidate.id,
                }
        return updates

    @staticmethod
    def _updated_image(
        image: Mapping[str, Mapping[str, Any]], updates: Mapping[int, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        result = {key: dict(value) for key, value in image.items()}
        for candidate_id, values in updates.items():
            result[str(candidate_id)].update(values)
        return result

    def _apply_image(self, image: Mapping[str, Mapping[str, Any]]) -> None:
        # Partial unique indexes are checked per UPDATE on SQLite/Postgres.  Release
        # the old active holder before activating its replacement in the same
        # transaction so a valid whole-family transfer never transiently violates it.
        current = {candidate.id: candidate for candidate in self._family_for_image(image)}
        for candidate_id, values in image.items():
            candidate = current[int(candidate_id)]
            if (
                candidate.state in ACTIVE_CANDIDATE_STATES
                and values["state"] not in ACTIVE_CANDIDATE_STATES
            ):
                candidate.state = str(values["state"])
        self.session.flush()
        for candidate_id, values in image.items():
            candidate = current[int(candidate_id)]
            candidate.state = str(values["state"])
            candidate.superseded_by_candidate_id = values.get("superseded_by_candidate_id")
            candidate.merge_target_candidate_id = values.get("merge_target_candidate_id")
            if "content_snapshot" in values:
                candidate.content_snapshot = dict(values["content_snapshot"])
        try:
            self.session.flush()
        except IntegrityError as exc:
            raise LifecycleConflictError(
                "active fingerprint uniqueness race could not be reconciled"
            ) from exc

    def _family_for_image(self, image: Mapping[str, Mapping[str, Any]]) -> list[BacklogCandidate]:
        return [self._candidate(int(candidate_id)) for candidate_id in image]

    def _collision_reconciled_image(
        self, desired: Mapping[str, Mapping[str, Any]], current: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        result = {key: dict(value) for key, value in current.items()}
        for key, value in desired.items():
            result[key] = dict(value)
        active = [key for key, value in result.items() if value["state"] in ACTIVE_CANDIDATE_STATES]
        desired_active = [
            key for key, value in desired.items() if value["state"] in ACTIVE_CANDIDATE_STATES
        ]
        if len(desired_active) == 1:
            holder = desired_active[0]
            for key in active:
                if key != holder:
                    result[key]["state"] = "superseded"
                    result[key]["superseded_by_candidate_id"] = int(holder)
        elif len(active) > 1:
            raise LifecycleConflictError("family image has multiple active candidates")
        return result

    def _abandon_redo_future(self, fingerprint: str) -> list[str]:
        frames = list(
            self.session.scalars(
                select(BacklogCandidateDisposition).where(
                    BacklogCandidateDisposition.fingerprint == fingerprint,
                    BacklogCandidateDisposition.record_kind == "disposition",
                    BacklogCandidateDisposition.stack_state == "reversible",
                )
            )
        )
        for frame in frames:
            frame.stack_state = "inert"
        return [frame.frame_id for frame in frames]

    def _control_record(
        self,
        fingerprint: str,
        *,
        action: str,
        moved: BacklogCandidateDisposition,
        preimage: Mapping[str, Mapping[str, Any]],
        postimage: Mapping[str, Mapping[str, Any]],
        actor: str,
        rationale: str,
    ) -> BacklogCandidateDisposition:
        return BacklogCandidateDisposition(
            frame_id=str(uuid4()),
            fingerprint=fingerprint,
            sequence_no=self._next_sequence(fingerprint),
            parent_frame_id=self._applied_head(fingerprint).frame_id
            if self._applied_head(fingerprint)
            else None,
            record_kind="control",
            action=action,
            moved_frame_id=moved.frame_id,
            affected_candidate_ids=sorted(int(key) for key in postimage),
            preimage=dict(preimage),
            postimage=dict(postimage),
            abandoned_frame_ids=[],
            actor=actor,
            rationale=rationale,
            stack_state="audit",
        )


class CandidateTaskLinkRepository:
    """Fingerprint-scoped task links and append-only promotion-attempt journal."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_link(
        self,
        *,
        candidate_id: int,
        task_store: str,
        external_task_key: str,
        relationship_role: str | None = None,
        operation_id: str | None = None,
        observed: TaskInventoryItem | None = None,
        closure_class: str = "unknown",
        closure_class_source: str | None = None,
    ) -> BacklogCandidateTaskLink:
        candidate = self.session.get(BacklogCandidate, candidate_id)
        if candidate is None:
            raise CandidateNotFoundError(f"candidate {candidate_id} does not exist")
        link = self.session.scalar(
            select(BacklogCandidateTaskLink).where(
                BacklogCandidateTaskLink.fingerprint == candidate.fingerprint,
                BacklogCandidateTaskLink.task_store == task_store,
                BacklogCandidateTaskLink.external_task_key == external_task_key,
            )
        )
        if link is None:
            link = BacklogCandidateTaskLink(
                fingerprint=candidate.fingerprint,
                task_store=task_store,
                external_task_key=external_task_key,
                origin_candidate_id=candidate.id,
                last_reconciled_candidate_id=candidate.id,
                first_operation_id=operation_id,
            )
            self.session.add(link)
        link.last_reconciled_candidate_id = candidate.id
        link.relationship_role = relationship_role or link.relationship_role
        link.last_operation_id = operation_id or link.last_operation_id
        link.closure_class = (
            closure_class if closure_class in {"implemented", "rejected", "unknown"} else "unknown"
        )
        link.closure_class_source = closure_class_source
        if observed is not None:
            self._apply_observation(link, observed)
        self.session.flush()
        return link

    def links_for_candidate(self, candidate_id: int) -> list[BacklogCandidateTaskLink]:
        candidate = self.session.get(BacklogCandidate, candidate_id)
        if candidate is None:
            raise CandidateNotFoundError(f"candidate {candidate_id} does not exist")
        return list(
            self.session.scalars(
                select(BacklogCandidateTaskLink)
                .where(BacklogCandidateTaskLink.fingerprint == candidate.fingerprint)
                .order_by(BacklogCandidateTaskLink.id)
            )
        )

    def refresh_links(
        self, candidate_id: int, reader: TaskStoreReader, *, actor: str = "system:link-refresh"
    ) -> list[BacklogCandidateTaskLink]:
        links = self.links_for_candidate(candidate_id)
        inventory = reader.list_inventory()
        if not inventory.available:
            self.record_failed_attempt(
                candidate_id=candidate_id,
                actor=actor,
                error_payload={"inventory_unavailable": inventory.unavailable_reason},
            )
            raise LifecycleConflictError(
                "task link refresh could not obtain complete fresh inventory"
            )
        by_identity = {(item.task_store, item.external_task_key): item for item in inventory.items}
        for link in links:
            item = by_identity.get((link.task_store, link.external_task_key))
            if item is None:
                self.record_failed_attempt(
                    candidate_id=candidate_id,
                    actor=actor,
                    error_payload={
                        "identity_mismatch": f"{link.task_store}:{link.external_task_key}"
                    },
                )
                raise LifecycleConflictError(
                    "task link refresh returned a different external identity"
                )
            self._apply_observation(link, item)
        self.session.flush()
        return links

    def record_promotion(
        self,
        *,
        operation_id: str,
        candidate_id: int,
        primitives: Iterable[str],
        actor: str,
        requested_payload: Mapping[str, Any],
        selected_task_links: Iterable[Mapping[str, Any]] = (),
        step_receipts: Iterable[Mapping[str, Any]] = (),
        before_observation: Mapping[str, Any] | None = None,
        after_observation: Mapping[str, Any] | None = None,
        state: str = "completed",
    ) -> BacklogCandidatePromotion:
        """Append the attributable before/after receipt for a promotion operation.

        Phase 3 owns execution; Phase 1 owns its durable audit journal so no
        task-store write capability leaks into candidate lifecycle code.
        """
        candidate = self.session.get(BacklogCandidate, candidate_id)
        if candidate is None:
            raise CandidateNotFoundError(f"candidate {candidate_id} does not exist")
        attempt = BacklogCandidatePromotion(
            operation_id=operation_id,
            candidate_id=candidate.id,
            fingerprint=candidate.fingerprint,
            primitives=list(primitives),
            selected_task_links=[dict(link) for link in selected_task_links],
            requested_payload=dict(requested_payload),
            actor=actor,
            state=state,
            step_receipts=[dict(receipt) for receipt in step_receipts],
            correlation_markers={},
            before_observation=dict(before_observation) if before_observation is not None else None,
            after_observation=dict(after_observation) if after_observation is not None else None,
        )
        self.session.add(attempt)
        self.session.flush()
        return attempt

    def record_failed_attempt(
        self, *, candidate_id: int, actor: str, error_payload: Mapping[str, Any]
    ) -> BacklogCandidatePromotion:
        candidate = self.session.get(BacklogCandidate, candidate_id)
        if candidate is None:
            raise CandidateNotFoundError(f"candidate {candidate_id} does not exist")
        attempt = BacklogCandidatePromotion(
            operation_id=str(uuid4()),
            candidate_id=candidate.id,
            fingerprint=candidate.fingerprint,
            primitives=[],
            selected_task_links=[],
            requested_payload={},
            actor=actor,
            state="failed",
            step_receipts=[],
            correlation_markers={},
            error_payload=dict(error_payload),
        )
        self.session.add(attempt)
        self.session.flush()
        return attempt

    def begin_operation(
        self,
        *,
        operation_id: str,
        candidate_id: int,
        actor: str,
        primitives: Iterable[str],
        requested_payload: Mapping[str, Any],
        lease_seconds: int = 300,
    ) -> BacklogCandidatePromotion:
        """Return a durable pending operation, acquiring its candidate lease once."""
        existing = self.session.scalar(
            select(BacklogCandidatePromotion).where(
                BacklogCandidatePromotion.operation_id == operation_id
            )
        )
        if existing is not None:
            if (
                existing.candidate_id != candidate_id
                or existing.actor != actor
                or existing.primitives != list(primitives)
                or existing.requested_payload != dict(requested_payload)
            ):
                raise LifecycleConflictError(
                    f"promotion operation {operation_id} does not match this promotion request"
                )
            if existing.state == "failed":
                now = datetime.now(UTC)
                active = self.session.scalar(
                    select(BacklogCandidatePromotion).where(
                        BacklogCandidatePromotion.candidate_id == candidate_id,
                        BacklogCandidatePromotion.operation_id != operation_id,
                        BacklogCandidatePromotion.state == "pending",
                        BacklogCandidatePromotion.lease_expires_at > now,
                    )
                )
                if active is not None:
                    raise LifecycleConflictError(
                        f"candidate {candidate_id} already has live promotion operation "
                        f"{active.operation_id}"
                    )
                existing.state = "pending"
                existing.lease_expires_at = now + timedelta(seconds=lease_seconds)
                self.session.commit()
            return existing
        # The candidate row is the serialization anchor: PostgreSQL locks it for
        # the pending-operation check below (SQLite retains its single-writer
        # semantics in local/test deployments).
        candidate = self.session.scalar(
            select(BacklogCandidate).where(BacklogCandidate.id == candidate_id).with_for_update()
        )
        if candidate is None:
            raise CandidateNotFoundError(f"candidate {candidate_id} does not exist")
        now = datetime.now(UTC)
        active = self.session.scalar(
            select(BacklogCandidatePromotion).where(
                BacklogCandidatePromotion.candidate_id == candidate_id,
                BacklogCandidatePromotion.state == "pending",
                BacklogCandidatePromotion.lease_expires_at > now,
            )
        )
        if active is not None:
            raise LifecycleConflictError(
                f"candidate {candidate_id} already has live promotion operation {active.operation_id}"
            )
        attempt = BacklogCandidatePromotion(
            operation_id=operation_id,
            candidate_id=candidate.id,
            fingerprint=candidate.fingerprint,
            primitives=list(primitives),
            selected_task_links=[],
            requested_payload=dict(requested_payload),
            actor=actor,
            state="pending",
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            step_receipts=[],
            correlation_markers={},
        )
        self.session.add(attempt)
        self.session.commit()
        return attempt

    def checkpoint_operation(
        self,
        operation_id: str,
        *,
        step_receipts: Iterable[Mapping[str, Any]],
        correlation_markers: Mapping[str, Any],
    ) -> BacklogCandidatePromotion:
        """Durably record every completed external step before the next one starts."""
        attempt = self._pending_operation(operation_id)
        attempt.step_receipts = [dict(receipt) for receipt in step_receipts]
        attempt.correlation_markers = dict(correlation_markers)
        self.session.commit()
        return attempt

    def complete_operation(
        self,
        operation_id: str,
        *,
        selected_task_links: Iterable[Mapping[str, Any]] = (),
        step_receipts: Iterable[Mapping[str, Any]] = (),
        before_observation: Mapping[str, Any] | None = None,
        after_observation: Mapping[str, Any] | None = None,
        correlation_markers: Mapping[str, Any] | None = None,
    ) -> BacklogCandidatePromotion:
        """Complete exactly one pending operation and release its serialization lease."""
        attempt = self._pending_operation(operation_id)
        attempt.state = "completed"
        attempt.selected_task_links = [dict(link) for link in selected_task_links]
        attempt.step_receipts = [dict(receipt) for receipt in step_receipts]
        attempt.before_observation = (
            dict(before_observation) if before_observation is not None else None
        )
        attempt.after_observation = (
            dict(after_observation) if after_observation is not None else None
        )
        attempt.correlation_markers = dict(correlation_markers or {})
        attempt.lease_expires_at = None
        self.session.commit()
        return attempt

    def fail_operation(
        self, operation_id: str, *, error_payload: Mapping[str, Any]
    ) -> BacklogCandidatePromotion:
        """Record an operation failure and release its serialization lease."""
        attempt = self._pending_operation(operation_id)
        attempt.state = "failed"
        attempt.error_payload = dict(error_payload)
        attempt.lease_expires_at = None
        self.session.commit()
        return attempt

    def _pending_operation(self, operation_id: str) -> BacklogCandidatePromotion:
        attempt = self.session.scalar(
            select(BacklogCandidatePromotion).where(
                BacklogCandidatePromotion.operation_id == operation_id
            )
        )
        if attempt is None:
            raise LifecycleConflictError(f"promotion operation {operation_id} does not exist")
        if attempt.state != "pending":
            raise LifecycleConflictError(
                f"promotion operation {operation_id} is {attempt.state}, not pending"
            )
        return attempt

    @staticmethod
    def _apply_observation(link: BacklogCandidateTaskLink, item: TaskInventoryItem) -> None:
        if (link.task_store, link.external_task_key) != (item.task_store, item.external_task_key):
            raise LifecycleConflictError("task-store reader returned a mismatched task identity")
        link.last_observed_title = item.title
        link.last_observed_state = item.state
        link.last_observed_closure_reason = item.closure_reason_text
        link.checked_at = datetime.now(UTC)
