"""The sole Beads-writing boundary for human-approved candidate promotions."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from rumen.backlog_mining.repository import CandidateTaskLinkRepository
from rumen.backlog_mining.review import render_candidate
from rumen.backlog_mining.task_store import BeadsTaskStoreReader
from rumen.models import BacklogCandidate, BacklogCandidateTaskLink


class PromotionError(ValueError):
    """A promotion was rejected before, or safely stopped during, a Beads write."""


_STATUS_VALUES = frozenset({"open", "in_progress", "blocked", "deferred", "closed"})
_EXISTING_PRIMITIVES = frozenset({"related", "shared-epic", "update", "status"})


@dataclass(frozen=True, slots=True)
class PromotionRequest:
    primitive: str
    actor: str
    existing_task: str | None = None
    operation_id: str | None = None
    body_file: str | None = None
    task_status: str | None = None
    reason: str | None = None
    new_task_title: str | None = None
    new_task_description_file: str | None = None
    new_task_priority: int | None = None
    new_task_status: str | None = None
    epic_title: str | None = None
    epic_description_file: str | None = None
    epic_priority: int | None = None
    epic_status: str | None = None
    approved_plan: str | None = None
    confirm_plan: str | None = None


@dataclass(slots=True)
class _Operation:
    candidate: BacklogCandidate
    request: PromotionRequest
    operation_id: str
    task_store: str
    existing: BacklogCandidateTaskLink | None
    existing_show: dict[str, Any] | None
    guarded: dict[str, bytes]
    receipts: list[dict[str, Any]] = field(default_factory=list)
    markers: dict[str, str] = field(default_factory=dict)


class PromotionExecutor:
    """Execute AC-DT-1 through one real ``bd`` subprocess boundary."""

    def __init__(self, session: Session, *, executable: str = "bd") -> None:
        self.session = session
        self.executable = executable

    def execute(self, candidate: BacklogCandidate, request: PromotionRequest) -> dict[str, Any]:
        """Validate and execute a single human-selected promotion operation."""
        if request.primitive not in {"new", *_EXISTING_PRIMITIVES}:
            raise PromotionError(f"unsupported promotion primitive {request.primitive!r}")
        task_store = candidate.configuration_snapshot.get("task_store")
        if not isinstance(task_store, str) or not task_store:
            raise PromotionError("candidate has no configured task-store identity")
        if (
            request.primitive == "shared-epic"
            and request.approved_plan
            and not request.operation_id
        ):
            raise PromotionError("--approved-plan requires an explicit --operation-id")
        operation_id = request.operation_id or str(uuid4())
        try:
            existing, existing_show = self._resolve_existing(candidate, request, task_store)
            linked_class = self._linked_class(existing, existing_show)
            self._validate_rule(request, linked_class, existing)
            guarded = self._capture_guard(request, existing_show)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PromotionError(f"Beads preflight failed: {exc}") from exc
        operation = _Operation(
            candidate=candidate,
            request=request,
            operation_id=operation_id,
            task_store=task_store,
            existing=existing,
            existing_show=existing_show,
            guarded=guarded,
        )
        try:
            plan = self._plan_for_shared_epic(operation)
        except OSError as exc:
            raise PromotionError(f"approved plan is unreadable: {exc}") from exc
        journal = CandidateTaskLinkRepository(self.session)
        attempt = journal.begin_operation(
            operation_id=operation_id,
            candidate_id=candidate.id,
            actor=request.actor,
            primitives=self._primitives(request),
            requested_payload=self._payload(request, plan),
        )
        if attempt.state != "pending":
            return {"operation_id": operation_id, "state": attempt.state, "resumed": True}
        operation.receipts = [dict(receipt) for receipt in attempt.step_receipts]
        operation.markers = dict(attempt.correlation_markers)
        try:
            if request.primitive == "new":
                self._create_candidate_task(operation, "new")
            elif request.primitive == "related":
                self._create_candidate_task(
                    operation, "related", related_to=existing.external_task_key
                )
            elif request.primitive == "update":
                self._update(operation)
                if request.task_status is not None:
                    self._change_status(operation)
            elif request.primitive == "status":
                self._change_status(operation)
            else:
                self._shared_epic(operation, plan)
            links = self._persist_links(operation)
            completed = journal.complete_operation(
                operation_id,
                selected_task_links=links,
                step_receipts=operation.receipts,
                before_observation=operation.existing_show,
                after_observation=self._after_observation(operation),
                correlation_markers=operation.markers,
            )
            return {"operation_id": completed.operation_id, "state": completed.state}
        except Exception as exc:
            journal.fail_operation(operation_id, error_payload={"error": str(exc)})
            if isinstance(exc, PromotionError):
                raise
            raise PromotionError(str(exc)) from exc

    def _resolve_existing(
        self,
        candidate: BacklogCandidate,
        request: PromotionRequest,
        task_store: str,
    ) -> tuple[BacklogCandidateTaskLink | None, dict[str, Any] | None]:
        if request.primitive == "new":
            if request.existing_task is not None:
                raise PromotionError("new promotion must not select an existing task")
            return None, None
        if not request.existing_task:
            raise PromotionError(
                "an explicit --existing-task selection is required before any Beads write"
            )
        store, key = self._split_task_key(request.existing_task)
        if store != task_store:
            raise PromotionError("selected task-store does not match the candidate task-store")
        inventory = BeadsTaskStoreReader(task_store, self.executable).list_inventory()
        if not inventory.available:
            raise PromotionError(
                f"task selection could not be refreshed: {inventory.unavailable_reason}"
            )
        selected = [
            item
            for item in inventory.items
            if (item.task_store, item.external_task_key) == (store, key)
        ]
        if len(selected) != 1:
            raise PromotionError(
                "explicit task selection is missing or ambiguous; select one refreshed task"
            )
        links = CandidateTaskLinkRepository(self.session)
        prior = next(
            (
                link
                for link in links.links_for_candidate(candidate.id)
                if (link.task_store, link.external_task_key) == (store, key)
            ),
            None,
        )
        link = links.upsert_link(
            candidate_id=candidate.id,
            task_store=store,
            external_task_key=key,
            observed=selected[0],
            closure_class=prior.closure_class if prior is not None else "unknown",
            closure_class_source=prior.closure_class_source if prior is not None else None,
        )
        return link, self._show(key)

    @staticmethod
    def _split_task_key(value: str) -> tuple[str, str]:
        store, separator, key = value.partition(":")
        if not separator or not store or not key:
            raise PromotionError("--existing-task must be <store:key>")
        return store, key

    @staticmethod
    def _linked_class(
        link: BacklogCandidateTaskLink | None, observed: dict[str, Any] | None
    ) -> str:
        if link is None or observed is None:
            return "none"
        if observed.get("status") != "closed":
            return "not_done"
        if link.closure_class == "implemented":
            return "done_implemented"
        if link.closure_class == "rejected":
            return "closed_rejected"
        return "closed_unknown"

    @staticmethod
    def _validate_rule(
        request: PromotionRequest, linked_class: str, existing: BacklogCandidateTaskLink | None
    ) -> None:
        if request.primitive in _EXISTING_PRIMITIVES and existing is None:
            raise PromotionError("an explicit refreshed task selection is required")
        if linked_class == "done_implemented" and request.primitive in {"update", "status"}:
            raise PromotionError(
                "completed implemented tasks may not be edited, reopened, or status-changed"
            )
        if linked_class == "closed_unknown" and request.primitive != "related":
            raise PromotionError("closed task must be classified before this promotion primitive")
        if request.primitive == "update" and request.body_file is None:
            raise PromotionError("update promotion requires --body-file")
        if request.primitive == "update" and request.task_status not in {None, "open"}:
            raise PromotionError("update may only compose status=open for rejected-task revival")
        if request.task_status is not None and request.task_status not in _STATUS_VALUES:
            raise PromotionError(
                "task status must be open, in_progress, blocked, deferred, or closed"
            )
        if request.primitive == "status" and request.task_status is None:
            raise PromotionError("status promotion requires --task-status")
        if request.primitive in {"status", "update"} and request.task_status is not None:
            if not request.reason:
                raise PromotionError("status-changing promotion requires --reason")

    @staticmethod
    def _capture_guard(
        request: PromotionRequest, observed: dict[str, Any] | None
    ) -> dict[str, bytes]:
        if observed is None:
            return {}
        fields: set[str] = set()
        if request.primitive == "update":
            fields.update({"title", "description"})
        if request.primitive in {"status", "update"} and request.task_status is not None:
            fields.add("status")
        if request.primitive == "status":
            fields.add("status")
        if request.primitive == "shared-epic":
            fields.add("parent")
        return {field: _field_bytes(observed.get(field)) for field in fields}

    def _create_candidate_task(
        self, operation: _Operation, role: str, *, related_to: str | None = None
    ) -> str:
        draft = render_candidate(self.session, operation.candidate.id)
        marker = self._marker(operation.operation_id, role)
        command = [
            "create",
            "--title",
            draft.title,
            "--description",
            draft.body,
            "--external-ref",
            marker,
        ]
        if related_to is not None:
            command.extend(["--deps", f"relates-to:{related_to}"])
        task_id = self._recorded_task(operation, f"create:{role}") or self._adopt_or_create(
            marker, command
        )
        operation.markers[role] = marker
        self._record_step(operation, {"step": f"create:{role}", "task": task_id})
        return task_id

    def _update(self, operation: _Operation) -> None:
        assert operation.existing is not None
        assert operation.request.body_file is not None
        body_file = Path(operation.request.body_file)
        if not body_file.is_file():
            raise PromotionError("--body-file must name a readable regular file")
        draft = render_candidate(self.session, operation.candidate.id)
        desired_body = body_file.read_text()
        current = self._show(operation.existing.external_task_key)
        if current.get("title") == draft.title and current.get("description") == desired_body:
            self._record_step(
                operation,
                {"step": "update", "task": operation.existing.external_task_key, "adopted": True},
            )
            return
        self._guard(operation, operation.existing.external_task_key, {"title", "description"})
        self._run(
            "update",
            operation.existing.external_task_key,
            "--title",
            draft.title,
            "--body-file",
            str(body_file),
        )
        self._record_step(
            operation, {"step": "update", "task": operation.existing.external_task_key}
        )

    def _change_status(self, operation: _Operation) -> None:
        assert operation.existing is not None
        status = operation.request.task_status
        assert status is not None
        observed = self._show(operation.existing.external_task_key)
        current = observed.get("status")
        if current == status:
            self._record_step(
                operation,
                {"step": "status", "task": operation.existing.external_task_key, "adopted": True},
            )
            return
        self._guard(operation, operation.existing.external_task_key, {"status"})
        reason = operation.request.reason
        assert reason is not None
        if current == "closed":
            if status != "open":
                raise PromotionError("closed tasks may only be reopened to open in this operation")
            self._run("reopen", operation.existing.external_task_key, "--reason", reason)
        elif status == "closed":
            self._run("close", operation.existing.external_task_key, "--reason", reason)
        else:
            self._run("update", operation.existing.external_task_key, "--status", status)
        self._record_step(
            operation, {"step": "status", "task": operation.existing.external_task_key}
        )

    def _plan_for_shared_epic(self, operation: _Operation) -> dict[str, Any] | None:
        if operation.request.primitive != "shared-epic":
            return None
        request = operation.request
        if request.approved_plan:
            try:
                plan = json.loads(Path(request.approved_plan).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise PromotionError(f"--approved-plan is invalid: {exc}") from exc
        else:
            required = (
                request.operation_id,
                request.new_task_title,
                request.new_task_description_file,
                request.new_task_priority,
                request.new_task_status,
                request.epic_title,
                request.epic_description_file,
                request.epic_priority,
                request.epic_status,
                request.confirm_plan,
            )
            if any(value is None for value in required):
                raise PromotionError(
                    "shared-epic requires a complete approved plan and confirmation"
                )
            plan = self._flag_plan(operation)
        self._validate_plan(plan, operation)
        canonical = _canonical_bytes(plan)
        if request.confirm_plan != sha256(canonical).hexdigest():
            raise PromotionError("--confirm-plan does not match the canonical approved plan")
        return plan

    def _flag_plan(self, operation: _Operation) -> dict[str, Any]:
        request = operation.request
        assert operation.existing is not None and operation.existing_show is not None
        assert request.new_task_title is not None and request.new_task_description_file is not None
        assert request.new_task_priority is not None and request.new_task_status is not None
        assert request.epic_title is not None and request.epic_description_file is not None
        assert request.epic_priority is not None and request.epic_status is not None
        return {
            "version": 1,
            "candidate": {
                "id": operation.candidate.id,
                "fingerprint": operation.candidate.fingerprint,
            },
            "operation_id": operation.operation_id,
            "task_store": operation.task_store,
            "new_child": {
                "title": request.new_task_title,
                "description": Path(request.new_task_description_file).read_text(),
                "priority": request.new_task_priority,
                "status": request.new_task_status,
                "task_store": operation.task_store,
                "external_ref": self._marker(operation.operation_id, "new-child"),
            },
            "existing_child": {
                "task_store": operation.task_store,
                "external_task_key": operation.existing.external_task_key,
                "title": operation.existing_show.get("title"),
                "parent": operation.existing_show.get("parent"),
                "closure_class": operation.existing.closure_class,
            },
            "new_epic": {
                "title": request.epic_title,
                "description": Path(request.epic_description_file).read_text(),
                "priority": request.epic_priority,
                "status": request.epic_status,
                "task_store": operation.task_store,
                "external_ref": self._marker(operation.operation_id, "epic"),
            },
            "child_topology": {
                "new_child_external_ref": self._marker(operation.operation_id, "new-child"),
                "existing_child": f"{operation.task_store}:{operation.existing.external_task_key}",
                "epic_external_ref": self._marker(operation.operation_id, "epic"),
                "replaces_existing_parent": operation.existing_show.get("parent") is not None,
            },
        }

    def _validate_plan(self, plan: Any, operation: _Operation) -> None:
        if not isinstance(plan, dict):
            raise PromotionError("approved plan must be a JSON object")
        required = {
            "version",
            "candidate",
            "operation_id",
            "task_store",
            "new_child",
            "existing_child",
            "new_epic",
            "child_topology",
        }
        if set(plan) != required or plan.get("version") != 1:
            raise PromotionError("approved plan does not match the versioned shared-epic schema")
        expected_identity = {
            "id": operation.candidate.id,
            "fingerprint": operation.candidate.fingerprint,
        }
        if (
            plan["candidate"] != expected_identity
            or plan["operation_id"] != operation.operation_id
            or plan["task_store"] != operation.task_store
        ):
            raise PromotionError("approved plan identity does not match this promotion")
        for name, role in (("new_child", "new-child"), ("new_epic", "epic")):
            value = plan[name]
            if not isinstance(value, dict) or set(value) != {
                "title",
                "description",
                "priority",
                "status",
                "task_store",
                "external_ref",
            }:
                raise PromotionError(f"approved plan {name} is incomplete")
            if (
                not isinstance(value["title"], str)
                or not isinstance(value["description"], str)
                or value["priority"] not in range(5)
            ):
                raise PromotionError(f"approved plan {name} has invalid content")
            if (
                value["status"] != "open"
                or value["task_store"] != operation.task_store
                or value["external_ref"] != self._marker(operation.operation_id, role)
            ):
                raise PromotionError(f"approved plan {name} has invalid initial identity or status")
        existing = plan["existing_child"]
        assert operation.existing is not None and operation.existing_show is not None
        expected_existing = {"task_store", "external_task_key", "title", "parent", "closure_class"}
        if not isinstance(existing, dict) or set(existing) != expected_existing:
            raise PromotionError("approved plan existing child is incomplete")
        current_existing = {
            "task_store": operation.task_store,
            "external_task_key": operation.existing.external_task_key,
            "title": operation.existing_show.get("title"),
            "parent": operation.existing_show.get("parent"),
            "closure_class": operation.existing.closure_class,
        }
        if existing != current_existing and not self._is_resumed_epic_parent(
            existing, current_existing, plan["new_epic"]["external_ref"]
        ):
            raise PromotionError("approved plan existing child is stale")
        topology = plan["child_topology"]
        if not isinstance(topology, dict) or set(topology) != {
            "new_child_external_ref",
            "existing_child",
            "epic_external_ref",
            "replaces_existing_parent",
        }:
            raise PromotionError("approved plan topology is incomplete")
        if topology != {
            "new_child_external_ref": self._marker(operation.operation_id, "new-child"),
            "existing_child": f"{operation.task_store}:{operation.existing.external_task_key}",
            "epic_external_ref": self._marker(operation.operation_id, "epic"),
            "replaces_existing_parent": existing["parent"] is not None,
        }:
            raise PromotionError("approved plan topology is not explicitly confirmed")

    def _is_resumed_epic_parent(
        self,
        planned: dict[str, Any],
        current: dict[str, Any],
        epic_marker: str,
    ) -> bool:
        if {key: value for key, value in planned.items() if key != "parent"} != {
            key: value for key, value in current.items() if key != "parent"
        }:
            return False
        try:
            epic_id = self._find_marker(epic_marker)
        except PromotionError:
            return False
        return current["parent"] == epic_id

    def _shared_epic(self, operation: _Operation, plan: dict[str, Any] | None) -> None:
        assert plan is not None and operation.existing is not None
        epic = plan["new_epic"]
        child = plan["new_child"]
        epic_id = self._recorded_task(operation, "create:epic") or self._adopt_or_create(
            epic["external_ref"],
            [
                "create",
                "--type",
                "epic",
                "--title",
                epic["title"],
                "--description",
                epic["description"],
                "--priority",
                str(epic["priority"]),
                "--external-ref",
                epic["external_ref"],
            ],
            expected_type="epic",
        )
        self._require_task_type(epic_id, "epic")
        operation.markers["epic"] = epic["external_ref"]
        self._record_step(operation, {"step": "create:epic", "task": epic_id})
        child_id = self._recorded_task(operation, "create:new-child") or self._adopt_or_create(
            child["external_ref"],
            [
                "create",
                "--title",
                child["title"],
                "--description",
                child["description"],
                "--priority",
                str(child["priority"]),
                "--external-ref",
                child["external_ref"],
            ],
        )
        operation.markers["new-child"] = child["external_ref"]
        self._record_step(operation, {"step": "create:new-child", "task": child_id})
        child_before = self._show(child_id)
        if child_before.get("parent") != epic_id:
            self._guard_values(child_id, {"parent": _field_bytes(child_before.get("parent"))})
            self._run("update", child_id, "--parent", epic_id)
            self._record_step(
                operation, {"step": "parent:new-child", "task": child_id, "parent": epic_id}
            )
        else:
            self._record_step(
                operation, {"step": "parent:new-child", "task": child_id, "adopted": True}
            )
        existing_before = self._show(operation.existing.external_task_key)
        if existing_before.get("parent") != epic_id:
            self._guard(operation, operation.existing.external_task_key, {"parent"})
            self._run("update", operation.existing.external_task_key, "--parent", epic_id)
            self._record_step(
                operation,
                {
                    "step": "parent:existing-child",
                    "task": operation.existing.external_task_key,
                    "parent": epic_id,
                },
            )
        else:
            self._record_step(
                operation,
                {
                    "step": "parent:existing-child",
                    "task": operation.existing.external_task_key,
                    "adopted": True,
                },
            )

    def _guard(self, operation: _Operation, task_id: str, fields: set[str]) -> None:
        expected = {field: operation.guarded[field] for field in fields}
        self._guard_values(task_id, expected)

    def _guard_values(self, task_id: str, expected: dict[str, bytes]) -> None:
        current = self._show(task_id)
        changed = [
            field
            for field, before in expected.items()
            if _field_bytes(current.get(field)) != before
        ]
        if changed:
            raise PromotionError(
                f"task {task_id} changed fields {', '.join(sorted(changed))}; fresh confirmation is required"
            )

    def _persist_links(self, operation: _Operation) -> list[dict[str, Any]]:
        repo = CandidateTaskLinkRepository(self.session)
        links: list[dict[str, Any]] = []
        if operation.existing is not None:
            links.append(_link_payload(operation.existing))
        for role, marker in operation.markers.items():
            task_id = self._find_marker(marker)
            link = repo.upsert_link(
                candidate_id=operation.candidate.id,
                task_store=operation.task_store,
                external_task_key=task_id,
                relationship_role=role,
                operation_id=operation.operation_id,
            )
            links.append(_link_payload(link))
        return links

    def _after_observation(self, operation: _Operation) -> dict[str, Any] | None:
        if operation.existing is None:
            return None
        return self._show(operation.existing.external_task_key)

    def _find_marker(self, marker: str) -> str:
        matches = self._marker_matches(marker)
        if len(matches) != 1:
            raise PromotionError(f"correlation marker {marker} has {len(matches)} matching tasks")
        return matches[0]

    def _adopt_or_create(
        self, marker: str, command: list[str], *, expected_type: str | None = None
    ) -> str:
        """Recover one prior create, or issue exactly one new create under the lease."""
        matches = self._marker_matches(marker)
        if len(matches) == 1:
            if expected_type is not None:
                self._require_task_type(matches[0], expected_type)
            return matches[0]
        if len(matches) > 1:
            raise PromotionError(f"correlation marker {marker} has multiple matching tasks")
        self._run(*command)
        task_id = self._find_marker(marker)
        if expected_type is not None:
            self._require_task_type(task_id, expected_type)
        return task_id

    def _require_task_type(self, task_id: str, expected_type: str) -> None:
        task = self._show(task_id)
        task_type = task.get("issue_type") or task.get("type")
        if task_type != expected_type:
            raise PromotionError(
                f"correlation marker adopted task {task_id} is not an {expected_type}"
            )

    @staticmethod
    def _recorded_task(operation: _Operation, step: str) -> str | None:
        for receipt in reversed(operation.receipts):
            if receipt.get("step") == step and isinstance(receipt.get("task"), str):
                return receipt["task"]
        return None

    def _record_step(self, operation: _Operation, receipt: dict[str, Any]) -> None:
        if not any(existing.get("step") == receipt["step"] for existing in operation.receipts):
            operation.receipts.append(receipt)
        CandidateTaskLinkRepository(self.session).checkpoint_operation(
            operation.operation_id,
            step_receipts=operation.receipts,
            correlation_markers=operation.markers,
        )

    def _marker_matches(self, marker: str) -> list[str]:
        matches: list[str] = []
        for row in self._list(external_ref=marker):
            if row.get("external_ref") == marker:
                task_id = row.get("id") or row.get("key")
                if isinstance(task_id, str) and task_id:
                    matches.append(task_id)
        return matches

    @staticmethod
    def _marker(operation_id: str, role: str) -> str:
        return f"rumen-promotion:{operation_id}:{role}"

    @staticmethod
    def _primitives(request: PromotionRequest) -> list[str]:
        return (
            ["update", "status"]
            if request.primitive == "update" and request.task_status
            else [request.primitive]
        )

    @staticmethod
    def _payload(request: PromotionRequest, plan: dict[str, Any] | None) -> dict[str, Any]:
        payload = {name: value for name, value in asdict(request).items() if value is not None}
        if plan is not None:
            payload["approved_plan"] = plan
        return payload

    def _show(self, task_id: str) -> dict[str, Any]:
        payload = self._json_command("show", task_id, "--json")
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise PromotionError("bd show did not return exactly one JSON object")
        return payload[0]

    def _list(self, *, external_ref: str | None = None) -> list[dict[str, Any]]:
        args = ["list", "--all", "--limit", "0"]
        if external_ref is not None:
            args.extend(["--external-ref", external_ref])
        args.append("--json")
        payload = self._json_command(*args)
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise PromotionError("bd list did not return a JSON array")
        return payload

    def _json_command(self, *args: str) -> Any:
        completed = self._run(*args)
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PromotionError(f"bd {' '.join(args)} returned malformed JSON") from exc

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([self.executable, *args], check=True, capture_output=True, text=True)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _field_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def _link_payload(link: BacklogCandidateTaskLink) -> dict[str, Any]:
    return {"task_store": link.task_store, "external_task_key": link.external_task_key}
