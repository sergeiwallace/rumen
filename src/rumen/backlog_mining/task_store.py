"""Read-only normalized task-store inventory seam."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TaskInventoryItem:
    task_store: str
    external_task_key: str
    title: str
    state: str
    labels: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None
    closure_reason_text: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskInventory:
    items: tuple[TaskInventoryItem, ...] = ()
    complete: bool = False
    fresh: bool = False
    checked_at: datetime | None = None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.complete and self.fresh and self.unavailable_reason is None

    @classmethod
    def unavailable(cls, reason: str) -> TaskInventory:
        return cls(unavailable_reason=reason)


@runtime_checkable
class TaskStoreReader(Protocol):
    """A reader cannot create, update, delete, or otherwise mutate tasks."""

    def list_inventory(self) -> TaskInventory:
        """Return a complete, fresh open-and-closed inventory or an unavailable result."""


class BeadsTaskStoreReader:
    """Read-only adapter over ``bd list --json`` output.

    The exact command set is deliberately constrained to list queries.  Failures,
    malformed records, or an incomplete result fail closed rather than yielding a
    partial inventory to mining.
    """

    def __init__(self, task_store: str, executable: str = "bd") -> None:
        self.task_store = task_store
        self.executable = executable

    def _run_list(self, status: str) -> list[dict[str, object]]:
        completed = subprocess.run(
            [self.executable, "list", "--status", status, "--json", "--limit", "0"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("bd list did not return a JSON array of task objects")
        return payload

    def list_inventory(self) -> TaskInventory:
        try:
            rows = [
                row
                for status in ("open", "in_progress", "blocked", "deferred", "closed")
                for row in self._run_list(status)
            ]
            items = tuple(self._normalize(row) for row in rows)
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            return TaskInventory.unavailable(str(exc))
        return TaskInventory(items=items, complete=True, fresh=True, checked_at=datetime.now(UTC))

    def _normalize(self, row: dict[str, object]) -> TaskInventoryItem:
        key = row.get("id") or row.get("key")
        title = row.get("title")
        state = row.get("status") or row.get("state")
        if not all(isinstance(value, str) and value for value in (key, title, state)):
            raise ValueError("bd inventory record lacks id, title, or status")
        labels = row.get("labels") or []
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError("bd inventory record has invalid labels")
        return TaskInventoryItem(
            task_store=self.task_store,
            external_task_key=key,
            title=title,
            state=state,
            labels=tuple(labels),
            closure_reason_text=_optional_text(row.get("closed_reason") or row.get("close_reason")),
            metadata={
                key: value
                for key, value in row.items()
                if key not in {"id", "key", "title", "status", "state", "labels"}
            },
        )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None
