"""Read-only reader contract coverage."""

from __future__ import annotations

import json
import subprocess

from rumen.backlog_mining.task_store import BeadsTaskStoreReader, TaskInventory
from rumen.models import BacklogCandidate


def test_beads_reader_when_all_statuses_are_unlimited_then_normalizes_without_write_commands(
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        status = command[command.index("--status") + 1]
        payload = json.dumps(
            [
                {
                    "id": f"TASK-{status}",
                    "title": status.title(),
                    "status": status,
                    **({"close_reason": "rejected"} if status == "closed" else {}),
                }
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    inventory = BeadsTaskStoreReader("rumen").list_inventory()
    assert inventory.available
    assert {(item.external_task_key, item.state) for item in inventory.items} == {
        (f"TASK-{status}", status)
        for status in ("open", "in_progress", "blocked", "deferred", "closed")
    }
    assert all(command[1] == "list" for command in commands)
    assert all(command[-2:] == ["--limit", "0"] for command in commands)
    assert not {"create", "update", "delete"} & {part for command in commands for part in command}


def test_beads_reader_when_inventory_is_incomplete_then_returns_unavailable_and_no_candidate_is_persisted(
    monkeypatch, session
) -> None:
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["bd", "list"])

    monkeypatch.setattr(subprocess, "run", fail)
    inventory = BeadsTaskStoreReader("rumen").list_inventory()
    assert not inventory.available
    assert inventory.unavailable_reason
    # Mining must gate candidate creation on this result.  Phase 1 exposes the
    # fail-closed result rather than silently turning an incomplete list into a draft.
    assert session.query(BacklogCandidate).count() == 0


def test_reader_protocol_when_unavailable_then_exposes_no_partial_items() -> None:
    inventory = TaskInventory.unavailable("connection refused")
    assert inventory.items == ()
    assert inventory.complete is False
    assert inventory.fresh is False
