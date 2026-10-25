"""Regression coverage for the repository's public GitHub Actions policy."""

from __future__ import annotations

import re
from pathlib import Path

CI_WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
AUTOMATIC_TRIGGER = re.compile(r"^  (?:push|pull_request):", re.MULTILINE)


def test_given_ci_workflow_when_checked_then_it_runs_for_pushes_and_pull_requests() -> None:
    workflow = CI_WORKFLOW.read_text()

    assert len(AUTOMATIC_TRIGGER.findall(workflow)) == 2
