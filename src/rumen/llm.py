"""Config-driven LLM command seam.

The engine invokes the configured command with the prompt on stdin and reads the
response from stdout. Tests inject a fake command. No provider is selected implicitly.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)


@runtime_checkable
class LlmCommand(Protocol):
    def run(self, prompt: str) -> str:
        """Return the model's raw stdout for ``prompt``."""
        ...


@dataclass(frozen=True, slots=True)
class LlmExecution:
    """Receipt from one concrete command execution."""

    response: str
    invocation_id: str
    provider_request_id: str
    context_id: str
    backing_context_id: str | None = None


@runtime_checkable
class LlmExecutionCommand(Protocol):
    """A command able to prove its concrete execution boundary.

    ``backing_context_id`` identifies the mutable session/conversation/connection
    backing this command, rather than the Python wrapper object. This is a receipt
    interface, not a trust grant: actionability admission accepts only the exact
    first-party adapters checked by ``is_trusted_actionability_command`` below.
    """

    @property
    def backing_context_id(self) -> str: ...

    def execute(self, prompt: str) -> LlmExecution: ...


class LlmUnavailable(RuntimeError):
    """Raised when the LLM budget is spent or the circuit breaker is open. Callers treat it
    like any other LLM failure — leave the item unclassified, never crash the run."""


class BudgetedLlmCommand:
    """Wrap an ``LlmCommand`` with a Tier-0-aware call budget and failure breaker.

    Tier-2 calls for a facet which has no deterministic Tier-0 route are expected work and may
    scale with a legitimate catch-up batch. Calls which bypass a deterministic Tier-0 route are
    instead limited by ``max_calls``: that is the regression shape which previously sent ~1300
    GitHub/npm items to the LLM. The consecutive-failure circuit breaker applies to both kinds
    of call, so an exhausted quota or unavailable backend is never hammered.

    - ``max_calls``: after this many unexpected Tier-2 invocations in one run, further
      unexpected calls fail fast (``LlmUnavailable``) without touching the backend.
    - ``max_consecutive_failures``: once the backend fails this many times in a row (quota
      hit, outage), the breaker opens and remaining calls fail fast — no point hammering a
      dead backend. A single success closes it again.

    ``calls`` / ``unexpected_calls`` / ``failures`` / ``breaker_open`` are readable for
    reporting. ``run()`` is retained for non-cascade callers and treats its call as unexpected.
    """

    def __init__(
        self, inner: LlmCommand, *, max_calls: int = 200, max_consecutive_failures: int = 5
    ) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.max_consecutive_failures = max_consecutive_failures
        self.calls = 0
        self.unexpected_calls = 0
        self.failures = 0
        self._consecutive_failures = 0
        self.breaker_open = False

    def tier2_available(self, *, expected: bool) -> bool:
        """Whether this route may reach the backend without spending a call."""
        return not self.breaker_open and (expected or self.unexpected_calls < self.max_calls)

    def for_tier2(self, *, expected: bool) -> LlmCommand:
        """Return an LLM seam which accounts a single Tier-2 route correctly."""
        return _RoutedBudgetedLlmCommand(self, expected=expected)

    def run(self, prompt: str) -> str:
        return self._run(prompt, expected=False)

    def _run(self, prompt: str, *, expected: bool) -> str:
        if not expected and self.unexpected_calls >= self.max_calls:
            raise LlmUnavailable(
                f"LLM unexpected-route budget exhausted ({self.max_calls} calls this run) — "
                "refusing further calls. This usually means a classifier-routing bug."
            )
        if self.breaker_open:
            raise LlmUnavailable(
                f"LLM circuit breaker open after {self._consecutive_failures} consecutive "
                "failures (quota/outage) — refusing further calls this run."
            )
        self.calls += 1
        if not expected:
            self.unexpected_calls += 1
        try:
            out = self.inner.run(prompt)
        except Exception:
            self.failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_consecutive_failures:
                self.breaker_open = True
            raise
        self._consecutive_failures = 0
        return out


class _RoutedBudgetedLlmCommand:
    """One Tier-2 route's accounting view of a ``BudgetedLlmCommand``."""

    def __init__(self, budget: BudgetedLlmCommand, *, expected: bool) -> None:
        self._budget = budget
        self._expected = expected

    def run(self, prompt: str) -> str:
        return self._budget._run(prompt, expected=self._expected)


class FallbackLlmCommand:
    """Try configured commands in order and return the first successful response.

    If every backend fails, re-raise the last exception. Compose under
    ``BudgetedLlmCommand`` so one budgeted call covers the whole ladder.
    """

    def __init__(self, commands: list[LlmCommand]) -> None:
        if not commands:
            raise ValueError("FallbackLlmCommand needs at least one backend command")
        self.commands = list(commands)
        self._backing_context_id = f"fallback-context:{uuid4()}"

    @property
    def backing_context_id(self) -> str:
        return self._backing_context_id

    def run(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for index, command in enumerate(self.commands):
            try:
                return command.run(prompt)
            except Exception as exc:  # noqa: BLE001 — any backend failure falls to the next tier
                last_exc = exc
                if index + 1 < len(self.commands):
                    logger.warning(
                        "LLM backend %d/%d failed (%s) — falling to the next tier",
                        index + 1,
                        len(self.commands),
                        exc,
                    )
        assert last_exc is not None  # commands is non-empty, so the loop ran at least once
        raise last_exc

    def execute(self, prompt: str) -> LlmExecution:
        """Return the receipt from the backend which actually served this request."""
        last_exc: Exception | None = None
        for index, command in enumerate(self.commands):
            if not isinstance(command, LlmExecutionCommand):
                raise LlmUnavailable("actionability backend cannot provide an execution receipt")
            try:
                return command.execute(prompt)
            except Exception as exc:  # noqa: BLE001 -- same ladder semantics as ``run``
                last_exc = exc
                if index + 1 < len(self.commands):
                    logger.warning(
                        "LLM backend %d/%d failed (%s) — falling to the next tier",
                        index + 1,
                        len(self.commands),
                        exc,
                    )
        assert last_exc is not None
        raise last_exc


class SubprocessLlmCommand:
    """Runs the configured command as a subprocess, prompt on stdin → stdout.

    ``command`` is the exact argv from config (e.g. ``["model-cli", "--model", "model-name"]``). No
    implicit fallback to a metered API; the environment is passed through unchanged
    (Rumen never injects ``ANTHROPIC_API_KEY``).
    """

    def __init__(self, command: list[str], *, timeout: float = 120.0) -> None:
        if not command:
            raise ValueError("LLM command must be a non-empty argv list")
        self.command = list(command)
        self.timeout = timeout
        self._context_id = f"subprocess-context:{uuid4()}"
        self._backing_context_id = f"subprocess-backing-context:{uuid4()}"

    @property
    def backing_context_id(self) -> str:
        return self._backing_context_id

    def run(self, prompt: str) -> str:
        return self.execute(prompt).response

    def execute(self, prompt: str) -> LlmExecution:
        """Run one isolated subprocess and retain its actual process identity."""
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            stdout, stderr = process.communicate(prompt, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, self.command, output=stdout, stderr=stderr
            )
        provider_request_id = f"subprocess:{process.pid}"
        return LlmExecution(
            response=stdout,
            invocation_id=f"invocation:{uuid4()}",
            provider_request_id=provider_request_id,
            context_id=self._context_id,
            backing_context_id=self._backing_context_id,
        )


def is_trusted_actionability_command(command: object) -> bool:
    """Return whether a command has Rumen's verified local process boundary.

    The actionability judge does not treat an arbitrary adapter's reported context
    identifier as proof of isolation. Exact first-party subprocess commands are
    trusted, as are exact first-party fallback ladders composed exclusively of those
    commands. Any other adapter is a deterministic abstention until its execution
    boundary has its own trusted-adapter review and regression coverage.
    """
    if type(command) is SubprocessLlmCommand:
        return True
    return type(command) is FallbackLlmCommand and all(
        type(inner) is SubprocessLlmCommand for inner in command.commands
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_response(text: str):
    """Parse a JSON object/array from an LLM response, tolerating ```json fences.

    Raises ``ValueError`` if no valid JSON is found — callers treat that as "leave the
    row unclassified / fail the digest cleanly", never as a silent mislabel.
    """
    candidate = text.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {text[:200]!r}") from exc
