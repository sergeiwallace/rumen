"""T-05 seam: config-driven LLM command runner + JSON parsing (D-3 / AD-2)."""

from __future__ import annotations

import inspect

import pytest

from rumen import llm as llm_mod
from rumen.llm import (
    FallbackLlmCommand,
    SubprocessLlmCommand,
    parse_json_response,
)


class _FailLlm:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, prompt: str) -> str:
        self.calls += 1
        raise RuntimeError("backend down")


class _OkLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def run(self, prompt: str) -> str:
        self.calls += 1
        return self.response


def test_fallback_returns_first_backend_success() -> None:
    first, second = _OkLlm("primary"), _OkLlm("secondary")
    assert FallbackLlmCommand([first, second]).run("p") == "primary"
    assert second.calls == 0  # short-circuits — the second tier is never tried


def test_fallback_falls_through_to_next_backend_on_failure() -> None:
    down, up = _FailLlm(), _OkLlm("secondary")
    assert FallbackLlmCommand([down, up]).run("p") == "secondary"
    assert down.calls == 1 and up.calls == 1  # tried the primary, then fell to the next


def test_fallback_reraises_when_all_backends_fail() -> None:
    a, b = _FailLlm(), _FailLlm()
    with pytest.raises(RuntimeError, match="backend down"):
        FallbackLlmCommand([a, b]).run("p")
    assert a.calls == 1 and b.calls == 1  # every tier attempted before giving up


def test_fallback_requires_at_least_one_backend() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackLlmCommand([])


def test_subprocess_runs_exactly_the_configured_command() -> None:
    # `cat` echoes stdin → proves the prompt is passed to the configured argv on stdin.
    cmd = SubprocessLlmCommand(["cat"])
    assert cmd.run("hello prompt") == "hello prompt"


def test_empty_command_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SubprocessLlmCommand([])


def test_parse_json_response_plain() -> None:
    assert parse_json_response('{"category": "tool", "confidence": 0.9}') == {
        "category": "tool",
        "confidence": 0.9,
    }


def test_parse_json_response_tolerates_markdown_fence() -> None:
    fenced = '```json\n{"category": "paper"}\n```'
    assert parse_json_response(fenced) == {"category": "paper"}


def test_parse_json_response_raises_on_garbage() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_json_response("this is not json")


def test_llm_module_never_reads_environment() -> None:
    # No env reads at all → ANTHROPIC_API_KEY (or any metered key) is never picked up
    # implicitly; the subprocess inherits the operator's env unchanged.
    src = inspect.getsource(llm_mod)
    assert "os.environ" not in src
    assert "getenv" not in src
    assert "import os" not in src  # the module never touches the environment
