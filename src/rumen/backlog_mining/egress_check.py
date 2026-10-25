"""Diagnostic verification for trusted local actionability subprocesses.

This is deliberately not part of the actionability hot path.  It gives an operator or
CI a fail-closed way to check whether one configured local command can answer while
running in a fresh Linux network namespace.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from rumen.llm import SubprocessLlmCommand

_NAMESPACE_ARGUMENTS = ("--user", "--map-root-user", "--net", "--")
_SANDBOX_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class EgressCheckResult:
    """Outcome of one no-network execution attempt.

    ``sandbox_established`` is true only after a fresh namespace preflight succeeds;
    the utility never runs a configured command outside that namespace as a fallback.
    """

    command: tuple[str, ...]
    sandbox_established: bool
    passed: bool
    response: str | None
    stderr: str
    returncode: int | None
    failure: str | None


def verify_no_network_egress(
    command: SubprocessLlmCommand,
    prompt: str,
    *,
    response_is_valid: Callable[[str], bool],
) -> EgressCheckResult:
    """Run one exact trusted subprocess command with all network namespaces isolated.

    A user namespace makes the check usable by an unprivileged Linux user while the
    nested network namespace starts with no active network route. A successful response proves
    this particular argv can serve the supplied probe without network access; it does
    not make a claim about a remote backend or about untested prompt shapes.
    """
    if type(command) is not SubprocessLlmCommand:
        raise TypeError("egress checks require an exact SubprocessLlmCommand")

    unshare = shutil.which("unshare")
    true = shutil.which("true")
    argv = tuple(command.command)
    if unshare is None or true is None:
        return EgressCheckResult(
            argv,
            sandbox_established=False,
            passed=False,
            response=None,
            stderr="unshare or true executable is unavailable",
            returncode=None,
            failure="sandbox-unavailable",
        )

    try:
        probe = subprocess.run(
            [unshare, *_NAMESPACE_ARGUMENTS, true],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=min(command.timeout, _SANDBOX_PROBE_TIMEOUT_SECONDS),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return EgressCheckResult(
            argv,
            sandbox_established=False,
            passed=False,
            response=None,
            stderr=exc.stderr or "network namespace preflight timed out",
            returncode=None,
            failure="sandbox-unavailable",
        )
    if probe.returncode != 0:
        return EgressCheckResult(
            argv,
            sandbox_established=False,
            passed=False,
            response=None,
            stderr=probe.stderr,
            returncode=probe.returncode,
            failure="sandbox-unavailable",
        )

    process = subprocess.Popen(
        [unshare, *_NAMESPACE_ARGUMENTS, *argv],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        response, stderr = process.communicate(prompt, timeout=command.timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate()
        return EgressCheckResult(
            argv,
            sandbox_established=True,
            passed=False,
            response=None,
            stderr=stderr,
            returncode=None,
            failure="command-timed-out",
        )
    if process.returncode != 0:
        return EgressCheckResult(
            argv,
            sandbox_established=True,
            passed=False,
            response=response,
            stderr=stderr,
            returncode=process.returncode,
            failure="command-failed",
        )
    if not response_is_valid(response):
        return EgressCheckResult(
            argv,
            sandbox_established=True,
            passed=False,
            response=response,
            stderr=stderr,
            returncode=process.returncode,
            failure="invalid-response",
        )
    return EgressCheckResult(
        argv,
        sandbox_established=True,
        passed=True,
        response=response,
        stderr=stderr,
        returncode=process.returncode,
        failure=None,
    )
