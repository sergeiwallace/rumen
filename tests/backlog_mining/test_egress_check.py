from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from rumen.backlog_mining.egress_check import verify_no_network_egress
from rumen.llm import SubprocessLlmCommand


def _user_network_namespaces_available() -> bool:
    """Probe whether this host actually permits the unprivileged sandbox under test.

    Some hardened kernels (and several hosted CI runner images) disable unprivileged user
    namespaces outright, so `unshare --user` fails with EPERM before the code under test ever
    runs. That is an environment capability, not a regression in `verify_no_network_egress`
    itself -- it already reports this exact case as `failure="sandbox-unavailable"`.
    """
    unshare = shutil.which("unshare")
    true = shutil.which("true")
    if unshare is None or true is None:
        return False
    probe = subprocess.run(
        [unshare, "--user", "--map-root-user", "--net", "--", true],
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    return probe.returncode == 0


requires_user_netns = pytest.mark.skipif(
    not _user_network_namespaces_available(),
    reason="unprivileged user+network namespaces are unavailable on this host/runner",
)


def _response_is_ok(response: str) -> bool:
    return response == "local response\n"


@requires_user_netns
def test_local_command_when_network_namespace_denies_egress_then_valid_response_passes() -> None:
    command = SubprocessLlmCommand(
        [sys.executable, "-c", "import sys; sys.stdin.read(); print('local response')"]
    )

    result = verify_no_network_egress(command, "prompt", response_is_valid=_response_is_ok)

    assert result.sandbox_established is True
    assert result.passed is True
    assert result.response == "local response\n"
    assert result.failure is None


@requires_user_netns
def test_networking_command_when_network_namespace_denies_egress_then_failure_is_reported() -> None:
    command = SubprocessLlmCommand(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import os
                import socket
                import sys

                sys.stdin.read()
                try:
                    socket.create_connection(("203.0.113.1", 443), timeout=1)
                except OSError:
                    print(f"network-attempt-blocked:{os.readlink('/proc/self/ns/net')}")
                    raise SystemExit(1)
                raise SystemExit("network connection unexpectedly succeeded")
                """
            ),
        ]
    )

    result = verify_no_network_egress(command, "prompt", response_is_valid=_response_is_ok)

    assert result.sandbox_established is True
    assert result.passed is False
    assert result.failure == "command-failed"
    assert result.returncode != 0
    assert result.response is not None
    namespace = result.response.removeprefix("network-attempt-blocked:").strip()
    assert namespace != os.readlink("/proc/self/ns/net")
