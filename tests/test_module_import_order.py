"""Every public module must import cleanly as the first import of a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["rumen.taxonomy.hmc.fit", "rumen.classifier.cascade", "rumen.classifier"],
)
def test_given_a_fresh_interpreter_when_importing_first_then_the_module_imports(module):
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr[-400:]
