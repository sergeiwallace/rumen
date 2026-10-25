"""The base install (no optional extras) must reach the documented `rumen run` quick start."""

from __future__ import annotations

import subprocess
import sys

_BLOCK_OPTIONAL_EXTRAS = """
import sys
for name in ("rdflib", "skosify"):
    sys.modules[name] = None  # an ImportError for anything that imports it
import rumen.cli
import rumen.jobs
"""


def test_given_taxonomy_extra_missing_when_cli_and_jobs_import_then_the_run_path_loads() -> None:
    """Regression: `rumen run` imported the taxonomy audit, so a base install crashed on rdflib.

    The README quick start installs without extras; the taxonomy audit is a separate job that
    needs the optional `taxonomy` extra and must be imported only where it runs.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCK_OPTIONAL_EXTRAS],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
