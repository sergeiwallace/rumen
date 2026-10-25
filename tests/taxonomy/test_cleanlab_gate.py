"""T-17 AC 1: cleanlab removes the configured noisy fraction per facet."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cleanlab")

from rumen.taxonomy.hmc.cleanlab_gate import gate_label_issues


def test_cleanlab_gate_when_noise_fraction_configured_then_excludes_lowest_quality_examples() -> (
    None
):
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    probabilities = np.array(
        [
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.01, 0.99],
            [0.02, 0.98],
            [0.01, 0.99],
            [0.02, 0.98],
            [0.03, 0.97],
            [0.99, 0.01],
            [0.98, 0.02],
        ]
    )

    result = gate_label_issues(labels, probabilities, noise_fraction=0.2)

    assert len(result.excluded_indices) == 2
    assert set(result.excluded_indices).issubset({3, 4, 5, 6, 8, 9})
    assert len(result.kept_indices) == 8
