"""Real cleanlab confident-learning gate for noisy T-16 bootstrap labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CleanlabGateResult:
    """Indices retained/excluded from one facet's bootstrap examples."""

    kept_indices: tuple[int, ...]
    excluded_indices: tuple[int, ...]


def gate_label_issues(
    labels: list[int], pred_probs: np.ndarray, *, noise_fraction: float
) -> CleanlabGateResult:
    """Exclude cleanlab's highest-noise examples for one facet.

    ``get_label_quality_scores`` performs confident learning using the supplied
    (out-of-sample) label probabilities.  The configured fraction is then selected from
    cleanlab's own self-confidence ranking, rather than approximated from bootstrap
    confidence values.
    """
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be in [0, 1]")
    if not labels:
        return CleanlabGateResult((), ())
    if pred_probs.shape != (len(labels), len(set(labels))):
        raise ValueError("pred_probs must have one row per label and one column per class")
    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError as exc:  # optional extra keeps base installs lightweight
        raise RuntimeError("cleanlab is required for HMC fitting; install rumen[hmc-fit]") from exc

    quality_scores = get_label_quality_scores(
        labels=np.asarray(labels),
        pred_probs=pred_probs,
        method="self_confidence",
    )
    number_to_exclude = int(np.ceil(len(labels) * noise_fraction))
    # Lowest cleanlab quality score = least confidence in the supplied bootstrap label.
    excluded_indices = tuple(
        sorted(
            int(index) for index in np.argsort(quality_scores, kind="stable")[:number_to_exclude]
        )
    )
    excluded_set = set(excluded_indices)
    return CleanlabGateResult(
        kept_indices=tuple(index for index in range(len(labels)) if index not in excluded_set),
        excluded_indices=excluded_indices,
    )
