"""Weak-supervision entry points deferred until a security-reviewed dependency is available."""

from __future__ import annotations

_UNAVAILABLE = (
    "Weak supervision is unavailable in this release because its dependency has unresolved "
    "security advisories. Use curated labels or the independent LLM teacher API."
)


def apply_labeling_functions(*args, **kwargs):
    """Report that probabilistic labeling is deferred."""
    raise RuntimeError(_UNAVAILABLE)


def combine_with_label_model(*args, **kwargs):
    """Report that probabilistic aggregation is deferred."""
    raise RuntimeError(_UNAVAILABLE)


def persist_snorkel_labels(*args, **kwargs):
    """Report that probabilistic label persistence is deferred."""
    raise RuntimeError(_UNAVAILABLE)


def run_bootstrap(*args, **kwargs):
    """Fail before importing optional dependencies or writing candidate labels."""
    raise RuntimeError(_UNAVAILABLE)
