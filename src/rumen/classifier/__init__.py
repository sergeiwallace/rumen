"""Confidence-gated hybrid classification cascade (design D-12 / T-05).

Tier-0 deterministic rules → Tier-1 embedding classifier → Tier-2 gated LLM judge.
Tokens are spent only on genuinely ambiguous items (Tier-1 confidence < ``tau_class``).
``classification_method`` records which tier decided. Every exclusion is logged to
``dropped_items`` (never hard-deleted) — the false-negative audit.
"""

from rumen.classifier.cascade import (
    ClassificationResult,
    classify_item,
    classify_pending,
)

__all__ = ["ClassificationResult", "classify_item", "classify_pending"]
