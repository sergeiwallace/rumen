"""Interest-profile centroid (design Stage 5 / D-2 / T-11).

An EMA-updated centroid over embeddings represents "what the user cares about",
**seeded from the bookmark corpus**. Ratings shift it: ``+1`` pulls the centroid
toward the item (``c ← α·x + (1−α)·c``); ``−1`` pushes it away at the same α.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.embeddings import EMBEDDING_DIM, decode_embedding, encode_embedding
from rumen.models import FeedItem, InterestProfileRow

DEFAULT_ALPHA = 0.1  # EMA learning rate (design D-2; parameter research)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class InterestProfile:
    """A single interest centroid with EMA updates (multi-centroid is a Phase-4 evolution)."""

    def __init__(self, centroid: np.ndarray | None = None, *, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        self.centroid = (
            _normalize(np.asarray(centroid, dtype=np.float32))
            if centroid is not None
            else np.zeros(EMBEDDING_DIM, dtype=np.float32)
        )

    @classmethod
    def seed_from(
        cls, vectors: list[np.ndarray], *, alpha: float = DEFAULT_ALPHA
    ) -> InterestProfile:
        """Seed the centroid as the mean of the configured bookmark corpus."""
        if not vectors:
            return cls(alpha=alpha)
        mean = np.mean(np.vstack(vectors), axis=0)
        return cls(mean, alpha=alpha)

    def update(self, vector: np.ndarray, rating: int) -> None:
        """Shift the centroid by a rating: ``+1`` toward ``vector``, ``−1`` away."""
        x = np.asarray(vector, dtype=np.float32)
        if rating > 0:
            self.centroid = _normalize(self.alpha * x + (1 - self.alpha) * self.centroid)
        elif rating < 0:
            self.centroid = _normalize(self.centroid - self.alpha * x)


def load_or_seed_profile(session: Session) -> InterestProfile:
    """Load the global profile, or seed and persist it from embedded bookmarks."""
    rows = list(session.scalars(select(InterestProfileRow)))
    if len(rows) > 1:
        raise RuntimeError("interest_profiles must contain exactly one row")
    if rows:
        row = rows[0]
        return InterestProfile(decode_embedding(row.centroid), alpha=row.alpha)

    vectors = [
        vector
        for blob in session.scalars(
            select(FeedItem.embedding).where(
                FeedItem.source == "bookmarks", FeedItem.embedding.is_not(None)
            )
        )
        if (vector := decode_embedding(blob)) is not None
    ]
    profile = InterestProfile.seed_from(vectors)
    session.add(
        InterestProfileRow(centroid=encode_embedding(profile.centroid), alpha=profile.alpha)
    )
    return profile
