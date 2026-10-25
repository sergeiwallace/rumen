"""Embedding seam (design D-7 / D-2 / D-8 / D-12).

Rumen owns embedding generation. All
vectors are 384-dim float32, matching ``all-MiniLM-L6-v2`` — stored opaquely as a BLOB
and compared **in-process** with numpy cosine (no pgvector/sqlite-vec for the MVP).

Providers:

- ``HashingEmbeddingProvider`` — the zero-dependency default (numpy only): a hashing
  bag-of-words embedder. Deterministic and offline, so the engine runs and tests pass
  without the heavy model; texts sharing vocabulary land near each other.
- ``SentenceTransformerProvider`` — the real semantic model (``all-MiniLM-L6-v2``),
  behind the optional ``embeddings`` extra, lazily imported. Production selects this.

Both emit ``(n, 384)`` float32 arrays; consumers are provider-agnostic.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

EMBEDDING_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), EMBEDDING_DIM)`` float32 array of unit vectors."""
        ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class HashingEmbeddingProvider:
    """Deterministic hashing bag-of-words embedder (the hashing trick) — numpy only."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class SentenceTransformerProvider:
    """Real ``all-MiniLM-L6-v2`` embeddings (optional ``embeddings`` extra, lazy load)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy, optional dep

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        model = self._ensure_model()
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


_default_provider: EmbeddingProvider | None = None


def _auto_provider() -> EmbeddingProvider:
    """Prefer the real MiniLM model when the ``embeddings`` extra is installed; otherwise
    fall back to the zero-dependency hashing embedder (OSS on-ramp / no-extra installs)."""
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        return SentenceTransformerProvider()
    return HashingEmbeddingProvider()


def get_default_provider() -> EmbeddingProvider:
    """Return the process-wide default embedding provider.

    Auto-selects ``SentenceTransformerProvider`` (all-MiniLM-L6-v2, the production
    embedding) when the ``embeddings`` extra is installed, else the deterministic
    ``HashingEmbeddingProvider``. Override with ``set_default_provider`` (tests pin the
    hashing provider for determinism)."""
    global _default_provider
    if _default_provider is None:
        _default_provider = _auto_provider()
    return _default_provider


def set_default_provider(provider: EmbeddingProvider | None) -> None:
    global _default_provider
    _default_provider = provider


def encode_embedding(vector: np.ndarray) -> bytes:
    """Serialize a vector to the opaque BLOB stored on ``feed_items.embedding``."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def decode_embedding(blob: bytes | None) -> np.ndarray | None:
    """Deserialize a stored BLOB back into a float32 vector (``None`` passes through)."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (0.0 if either is a zero vector)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
