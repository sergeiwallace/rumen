"""T-05 seam: embedding provider + BLOB round-trip + cosine."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from rumen.embeddings import (
    EMBEDDING_DIM,
    HashingEmbeddingProvider,
    SentenceTransformerProvider,
    cosine_similarity,
    decode_embedding,
    encode_embedding,
)


def test_hashing_provider_is_deterministic_and_384_dim() -> None:
    provider = HashingEmbeddingProvider()
    a = provider.embed(["retrieval augmented generation"])
    b = provider.embed(["retrieval augmented generation"])
    assert a.shape == (1, EMBEDDING_DIM)
    assert np.array_equal(a, b)  # deterministic across calls


def test_shared_vocabulary_is_more_similar_than_disjoint() -> None:
    provider = HashingEmbeddingProvider()
    vecs = provider.embed(
        [
            "neural network transformer benchmark",
            "neural network transformer results",
            "casino poker betting gambling",
        ]
    )
    near = cosine_similarity(vecs[0], vecs[1])
    far = cosine_similarity(vecs[0], vecs[2])
    assert near > far


def test_empty_input_returns_empty_matrix() -> None:
    assert HashingEmbeddingProvider().embed([]).shape == (0, EMBEDDING_DIM)


def test_blob_roundtrip_preserves_vector() -> None:
    provider = HashingEmbeddingProvider()
    vec = provider.embed(["some text"])[0]
    blob = encode_embedding(vec)
    assert isinstance(blob, bytes)
    restored = decode_embedding(blob)
    assert np.allclose(restored, vec)


def test_decode_none_passes_through() -> None:
    assert decode_embedding(None) is None


def test_cosine_of_zero_vector_is_zero() -> None:
    assert cosine_similarity(np.zeros(EMBEDDING_DIM), np.ones(EMBEDDING_DIM)) == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or not os.environ.get("RUMEN_TEST_MINILM"),
    reason="MiniLM model load is heavy — set RUMEN_TEST_MINILM=1 (with the embeddings extra) to run",
)
@pytest.mark.unconstrained_memory
def test_sentence_transformer_provider_emits_384_dim_normalized() -> None:
    vecs = SentenceTransformerProvider().embed(["retrieval augmented generation", "different"])
    assert vecs.shape == (2, EMBEDDING_DIM)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)
