"""T-11: interest-profile EMA centroid — seed + rating updates (design D-2)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy import select

from rumen.embeddings import (
    EMBEDDING_DIM,
    cosine_similarity,
    decode_embedding,
    encode_embedding,
)
from rumen.models import FeedItem, InterestProfileRow
from rumen.ranking.interest_profile import InterestProfile, load_or_seed_profile


def _unit(i: int) -> np.ndarray:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _bookmark(source_id: str, vector: np.ndarray | None) -> FeedItem:
    return FeedItem(
        source="bookmarks",
        source_id=source_id,
        url=f"https://example.test/{source_id}",
        title=f"Bookmark {source_id}",
        published_at=datetime(2026, 7, 20, tzinfo=UTC),
        embedding=encode_embedding(vector) if vector is not None else None,
    )


def test_seed_from_is_normalized_mean() -> None:
    profile = InterestProfile.seed_from([_unit(0), _unit(1)])
    assert np.isclose(np.linalg.norm(profile.centroid), 1.0)
    # mean of e0,e1 points equally at both.
    assert np.isclose(
        cosine_similarity(profile.centroid, _unit(0)), cosine_similarity(profile.centroid, _unit(1))
    )


def test_positive_rating_shifts_centroid_toward_item() -> None:
    profile = InterestProfile(_unit(0))
    target = _unit(1)
    before = cosine_similarity(profile.centroid, target)
    profile.update(target, rating=1)
    after = cosine_similarity(profile.centroid, target)
    assert after > before


def test_negative_rating_shifts_centroid_away_from_item() -> None:
    profile = InterestProfile.seed_from([_unit(0), _unit(1)])
    target = _unit(1)
    before = cosine_similarity(profile.centroid, target)
    profile.update(target, rating=-1)
    after = cosine_similarity(profile.centroid, target)
    assert after < before


def test_zero_rating_is_noop() -> None:
    profile = InterestProfile(_unit(0))
    before = profile.centroid.copy()
    profile.update(_unit(1), rating=0)
    assert np.array_equal(profile.centroid, before)


def test_given_no_persisted_profile_when_loaded_then_it_seeds_from_the_bookmark_corpus_and_persists(
    session,
) -> None:
    session.add_all([_bookmark("one", _unit(0)), _bookmark("two", _unit(1))])
    session.commit()

    profile = load_or_seed_profile(session)
    session.commit()
    stored = session.scalar(select(InterestProfileRow))

    assert stored is not None
    assert np.allclose(profile.centroid, InterestProfile.seed_from([_unit(0), _unit(1)]).centroid)
    assert np.allclose(decode_embedding(stored.centroid), profile.centroid)
    assert stored.alpha == profile.alpha


def test_given_a_persisted_profile_when_loaded_then_it_returns_the_stored_centroid_and_alpha(
    session,
) -> None:
    stored_profile = InterestProfile(_unit(2), alpha=0.25)
    row = InterestProfileRow(
        centroid=encode_embedding(stored_profile.centroid), alpha=stored_profile.alpha
    )
    session.add(row)
    session.commit()
    original_updated_at = row.updated_at

    profile = load_or_seed_profile(session)
    session.commit()
    session.refresh(row)

    assert np.allclose(profile.centroid, stored_profile.centroid)
    assert profile.alpha == stored_profile.alpha
    assert row.updated_at == original_updated_at


def test_given_no_bookmark_embeddings_when_seeded_then_it_persists_a_zero_vector_profile(
    session,
) -> None:
    session.add(_bookmark("without-vector", None))
    session.commit()

    profile = load_or_seed_profile(session)
    session.commit()
    stored = session.scalar(select(InterestProfileRow))

    assert stored is not None
    assert np.array_equal(profile.centroid, np.zeros(EMBEDDING_DIM, dtype=np.float32))
    assert np.array_equal(decode_embedding(stored.centroid), profile.centroid)


def test_given_more_than_one_interest_profiles_row_when_loaded_then_it_raises(session) -> None:
    session.add_all(
        [
            InterestProfileRow(centroid=encode_embedding(_unit(0)), alpha=0.1),
            InterestProfileRow(centroid=encode_embedding(_unit(1)), alpha=0.1),
        ]
    )
    session.commit()

    with pytest.raises(RuntimeError, match="exactly one"):
        load_or_seed_profile(session)
