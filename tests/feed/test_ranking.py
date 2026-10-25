"""T-11: relevance ranking — cosine + MMR + time-decay + exploration (design D-2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from rumen.embeddings import EMBEDDING_DIM, encode_embedding
from rumen.models import TopicEntity
from rumen.ranking.interest_profile import InterestProfile
from rumen.ranking.rank import (
    FAST_HALF_LIFE,
    MEDIUM_HALF_LIFE,
    SLOW_HALF_LIFE,
    RankInput,
    rank,
    rank_topic_entities,
)

_NOW = datetime(2026, 7, 16, tzinfo=UTC)


def _unit(i: int) -> np.ndarray:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _rank_by_id(results):
    return {r.entity_id: r for r in results}


def test_near_centroid_ranks_above_far() -> None:
    profile = InterestProfile(_unit(0))
    inputs = [
        RankInput(1, _unit(0), _NOW, MEDIUM_HALF_LIFE),  # aligned
        RankInput(2, _unit(5), _NOW, MEDIUM_HALF_LIFE),  # orthogonal
    ]
    results = rank(inputs, profile, now=_NOW, exploration_fraction=0.0)
    ranks = _rank_by_id(results)
    assert ranks[1].rank < ranks[2].rank


def test_mmr_demotes_near_duplicate_below_a_diverse_item() -> None:
    # profile is the diagonal; A and B both point at e0 (near-dups), C at e1.
    profile = InterestProfile.seed_from([_unit(0), _unit(1)])
    inputs = [
        RankInput(1, _unit(0), _NOW, MEDIUM_HALF_LIFE),  # A
        RankInput(2, _unit(0), _NOW, MEDIUM_HALF_LIFE),  # B (near-dup of A)
        RankInput(3, _unit(1), _NOW, MEDIUM_HALF_LIFE),  # C (diverse, equal relevance)
    ]
    results = rank(inputs, profile, now=_NOW, lambda_mmr=0.5, exploration_fraction=0.0)
    ranks = _rank_by_id(results)
    # The diverse C is promoted above the near-duplicate B.
    assert ranks[3].rank < ranks[2].rank


def test_time_decay_favours_the_more_recent_of_equal_relevance() -> None:
    profile = InterestProfile(_unit(0))
    inputs = [
        RankInput(1, _unit(0), _NOW, MEDIUM_HALF_LIFE),  # recent
        RankInput(2, _unit(0), _NOW - timedelta(days=30), MEDIUM_HALF_LIFE),  # old
    ]
    results = rank(inputs, profile, now=_NOW, exploration_fraction=0.0)
    ranks = _rank_by_id(results)
    assert ranks[1].rank < ranks[2].rank
    assert ranks[1].score > ranks[2].score


def test_time_decay_is_per_source_class() -> None:
    # Equal relevance + equal age, but a fast-decay release vs a slow-decay essay.
    profile = InterestProfile(_unit(0))
    age = _NOW - timedelta(days=14)
    inputs = [
        RankInput(1, _unit(0), age, FAST_HALF_LIFE),  # release — decays fast
        RankInput(2, _unit(0), age, SLOW_HALF_LIFE),  # essay — decays slow
    ]
    results = rank(inputs, profile, now=_NOW, exploration_fraction=0.0)
    ranks = _rank_by_id(results)
    assert ranks[2].score > ranks[1].score  # the evergreen essay outranks the stale release
    assert ranks[2].rank < ranks[1].rank


def test_exploration_quota_reserves_at_least_one_slot() -> None:
    profile = InterestProfile(_unit(0))
    inputs = [RankInput(i, _unit(0), _NOW, MEDIUM_HALF_LIFE) for i in range(1, 6)]
    results = rank(inputs, profile, now=_NOW, exploration_fraction=0.2, page_size=5)
    assert any(r.is_exploration for r in results)


def test_mmr_selection_returns_full_permutation_at_scale() -> None:
    # Regression guard for the vectorized MMR: at corpus scale the greedy selection must
    # still return every embedded entity exactly once (the O(n^3) Python loop it replaced
    # made this hang; a broken vectorization would drop or duplicate items).
    profile = InterestProfile(_unit(0))
    n = 250
    inputs = [
        RankInput(i, _unit(i % EMBEDDING_DIM), _NOW, MEDIUM_HALF_LIFE) for i in range(1, n + 1)
    ]
    results = rank(inputs, profile, now=_NOW, exploration_fraction=0.0)
    ids = [r.entity_id for r in results]
    assert sorted(ids) == list(range(1, n + 1))  # every entity present, none duplicated
    assert [r.rank for r in results] == list(range(n))  # ranks are a dense 0..n-1 permutation


def test_no_embedding_entity_sinks_to_tail_not_dropped() -> None:
    profile = InterestProfile(_unit(0))
    inputs = [
        RankInput(1, _unit(0), _NOW, MEDIUM_HALF_LIFE),
        RankInput(2, None, _NOW, MEDIUM_HALF_LIFE),  # no embedding
    ]
    results = rank(inputs, profile, now=_NOW, exploration_fraction=0.0)
    ids = [r.entity_id for r in results]
    assert 2 in ids  # not dropped
    assert ids[-1] == 2  # deterministically at the tail


def test_rank_topic_entities_persists_ranked_entity_values(session) -> None:
    entities = [
        TopicEntity(
            canonical_title=f"Entity {i}",
            canonical_url=f"https://example.com/{i}",
            earliest_seen=_NOW,
            source_count=1,
            embedding=encode_embedding(vector),
        )
        for i, vector in enumerate([_unit(0), _unit(0), _unit(1)], start=1)
    ]
    session.add_all(entities)
    session.commit()

    ranked = rank_topic_entities(
        session,
        InterestProfile.seed_from([_unit(0), _unit(1)]),
        now=_NOW,
        exploration_fraction=0.34,
    )
    session.expire_all()

    for result in ranked:
        stored = session.get(TopicEntity, result.entity_id)
        assert stored is not None
        assert stored.rank_position == result.rank
        assert stored.is_exploration == int(result.is_exploration)
        assert stored.relevance_score == pytest.approx(result.score)
