"""Relevance ranking — interest cosine + MMR + time-decay + exploration (design D-2 / T-11).

All programmatic (no LLM, no query). Values are config-driven day-one heuristics from the
parameter research (``docs/research/curation-ranking-parameter-defaults.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.db.engine import batched_write_mode
from rumen.embeddings import cosine_similarity, decode_embedding
from rumen.models import TopicEntity, topic_entity_not_fully_source_deleted
from rumen.ranking.interest_profile import InterestProfile

DEFAULT_LAMBDA = 0.5  # MMR relevance-vs-diversity (design D-2)
DEFAULT_EXPLORATION = 0.12  # 10–15% of slots reserved (design D-2)

# Per-source-class time-decay half-lives in days (design D-2).
FAST_HALF_LIFE = 3.0  # releases / launches
MEDIUM_HALF_LIFE = 14.0  # papers / news
SLOW_HALF_LIFE = 60.0  # essays / evergreen

_CATEGORY_HALF_LIFE = {
    "tool": FAST_HALF_LIFE,
    "launch": FAST_HALF_LIFE,
    "paper": MEDIUM_HALF_LIFE,
    "research": MEDIUM_HALF_LIFE,
    "ai-news": MEDIUM_HALF_LIFE,
    "news": MEDIUM_HALF_LIFE,
    "technique": MEDIUM_HALF_LIFE,
    "essay": SLOW_HALF_LIFE,
    "opinion": SLOW_HALF_LIFE,
}


def half_life_for(category: str | None) -> float:
    return _CATEGORY_HALF_LIFE.get(category or "", MEDIUM_HALF_LIFE)


@dataclass(slots=True)
class RankInput:
    entity_id: int
    embedding: np.ndarray | None
    timestamp: datetime
    half_life_days: float


@dataclass(slots=True)
class RankedEntity:
    entity_id: int
    relevance: float
    score: float
    rank: int
    is_exploration: bool


def _time_decay(age_days: float, half_life_days: float) -> float:
    return float(0.5 ** (age_days / half_life_days))


def rank(
    inputs: list[RankInput],
    profile: InterestProfile,
    *,
    now: datetime,
    lambda_mmr: float = DEFAULT_LAMBDA,
    exploration_fraction: float = DEFAULT_EXPLORATION,
    page_size: int | None = None,
) -> list[RankedEntity]:
    """Rank entities: time-decayed interest cosine → MMR diversity → exploration quota.

    Entities with no embedding are deterministically placed at the exploration tail
    (never dropped, never crash the ranker).
    """
    scored: list[tuple[RankInput, float, float]] = []  # (input, relevance, effective_score)
    no_embedding: list[RankInput] = []
    for item in inputs:
        if item.embedding is None:
            no_embedding.append(item)
            continue
        rel = cosine_similarity(item.embedding, profile.centroid)
        age_days = (now - item.timestamp).total_seconds() / 86400.0
        eff = rel * _time_decay(age_days, item.half_life_days)
        scored.append((item, rel, eff))

    # MMR greedy re-selection over the time-decayed relevance. Vectorized: a running
    # max-similarity-to-selected vector is updated with one matvec per pick, so the whole
    # selection is O(n²·d) numpy instead of the O(n³) Python triple-loop that made ranking
    # hang at corpus scale (~2400 entities). Semantics (greedy argmax, first-index tie-break)
    # are identical to the naive form; the existing MMR tests pin that.
    selected: list[tuple[RankInput, float, float]] = []
    if scored:
        embeddings = np.array([s[0].embedding for s in scored], dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.where(norms == 0.0, 1.0, norms)
        effs = np.array([eff for _, _, eff in scored], dtype=np.float64)
        max_sim = np.zeros(len(scored), dtype=np.float64)
        chosen = np.zeros(len(scored), dtype=bool)
        for _ in range(len(scored)):
            mmr = lambda_mmr * effs - (1.0 - lambda_mmr) * max_sim
            mmr[chosen] = -np.inf
            j = int(np.argmax(mmr))
            chosen[j] = True
            selected.append(scored[j])
            max_sim = np.maximum(max_sim, normed @ normed[j])

    # No-embedding entities sink to the tail, deterministically ordered by id.
    ordered = selected + [(i, 0.0, 0.0) for i in sorted(no_embedding, key=lambda x: x.entity_id)]
    size = page_size if page_size is not None else len(ordered)
    page = ordered[:size]

    # Reserve the last k slots of the page as exploration (lower-relevance / novel).
    k = min(len(page), max(1, int(exploration_fraction * size))) if page else 0
    explore_start = max(0, len(page) - k)

    results = []
    for rank_idx, (item, rel, eff) in enumerate(page):
        results.append(
            RankedEntity(
                entity_id=item.entity_id,
                relevance=rel,
                score=eff,
                rank=rank_idx,
                is_exploration=rank_idx >= explore_start,
            )
        )
    return results


def rank_topic_entities(
    session: Session,
    profile: InterestProfile,
    *,
    now: datetime,
    lambda_mmr: float = DEFAULT_LAMBDA,
    exploration_fraction: float = DEFAULT_EXPLORATION,
    half_life_days: dict[str, float] | None = None,
    default_half_life_days: float = MEDIUM_HALF_LIFE,
    page_size: int | None = None,
    commit_batch_size: int = 500,
) -> list[RankedEntity]:
    """Rank all ``topic_entities`` and persist their effective score and final order.

    ``half_life_days`` (per-category, config-driven) layers over the built-in defaults;
    ``default_half_life_days`` is the fallback for unlisted categories. Score-writes commit
    every ``commit_batch_size`` rows (design D-14). Ranking is a pure recompute over the
    current entities, so an interrupted pass simply re-runs — the batched commits just bound
    how much a re-run has to redo.
    """
    half_lives = {**_CATEGORY_HALF_LIFE, **(half_life_days or {})}
    entities = list(
        session.scalars(select(TopicEntity).where(topic_entity_not_fully_source_deleted()))
    )
    inputs = [
        RankInput(
            entity_id=e.id,
            embedding=decode_embedding(e.embedding),
            timestamp=e.earliest_seen,
            half_life_days=half_lives.get(e.category or "", default_half_life_days),
        )
        for e in entities
    ]
    ranked = rank(
        inputs,
        profile,
        now=now,
        lambda_mmr=lambda_mmr,
        exploration_fraction=exploration_fraction,
        page_size=page_size,
    )
    by_id = {e.id: e for e in entities}
    # expire_on_commit=False so batched score-writes don't expire the entity set into per-row
    # re-fetches over the network (see db.engine.batched_write_mode).
    with batched_write_mode(session):
        for i, r in enumerate(ranked, start=1):
            entity = by_id[r.entity_id]
            entity.relevance_score = r.score
            entity.rank_position = r.rank
            entity.is_exploration = int(r.is_exploration)
            if i % commit_batch_size == 0:
                session.commit()
        session.commit()
    return ranked
