"""Embedding-second pass — cluster the no-ID / paraphrase residue (design Stage 4 / D-8).

Runs only after identity grouping. Compares entity **centroids** in-process (numpy cosine,
no pgvector/sqlite-vec). Two thresholds: ``tau_high`` auto-merges; the ``tau_low..tau_high``
band routes to the non-blocking ``merge_candidates`` review queue (both items stay live).
**Identity vetoes cosine** — two entities with different stable IDs never auto-merge.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.embeddings import decode_embedding
from rumen.grouping.merge import attach_item
from rumen.grouping.references import repoint_entity_references
from rumen.models import (
    FeedItem,
    MergeCandidate,
    TopicEntity,
    topic_entity_not_fully_source_deleted,
)

DEFAULT_TAU_HIGH = 0.92
DEFAULT_TAU_LOW = 0.80


def _identity_veto(a: TopicEntity, b: TopicEntity) -> bool:
    """True if the two entities carry *different* stable IDs (a merge is forbidden)."""
    return bool(a.stable_id_value and b.stable_id_value and a.stable_id_value != b.stable_id_value)


def _candidate_pairs(
    embedded: list[tuple[TopicEntity, np.ndarray]], tau_low: float
) -> list[tuple[float, int, int]]:
    """Return ``(sim, a_id, b_id)`` for every entity pair with cosine ≥ ``tau_low``, highest
    first, identity-vetoed pairs removed.

    Computed with one vectorized cosine over the centroid matrix rather than a Python
    double-loop over all O(n²) pairs — the loop materialized and sorted millions of
    near-zero sims to act on a handful; this materializes only the above-threshold pairs.
    The cosine values are identical (dot / product-of-norms, float32), and a zero-norm
    centroid yields sim 0 (< any positive ``tau_low``), matching ``cosine_similarity``.
    """
    n = len(embedded)
    if n < 2:
        return []
    matrix = np.vstack([v for _, v in embedded]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    normed = matrix / safe[:, None]
    sims = normed @ normed.T  # cosine for every pair
    iu, ju = np.triu_indices(n, k=1)
    selected = sims[iu, ju] >= tau_low
    pairs: list[tuple[float, int, int]] = []
    for i, j in zip(iu[selected].tolist(), ju[selected].tolist()):
        a, b = embedded[i][0], embedded[j][0]
        if _identity_veto(a, b):
            continue
        pairs.append((float(sims[i, j]), a.id, b.id))
    pairs.sort(reverse=True)
    return pairs


def _is_non_duplicate(session: Session, a_id: int, b_id: int) -> bool:
    lo, hi = sorted((a_id, b_id))
    status = session.scalars(
        select(MergeCandidate.status).where(
            MergeCandidate.entity_a_id == lo, MergeCandidate.entity_b_id == hi
        )
    ).first()
    return status == "rejected"


def cluster_residue(
    session: Session,
    *,
    tau_high: float = DEFAULT_TAU_HIGH,
    tau_low: float = DEFAULT_TAU_LOW,
    commit_batch_size: int = 500,
    residue_window_days: int = 30,
    now: datetime | None = None,
) -> dict[str, int]:
    """Auto-merge ≥ tau_high; queue the tau_low..tau_high band. Returns pass counts.

    Commits every ``commit_batch_size`` decisions (merges + queued candidates) so a large
    residue pass is incrementally durable (design D-14); every write is a committed
    ``merge_log``/``merge_candidates`` row, so an interrupted pass keeps its progress.
    """
    if residue_window_days < 1:
        raise ValueError("residue_window_days must be positive")
    cutoff = (now or datetime.now(UTC)) - timedelta(days=residue_window_days)
    entities = list(
        session.scalars(
            select(TopicEntity).where(
                TopicEntity.stable_id_value.is_(None),
                TopicEntity.earliest_seen >= cutoff,
                topic_entity_not_fully_source_deleted(),
            )
        )
    )
    merged_away: set[int] = set()
    counts = {"embedding_merges": 0, "candidates": 0}

    # Candidate pairs (cosine ≥ tau_low), highest-similarity first so the strongest merges
    # win — built with one vectorized cosine, not an O(n²) Python loop.
    embedded = [(e, decode_embedding(e.embedding)) for e in entities]
    embedded = [(e, v) for e, v in embedded if v is not None]
    pairs = _candidate_pairs(embedded, tau_low)

    since_commit = 0
    for sim, a_id, b_id in pairs:
        if a_id in merged_away or b_id in merged_away:
            continue
        if _is_non_duplicate(session, a_id, b_id):
            continue
        if sim >= tau_high:
            keep = session.get(TopicEntity, a_id)
            drop = session.get(TopicEntity, b_id)
            for item in list(
                session.scalars(select(FeedItem).where(FeedItem.topic_entity_id == b_id))
            ):
                # attach_item() already calls recompute_entity() (project_facets=True) for
                # every item it moves, re-querying keep's full constituent set fresh from the
                # DB each time — the last call in this loop already reflects every absorbed
                # item. A further recompute_entity(keep) call here would re-run
                # _project_facet_survivorship() a second time before that last call's insert is
                # flushed and double-write it (same failure shape as the group_pending() bug
                # this pattern mirrors) — do not add one back.
                attach_item(
                    session,
                    keep,
                    item,
                    method="embedding",
                    confidence=sim,
                    evidence={"cosine": sim},
                )
            repoint_entity_references(session, keep, drop)
            merged_away.add(b_id)
            session.delete(drop)
            session.flush()
            counts["embedding_merges"] += 1
            since_commit += 1
        elif sim >= tau_low:
            lo, hi = sorted((a_id, b_id))
            candidate_exists = session.scalars(
                select(MergeCandidate).where(
                    MergeCandidate.entity_a_id == lo, MergeCandidate.entity_b_id == hi
                )
            ).first()
            if candidate_exists is None:
                session.add(
                    MergeCandidate(
                        entity_a_id=lo,
                        entity_b_id=hi,
                        similarity=sim,
                        reason="embedding_band",
                        status="pending",
                    )
                )
                counts["candidates"] += 1
                since_commit += 1
        if since_commit >= commit_batch_size:
            session.commit()
            since_commit = 0
    session.commit()
    return counts
