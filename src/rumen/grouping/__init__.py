"""Cross-source reference grouping — identity-first, embedding-second (design D-8 / T-07).

Two ordered passes: (1) canonicalize each pointer's URL + extract a work-level stable ID
(alias-resolved) and merge deterministically on shared identity; (2) embed the no-ID
residue and cluster in-process (cosine), fenced by identity veto + a review band. Every
merge is a reversible ``merge_log`` row; the review queue is non-blocking.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.db.engine import batched_write_mode
from rumen.embeddings import EmbeddingProvider, encode_embedding, get_default_provider
from rumen.grouping.canonicalize import canonicalize_url
from rumen.grouping.embedding_residue import (
    DEFAULT_TAU_HIGH,
    DEFAULT_TAU_LOW,
    cluster_residue,
)
from rumen.grouping.merge import (
    _entity_items,
    _project_facet_survivorship,
    recompute_entity,
    unmerge_item,
)
from rumen.grouping.stable_id import (
    apply_alias,
    extract_stable_id,
    load_alias_map,
    stable_id_from_content,
)
from rumen.models import (
    FeedItem,
    ItemConcept,
    MergeLog,
    TopicEntity,
    topic_entity_not_fully_source_deleted,
)

__all__ = ["group_pending", "unmerge_item"]

# In-memory identity index: mirrors the two DB lookups the identity pass used to run
# per item (`stable_id` exact, then `canonical_url` among stable-id-less entities). Keyed
# ``("stable", (kind, value))`` or ``("url", canonical_url)``.
_IdentityKey = tuple[str, object]
_IdentityIndex = dict[_IdentityKey, TopicEntity]


def _identity_key(entity: TopicEntity) -> _IdentityKey | None:
    """The index key an entity is currently reachable by (mirrors match precedence)."""
    if entity.stable_id_value:
        return "stable", (entity.stable_id_kind, entity.stable_id_value)
    if entity.canonical_url:
        return "url", entity.canonical_url
    return None


def _match_key(item: FeedItem) -> _IdentityKey | None:
    if item.stable_id_value:
        return "stable", (item.stable_id_kind, item.stable_id_value)
    return None


def group_pending(
    session: Session,
    *,
    provider: EmbeddingProvider | None = None,
    embed: bool = True,
    tau_high: float = DEFAULT_TAU_HIGH,
    tau_low: float = DEFAULT_TAU_LOW,
    resolve_redirect: Callable[[str], str] | None = None,
    residue_window_days: int = 30,
    now: datetime | None = None,
    commit_batch_size: int = 500,
) -> dict[str, int]:
    """Group ungrouped ``feed_items`` into ``topic_entities``. Returns pass counts.

    Batched for remote-Postgres scale: the alias table and the existing identity index are
    loaded once, all embeddings are computed in a single provider call, and identity
    matching runs against in-memory dicts — so the whole identity pass costs a handful of
    round-trips instead of one-or-more per pointer.

    Incrementally durable (design D-14): the identity pass and the residue pass each commit
    every ``commit_batch_size`` entities, so an interrupted run keeps completed batches and a
    re-run resumes on the still-ungrouped items (``topic_entity_id IS NULL``). ``expire_on_commit``
    is disabled for the duration so the in-memory identity index survives those commits.
    """
    provider = provider or get_default_provider()
    items = list(
        session.scalars(
            select(FeedItem)
            .where(FeedItem.status == "active", FeedItem.topic_entity_id.is_(None))
            .order_by(FeedItem.published_at.asc(), FeedItem.id.asc())
        )
    )
    if not items:
        return {"entities_created": 0, "identity_merges": 0}

    # Keep the identity index/entities live across commits + control flushes so writes batch
    # at the commit boundaries we choose (see db.engine.batched_write_mode).
    with batched_write_mode(session):
        alias_map = load_alias_map(session)

        # Enrich identity fields (no per-item query — alias resolves against the loaded map).
        for item in items:
            item.canonical_url = canonicalize_url(item.url, resolve_redirect=resolve_redirect)
            kind, value = extract_stable_id(item.url)
            if not value:
                # A social/aggregator pointer carries its identity in a linked URL.
                kind, value = stable_id_from_content(item.content, item.link_metadata)
            kind, value = apply_alias(alias_map, kind, value)
            item.stable_id_kind = kind
            item.stable_id_value = value

        # Embed text in one batch. Identity-resolved pointers retain constituent vectors so
        # recompute_entity can maintain an accurate centroid for a mixed-source entity.
        if embed:
            texts, positions = [], []
            for pos, item in enumerate(items):
                text = f"{item.title or ''}\n{item.content or ''}".strip()
                if text:
                    texts.append(text)
                    positions.append(pos)
            if texts:
                vectors = provider.embed(texts)
                for pos, vector in zip(positions, vectors):
                    items[pos].embedding = encode_embedding(vector)
        session.commit()  # checkpoint: enriched identity fields + embeddings persisted

        # Preload the existing identity index once, then keep it in sync in-process.
        index: _IdentityIndex = {}
        for entity in session.scalars(
            select(TopicEntity).where(topic_entity_not_fully_source_deleted())
        ):
            key = _identity_key(entity)  # url key only surfaces for stable-id-less entities
            if key is not None:
                index[key] = entity
        human_topic_entity_ids = set(
            session.scalars(
                select(ItemConcept.item_id).where(
                    ItemConcept.item_table == "topic_entities",
                    ItemConcept.facet == "topic",
                    ItemConcept.method.startswith("human:"),
                )
            )
        )
        # Constituents keyed by the entity *object* — new entities have no id until flushed.
        constituents_of: dict[TopicEntity, list[FeedItem]] = {}
        # Deferred writes: (entity, item, method, confidence) links realized at batch flush.
        pending_links: list[tuple[TopicEntity, FeedItem, str, float]] = []
        counts = {"entities_created": 0, "identity_merges": 0}

        def _flush_identity_batch() -> None:
            # One flush assigns ids to all new entities (batched INSERT via insertmanyvalues);
            # then item links + merge_log rows for the whole batch flush together — a handful
            # of round-trips per batch instead of a few per pointer.
            session.flush()
            # recompute_entity() ran with project_facets=False in the loop below — an entity
            # touched by 2+ items in this batch would otherwise call _project_facet_survivorship
            # more than once before anything is flushed, and its DELETE-then-INSERT pattern only
            # sees already-*flushed* rows (autoflush is off), so a second unflushed call
            # double-inserts. Project each batch-touched entity exactly once here, using its
            # complete constituent list, now that every entity (new or matched) has a real id.
            for entity in {ent for ent, _itm, _meth, _conf in pending_links}:
                _project_facet_survivorship(session, entity, constituents_of[entity])
            for ent, itm, meth, conf in pending_links:
                itm.topic_entity_id = ent.id
                itm.match_method = meth
                itm.match_confidence = conf
                evidence = (
                    {"stable_id_kind": itm.stable_id_kind, "stable_id_value": itm.stable_id_value}
                    if meth == "stable_id"
                    else {"canonical_url": itm.canonical_url}
                    if meth == "canonical_url"
                    else None
                )
                session.add(
                    MergeLog(
                        topic_entity_id=ent.id,
                        feed_item_id=itm.id,
                        match_method=meth,
                        match_confidence=conf,
                        actor="auto",
                        action="merge",
                        evidence=evidence,
                    )
                )
            session.commit()
            pending_links.clear()

        for i, item in enumerate(items, start=1):
            # Match precedence mirrors the old query pair: exact stable_id first, then
            # canonical_url among stable-id-less entities (a stable-id item with no stable
            # match still falls through to the url index).
            match = index.get(_match_key(item)) if item.stable_id_value else None
            method, confidence = "stable_id", 1.0
            if match is None and item.canonical_url:
                match = index.get(("url", item.canonical_url))
                method, confidence = "canonical_url", 0.9

            if match is not None:
                entity = match
                constituents = constituents_of.get(entity)
                if constituents is None:  # entity carried over from a previous run
                    constituents = _entity_items(session, entity.id)
                    constituents_of[entity] = constituents
                before = _identity_key(entity)
                constituents.append(item)
                counts["identity_merges"] += 1
            else:
                entity = TopicEntity(
                    canonical_title=item.title,
                    canonical_url=item.canonical_url or item.url,
                    earliest_seen=item.published_at,
                    source_count=0,
                )
                session.add(entity)
                constituents = [item]
                constituents_of[entity] = constituents
                before = None
                method, confidence = _seed_method(item), _seed_conf(item)
                counts["entities_created"] += 1

            pending_links.append((entity, item, method, confidence))
            # Survivorship recompute is in-memory (constituents passed) — no query, no flush;
            # it sets the entity's identity fields so the re-index below sees them. Facet
            # projection is deferred to the batch flush (see _flush_identity_batch).
            recompute_entity(
                session,
                entity,
                items=constituents,
                project_facets=False,
                human_topic_authoritative=entity.id in human_topic_entity_ids,
            )

            # Re-index if survivorship shifted the entity's identity (e.g. a stable-id
            # pointer joined a url-seeded entity), so later pointers still resolve to it.
            after = _identity_key(entity)
            if after != before:
                if before is not None and index.get(before) is entity:
                    del index[before]
                if after is not None:
                    index[after] = entity

            if i % commit_batch_size == 0:
                _flush_identity_batch()
        _flush_identity_batch()  # flush the final identity batch before the residue pass

        # Embedding-second pass over the residue.
        if embed:
            counts.update(
                cluster_residue(
                    session,
                    tau_high=tau_high,
                    tau_low=tau_low,
                    commit_batch_size=commit_batch_size,
                    residue_window_days=residue_window_days,
                    now=now,
                )
            )
        session.commit()
        return counts


def _seed_method(item: FeedItem) -> str:
    if item.stable_id_value:
        return "stable_id"
    if item.canonical_url:
        return "canonical_url"
    return "seed"


def _seed_conf(item: FeedItem) -> float:
    if item.stable_id_value:
        return 1.0
    if item.canonical_url:
        return 0.9
    return 1.0
