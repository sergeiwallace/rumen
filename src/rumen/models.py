"""Curation schema — SQLAlchemy models (design Data Model § / T-02).

All 12 tables of the self-contained backend. Types compile to Postgres on the
primary (``JSONB`` for list/dict columns, ``TIMESTAMPTZ`` for instants, ``BYTEA``
for embeddings) and to SQLite on the fallback, from one definition:

- ``_json()``   → ``JSONB`` on Postgres, generic ``JSON`` (text) on SQLite.
- ``TS``        → ``TIMESTAMPTZ`` on Postgres, ``DATETIME`` on SQLite.
- embeddings    → ``LargeBinary`` = ``BYTEA`` / ``BLOB`` (opaque; in-process cosine, D-7).

The persistence layer owns these models and their migrations.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import JSON as _GenericJSON
from sqlalchemy import (
    ColumnElement,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Text,
    TypeDecorator,
    UniqueConstraint,
    exists,
    or_,
    select,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _json():
    """JSONB on Postgres, generic JSON (stored as text) on SQLite."""
    return _GenericJSON().with_variant(JSONB, "postgresql")


class TZDateTime(TypeDecorator):
    """Timezone-aware UTC datetime, portable across Postgres and SQLite.

    SQLite has no timezone type, so ``DateTime(timezone=True)`` round-trips a
    tz-aware value to a **naive** one — which then can't be compared against an aware
    ``now`` (ranking, digest periods). This coerces naive values to UTC on both bind
    and result, so reads are always tz-aware regardless of dialect. DDL is unchanged
    (``TIMESTAMPTZ`` on Postgres, ``DATETIME`` on SQLite), so no migration change is
    needed. Canonical SQLAlchemy pattern for cross-dialect tz handling.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, _dialect):
        if value is not None:
            # Normalize every stored value to UTC: naive → assume UTC; aware → convert.
            value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return value

    def process_result_value(self, value, _dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


# TIMESTAMPTZ on Postgres; DATETIME on SQLite — always returns tz-aware UTC.
TS = TZDateTime()


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TopicEntity(Base):
    """Deduplicated topic entity — ONE canonical item, many pointers (D-8)."""

    __tablename__ = "topic_entities"
    __table_args__ = (
        Index("idx_topic_entities_stable_id", "stable_id_kind", "stable_id_value"),
        Index("idx_topic_entities_category", "category"),
        Index("idx_topic_entities_rank_position", "rank_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_title: Mapped[str] = mapped_column(Text, nullable=False)
    display_title: Mapped[str | None] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    stable_id_kind: Mapped[str | None] = mapped_column(Text)
    stable_id_value: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list | None] = mapped_column(_json())
    earliest_seen: Mapped[datetime] = mapped_column(TS, nullable=False)
    source_count: Mapped[int] = mapped_column(Integer, default=1)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    relevance_score: Mapped[float | None] = mapped_column(Float)
    rank_position: Mapped[int | None] = mapped_column(Integer)
    is_exploration: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class FeedItem(Base):
    """Raw ingested pointer. The set sharing a ``topic_entity_id`` IS that entity's
    Sources/References list (D-8)."""

    __tablename__ = "feed_items"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_feed_items_source"),
        Index("idx_feed_items_topic", "topic_entity_id"),
        Index("idx_feed_items_stable_id", "stable_id_kind", "stable_id_value"),
        Index("idx_feed_items_canonical_url", "canonical_url"),
        Index("idx_feed_items_published", "published_at"),
        Index("idx_feed_items_category", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    stable_id_kind: Mapped[str | None] = mapped_column(Text)
    stable_id_value: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    matched_keywords: Mapped[list | None] = mapped_column(_json())
    category: Mapped[str | None] = mapped_column(Text)
    categories: Mapped[list | None] = mapped_column(_json())
    domains: Mapped[list | None] = mapped_column(_json())
    tags: Mapped[list | None] = mapped_column(_json())
    link_metadata: Mapped[dict | None] = mapped_column(_json())
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    topic_entity_id: Mapped[int | None] = mapped_column(ForeignKey("topic_entities.id"))
    match_method: Mapped[str | None] = mapped_column(Text)
    match_confidence: Mapped[float | None] = mapped_column(Float)
    classification_confidence: Mapped[float | None] = mapped_column(Float)
    classification_method: Mapped[str | None] = mapped_column(Text)
    facets_classified_at: Mapped[datetime | None] = mapped_column(TS)
    source_revision_digest: Mapped[str | None] = mapped_column(Text)
    source_last_seen_at: Mapped[datetime | None] = mapped_column(TS)
    source_deleted_at: Mapped[datetime | None] = mapped_column(TS)


def topic_entity_not_fully_source_deleted() -> ColumnElement[bool]:
    """True unless every ``FeedItem`` pointing at this ``TopicEntity`` is source-deleted.

    An entity with no linked ``FeedItem`` at all (the common case for entities built
    directly in tests, or any pipeline stage before pointers are attached) is untouched
    by source-deletion tracking and must stay included — an earlier version of this
    filter required a *live* linked item to exist, which wrongly dropped every
    zero-pointer entity from ranking, grouping, and digests (a regression guard).
    """
    has_any_pointer = exists(select(1).where(FeedItem.topic_entity_id == TopicEntity.id))
    has_live_pointer = exists(
        select(1).where(
            FeedItem.topic_entity_id == TopicEntity.id,
            FeedItem.source_deleted_at.is_(None),
        )
    )
    return or_(~has_any_pointer, has_live_pointer)


class SkosConcept(Base):
    """One materialized controlled-vocabulary concept (D-24)."""

    __tablename__ = "skos_concepts"
    __table_args__ = (Index("idx_skos_concepts_facet", "facet", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    facet: Mapped[str] = mapped_column(Text, nullable=False)
    pref_label: Mapped[str] = mapped_column(Text, nullable=False)
    notation: Mapped[str | None] = mapped_column(Text)
    scheme: Mapped[str] = mapped_column(Text, nullable=False)
    scheme_version: Mapped[str | None] = mapped_column(Text)
    # The opaque runtime id is derived from this URI; retain the donor URI too
    # so imports remain auditable without reverse-engineering its hash.
    source_uri: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Text, default="active", server_default=sql_text("'active'"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, server_default=sql_text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TS,
        default=_utcnow,
        onupdate=_utcnow,
        server_default=sql_text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


class SkosRelation(Base):
    """A SKOS relation, with review provenance for cross-scheme mappings."""

    __tablename__ = "skos_relations"
    __table_args__ = (
        UniqueConstraint(
            "source_concept_id", "target_concept_id", "relation_type", name="uq_skos_relations"
        ),
        Index("idx_skos_relations_source", "source_concept_id", "relation_type"),
        Index("idx_skos_relations_target", "target_concept_id", "relation_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_concept_id: Mapped[int] = mapped_column(ForeignKey("skos_concepts.id"), nullable=False)
    target_concept_id: Mapped[int] = mapped_column(ForeignKey("skos_concepts.id"), nullable=False)
    relation_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_scheme: Mapped[str | None] = mapped_column(Text)
    source_version: Mapped[str | None] = mapped_column(Text)
    target_scheme: Mapped[str | None] = mapped_column(Text)
    target_version: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(Text)
    review_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, server_default=sql_text("CURRENT_TIMESTAMP"), nullable=False
    )


class ItemConcept(Base):
    """One controlled-vocabulary facet assignment for a feed item or topic entity."""

    __tablename__ = "item_concepts"
    __table_args__ = (
        UniqueConstraint(
            "item_table", "item_id", "concept_id", name="uq_item_concepts_item_concept"
        ),
        Index("idx_item_concepts_item", "item_table", "item_id"),
        Index("idx_item_concepts_facet", "facet", "concept_id"),
        Index("idx_item_concepts_concept", "concept_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_table: Mapped[str] = mapped_column(Text, nullable=False)
    item_id: Mapped[int] = mapped_column(Integer, nullable=False)
    concept_id: Mapped[int] = mapped_column(ForeignKey("skos_concepts.id"), nullable=False)
    facet: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, server_default=sql_text("CURRENT_TIMESTAMP"), nullable=False
    )


class ClassificationCorrection(Base):
    """Append-only human-review evidence for one item's complete facet assignment."""

    __tablename__ = "classification_corrections"
    __table_args__ = (
        Index(
            "idx_classification_corrections_item_facet",
            "item_table",
            "item_id",
            "facet",
            "submitted_at",
        ),
        Index("idx_classification_corrections_telemetry", "status", "facet", "decided_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_table: Mapped[str] = mapped_column(Text, nullable=False)
    item_id: Mapped[int] = mapped_column(Integer, nullable=False)
    facet: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    prior_concept_ids: Mapped[list] = mapped_column(_json(), nullable=False)
    corrected_concept_ids: Mapped[list] = mapped_column(_json(), nullable=False)
    classifier_context_json: Mapped[dict | None] = mapped_column(_json())
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, default="proposed", server_default=sql_text("'proposed'"), nullable=False
    )
    decision_actor: Mapped[str | None] = mapped_column(Text)
    decision_rationale: Mapped[str | None] = mapped_column(Text)
    submitted_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, server_default=sql_text("CURRENT_TIMESTAMP"), nullable=False
    )
    decided_at: Mapped[datetime | None] = mapped_column(TS)
    supersedes: Mapped[int | None] = mapped_column(ForeignKey("classification_corrections.id"))


class TaxonomyRevision(Base):
    """One reversible maintenance-audit mutation (D-23)."""

    __tablename__ = "taxonomy_revisions"
    __table_args__ = (Index("idx_taxonomy_revisions_run", "audit_run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audit_run_id: Mapped[str] = mapped_column(Text, nullable=False)
    op_type: Mapped[str] = mapped_column(Text, nullable=False)
    concept_ids: Mapped[list] = mapped_column(_json(), nullable=False)
    before_json: Mapped[dict | None] = mapped_column(_json())
    after_json: Mapped[dict | None] = mapped_column(_json())
    actor: Mapped[str] = mapped_column(
        Text,
        default="system:taxonomy-audit",
        server_default=sql_text("'system:taxonomy-audit'"),
        nullable=False,
    )
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    reversible: Mapped[int] = mapped_column(
        Integer, default=1, server_default=sql_text("1"), nullable=False
    )
    reverts: Mapped[int | None] = mapped_column(ForeignKey("taxonomy_revisions.id"))
    reverted_by: Mapped[int | None] = mapped_column(ForeignKey("taxonomy_revisions.id"))
    applied_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, server_default=sql_text("CURRENT_TIMESTAMP"), nullable=False
    )


class Digest(Base):
    """Generated digest (4-section JSON + rendered markdown)."""

    __tablename__ = "digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    period_start: Mapped[datetime] = mapped_column(TS, nullable=False)
    period_end: Mapped[datetime] = mapped_column(TS, nullable=False)
    content_json: Mapped[dict] = mapped_column(_json(), nullable=False)
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str | None] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class DigestEntity(Base):
    """Digest membership (many-to-many) — an entity can appear in multiple digests."""

    __tablename__ = "digest_entities"
    __table_args__ = (
        UniqueConstraint("digest_id", "topic_entity_id", name="uq_digest_entities"),
        Index("idx_digest_entities_digest", "digest_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    digest_id: Mapped[int] = mapped_column(ForeignKey("digests.id"), nullable=False)
    topic_entity_id: Mapped[int] = mapped_column(ForeignKey("topic_entities.id"), nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer)


class FeedKeyword(Base):
    """User keyword configuration (include / suppress)."""

    __tablename__ = "feed_keywords"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    keyword_type: Mapped[str] = mapped_column(Text, nullable=False, default="include")
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class FeedView(Base):
    """Saved presentation view (configurable tabs / filters)."""

    __tablename__ = "feed_views"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    filter_json: Mapped[dict] = mapped_column(_json(), nullable=False)
    sort_json: Mapped[dict | None] = mapped_column(_json())
    is_default: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class FeedFeedback(Base):
    """User feedback on topic entities (-1 noise / 0 neutral / 1 valuable)."""

    __tablename__ = "feed_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_entity_id: Mapped[int] = mapped_column(ForeignKey("topic_entities.id"), nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class InterestProfileRow(Base):
    """The single persisted global D-2 interest-profile centroid."""

    __tablename__ = "interest_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    centroid: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    alpha: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class ClassifierMetricSnapshot(Base):
    """Durable aggregate evidence for one resumable classification pass (D-15)."""

    __tablename__ = "classifier_metric_snapshots"
    __table_args__ = (
        UniqueConstraint("classification_pass_id", name="uq_classifier_metric_snapshots_pass"),
        Index("idx_classifier_metric_snapshots_status_started", "status", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    classification_pass_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="in_progress")
    started_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(TS)
    eligibility_high_water_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_item_cursor: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accumulator_json: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    effective_config_json: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)


class FeedRun(Base):
    """Ingestion run tracking (incremental-sync high-water-mark cursor)."""

    __tablename__ = "feed_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(TS)
    high_water_mark: Mapped[str | None] = mapped_column(Text)
    items_fetched: Mapped[int] = mapped_column(Integer, default=0)
    items_passed_filter: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    error_message: Mapped[str | None] = mapped_column(Text)
    source_state: Mapped[dict | None] = mapped_column(_json())
    diagnostics: Mapped[dict | None] = mapped_column(_json())


class SourceMaintenanceRun(Base):
    """Content-free lifecycle outcome for a source before ingestion."""

    __tablename__ = "source_maintenance_runs"
    __table_args__ = (Index("idx_source_maintenance_runs_source_started", "source", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(TS)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    items_checked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_purged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reason_counts: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)


class YouTubeVideoState(Base):
    """Operational lifecycle state for one official YouTube playlist pointer."""

    __tablename__ = "youtube_video_states"
    __table_args__ = (
        UniqueConstraint("feed_item_id", name="uq_youtube_video_states_feed_item"),
        UniqueConstraint("playlist_id", "video_id", name="uq_youtube_video_states_playlist_video"),
        Index(
            "idx_youtube_video_states_transcript_due",
            "transcript_status",
            "transcript_next_attempt_at",
        ),
        Index("idx_youtube_video_states_metadata_refreshed", "metadata_refreshed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feed_item_id: Mapped[int] = mapped_column(ForeignKey("feed_items.id"), nullable=False)
    playlist_id: Mapped[str] = mapped_column(Text, nullable=False)
    video_id: Mapped[str] = mapped_column(Text, nullable=False)
    membership_status: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    last_seen_in_playlist_at: Mapped[datetime | None] = mapped_column(TS)
    metadata_refreshed_at: Mapped[datetime | None] = mapped_column(TS)
    metadata_refresh_error: Mapped[str | None] = mapped_column(Text)
    transcript_status: Mapped[str] = mapped_column(Text, nullable=False, default="disabled")
    transcript_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    transcript_next_attempt_at: Mapped[datetime | None] = mapped_column(TS)
    transcript_provider: Mapped[str | None] = mapped_column(Text)
    transcript_language: Mapped[str | None] = mapped_column(Text)
    transcript_kind: Mapped[str | None] = mapped_column(Text)
    transcript_fetched_at: Mapped[datetime | None] = mapped_column(TS)
    transcript_feed_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("feed_items.id"), unique=True
    )
    last_error_code: Mapped[str | None] = mapped_column(Text)


class MergeLog(Base):
    """Reversible/auditable merge log — every grouping decision (D-8)."""

    __tablename__ = "merge_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_entity_id: Mapped[int] = mapped_column(ForeignKey("topic_entities.id"), nullable=False)
    feed_item_id: Mapped[int] = mapped_column(ForeignKey("feed_items.id"), nullable=False)
    match_method: Mapped[str] = mapped_column(Text, nullable=False)
    match_confidence: Mapped[float | None] = mapped_column(Float)
    actor: Mapped[str] = mapped_column(Text, nullable=False, default="auto")
    action: Mapped[str] = mapped_column(Text, nullable=False, default="merge")
    evidence: Mapped[dict | None] = mapped_column(_json())
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class StableIdAlias(Base):
    """Alias / redirect map — canonical-id survivorship (D-8)."""

    __tablename__ = "stable_id_alias"
    __table_args__ = (UniqueConstraint("kind", "from_value", name="uq_stable_id_alias"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    from_value: Mapped[str] = mapped_column(Text, nullable=False)
    to_value: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class MergeCandidate(Base):
    """Non-blocking review queue for tau_low..tau_high candidate merges (D-8)."""

    __tablename__ = "merge_candidates"
    __table_args__ = (
        UniqueConstraint("entity_a_id", "entity_b_id", name="uq_merge_candidates"),
        Index("idx_merge_candidates_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_a_id: Mapped[int] = mapped_column(ForeignKey("topic_entities.id"), nullable=False)
    entity_b_id: Mapped[int] = mapped_column(ForeignKey("topic_entities.id"), nullable=False)
    similarity: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    resolved_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(TS)


class DroppedItem(Base):
    """False-negative audit — every DROP logged, never hard-deleted (D-12)."""

    __tablename__ = "dropped_items"
    __table_args__ = (Index("idx_dropped_items_review", "reviewed", "drop_confidence"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    drop_reason: Mapped[str] = mapped_column(Text, nullable=False)
    drop_confidence: Mapped[float | None] = mapped_column(Float)
    dropped_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)
    reviewed: Mapped[int] = mapped_column(Integer, default=0)
    rescued: Mapped[int] = mapped_column(Integer, default=0)


# Backlog-mining draft queue.  These records deliberately live in Rumen's schema: a candidate is
# review evidence, not an authoritative task-store object.
class BacklogMiningRun(Base):
    __tablename__ = "backlog_mining_runs"
    __table_args__ = (Index("idx_backlog_mining_runs_status_started", "status", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    config_snapshot: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    counts: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    error_payload: Mapped[dict | None] = mapped_column(_json())
    started_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(TS)


class BacklogMappingReview(Base):
    """One durable routing decision: auto-created mapping or operator review."""

    __tablename__ = "backlog_mapping_reviews"
    __table_args__ = (
        Index("idx_backlog_mapping_reviews_run", "mining_run_id"),
        Index("idx_backlog_mapping_reviews_status", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mining_run_id: Mapped[int] = mapped_column(ForeignKey("backlog_mining_runs.id"), nullable=False)
    topic_entity_id: Mapped[int] = mapped_column(ForeignKey("topic_entities.id"), nullable=False)
    category: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    target_identity: Mapped[str | None] = mapped_column(Text)
    decision_inputs: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    notification_status: Mapped[str | None] = mapped_column(Text)
    notification_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class BacklogCandidate(Base):
    __tablename__ = "backlog_candidates"
    __table_args__ = (
        Index("idx_backlog_candidates_state_created", "state", "created_at"),
        Index("idx_backlog_candidates_entity", "topic_entity_id"),
        Index("idx_backlog_candidates_family", "topic_entity_id", "target_identity"),
        Index(
            "uq_backlog_candidates_active_fingerprint",
            "fingerprint",
            unique=True,
            sqlite_where=sql_text("state IN ('draft', 'accepted', 'edited', 'deferred')"),
            postgresql_where=sql_text("state IN ('draft', 'accepted', 'edited', 'deferred')"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_entity_id: Mapped[int | None] = mapped_column(ForeignKey("topic_entities.id"))
    digest_id: Mapped[int | None] = mapped_column(ForeignKey("digests.id"))
    # Nullable forever: Phase-1 rows may have been created with no resolvable target.
    target_identity: Mapped[str | None] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    source_snapshot: Mapped[dict] = mapped_column(_json(), nullable=False)
    content_snapshot: Mapped[dict] = mapped_column(_json(), nullable=False)
    assessment: Mapped[dict] = mapped_column(_json(), nullable=False)
    configuration_snapshot: Mapped[dict] = mapped_column(_json(), nullable=False)
    state: Mapped[str] = mapped_column(
        Text, nullable=False, default="draft", server_default=sql_text("'draft'")
    )
    superseded_by_candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("backlog_candidates.id")
    )
    merge_target_candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("backlog_candidates.id")
    )
    created_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, server_default=sql_text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TS,
        default=_utcnow,
        onupdate=_utcnow,
        server_default=sql_text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


class BacklogCandidateEvidence(Base):
    __tablename__ = "backlog_candidate_evidence"
    __table_args__ = (Index("idx_backlog_candidate_evidence_candidate", "candidate_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("backlog_candidates.id"), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    feed_item_id: Mapped[int | None] = mapped_column(ForeignKey("feed_items.id"))
    excerpt: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(_json())
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class BacklogTaskMatch(Base):
    __tablename__ = "backlog_task_matches"
    __table_args__ = (
        Index("idx_backlog_task_matches_candidate", "candidate_id"),
        Index("idx_backlog_task_matches_external_key", "task_store", "external_task_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("backlog_candidates.id"), nullable=False)
    mining_run_id: Mapped[int | None] = mapped_column(ForeignKey("backlog_mining_runs.id"))
    task_store: Mapped[str] = mapped_column(Text, nullable=False)
    external_task_key: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    closure_reason_text: Mapped[str | None] = mapped_column(Text)
    match_evidence: Mapped[dict | None] = mapped_column(_json())
    checked_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class BacklogCandidateRecommendation(Base):
    __tablename__ = "backlog_candidate_recommendations"
    __table_args__ = (Index("idx_backlog_candidate_recommendations_candidate", "candidate_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("backlog_candidates.id"), nullable=False)
    task_store: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_cc_task: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class BacklogCandidateDisposition(Base):
    __tablename__ = "backlog_candidate_dispositions"
    __table_args__ = (
        UniqueConstraint("frame_id", name="uq_backlog_candidate_dispositions_frame"),
        Index(
            "idx_backlog_candidate_dispositions_fingerprint_sequence", "fingerprint", "sequence_no"
        ),
        Index("idx_backlog_candidate_dispositions_stack", "fingerprint", "stack_state"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    frame_id: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_frame_id: Mapped[str | None] = mapped_column(Text)
    record_kind: Mapped[str] = mapped_column(Text, nullable=False, default="disposition")
    action: Mapped[str] = mapped_column(Text, nullable=False)
    target_candidate_id: Mapped[int | None] = mapped_column(ForeignKey("backlog_candidates.id"))
    moved_frame_id: Mapped[str | None] = mapped_column(Text)
    affected_candidate_ids: Mapped[list] = mapped_column(_json(), nullable=False, default=list)
    preimage: Mapped[dict] = mapped_column(_json(), nullable=False)
    postimage: Mapped[dict] = mapped_column(_json(), nullable=False)
    action_payload: Mapped[dict | None] = mapped_column(_json())
    abandoned_frame_ids: Mapped[list] = mapped_column(_json(), nullable=False, default=list)
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    stack_state: Mapped[str] = mapped_column(Text, nullable=False, default="applied")
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)


class BacklogCandidateTaskLink(Base):
    __tablename__ = "backlog_candidate_task_links"
    __table_args__ = (
        UniqueConstraint(
            "fingerprint", "task_store", "external_task_key", name="uq_backlog_candidate_task_link"
        ),
        Index("idx_backlog_candidate_task_links_fingerprint", "fingerprint"),
        Index("idx_backlog_candidate_task_links_external_key", "task_store", "external_task_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    task_store: Mapped[str] = mapped_column(Text, nullable=False)
    external_task_key: Mapped[str] = mapped_column(Text, nullable=False)
    origin_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("backlog_candidates.id"), nullable=False
    )
    last_reconciled_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("backlog_candidates.id"), nullable=False
    )
    relationship_role: Mapped[str | None] = mapped_column(Text)
    first_operation_id: Mapped[str | None] = mapped_column(Text)
    last_operation_id: Mapped[str | None] = mapped_column(Text)
    last_observed_title: Mapped[str | None] = mapped_column(Text)
    last_observed_state: Mapped[str | None] = mapped_column(Text)
    last_observed_closure_reason: Mapped[str | None] = mapped_column(Text)
    closure_class: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    closure_class_source: Mapped[str | None] = mapped_column(Text)
    checked_at: Mapped[datetime | None] = mapped_column(TS)


class BacklogCandidatePromotion(Base):
    __tablename__ = "backlog_candidate_promotions"
    __table_args__ = (
        UniqueConstraint("operation_id", name="uq_backlog_candidate_promotions_operation"),
        Index("idx_backlog_candidate_promotions_fingerprint", "fingerprint"),
        Index("idx_backlog_candidate_promotions_state", "state"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    operation_id: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("backlog_candidates.id"), nullable=False)
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    primitives: Mapped[list] = mapped_column(_json(), nullable=False, default=list)
    selected_task_links: Mapped[list] = mapped_column(_json(), nullable=False, default=list)
    requested_payload: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False, default="planned")
    lease_expires_at: Mapped[datetime | None] = mapped_column(TS)
    step_receipts: Mapped[list] = mapped_column(_json(), nullable=False, default=list)
    correlation_markers: Mapped[dict] = mapped_column(_json(), nullable=False, default=dict)
    before_observation: Mapped[dict | None] = mapped_column(_json())
    after_observation: Mapped[dict | None] = mapped_column(_json())
    error_payload: Mapped[dict | None] = mapped_column(_json())
    created_at: Mapped[datetime] = mapped_column(TS, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TS, default=_utcnow, onupdate=_utcnow, nullable=False
    )
