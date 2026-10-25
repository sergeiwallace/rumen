"""Persistent backlog-mining queue.

Revision ID: 0014
Revises: 0013
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    json = _json()
    op.create_table(
        "backlog_mining_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("config_snapshot", json, nullable=False),
        sa.Column("counts", json, nullable=False),
        sa.Column("error_payload", json),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "idx_backlog_mining_runs_status_started", "backlog_mining_runs", ["status", "started_at"]
    )
    op.create_table(
        "backlog_candidates",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("topic_entity_id", sa.Integer(), sa.ForeignKey("topic_entities.id")),
        sa.Column("digest_id", sa.Integer(), sa.ForeignKey("digests.id")),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("source_snapshot", json, nullable=False),
        sa.Column("content_snapshot", json, nullable=False),
        sa.Column("assessment", json, nullable=False),
        sa.Column("configuration_snapshot", json, nullable=False),
        sa.Column("state", sa.Text(), nullable=False, server_default=sa.text("'draft'")),
        sa.Column(
            "superseded_by_candidate_id", sa.Integer(), sa.ForeignKey("backlog_candidates.id")
        ),
        sa.Column(
            "merge_target_candidate_id", sa.Integer(), sa.ForeignKey("backlog_candidates.id")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index(
        "idx_backlog_candidates_state_created", "backlog_candidates", ["state", "created_at"]
    )
    op.create_index("idx_backlog_candidates_entity", "backlog_candidates", ["topic_entity_id"])
    op.create_index(
        "uq_backlog_candidates_active_fingerprint",
        "backlog_candidates",
        ["fingerprint"],
        unique=True,
        sqlite_where=sa.text("state IN ('draft', 'accepted', 'edited', 'deferred')"),
        postgresql_where=sa.text("state IN ('draft', 'accepted', 'edited', 'deferred')"),
    )
    op.create_table(
        "backlog_candidate_evidence",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "candidate_id", sa.Integer(), sa.ForeignKey("backlog_candidates.id"), nullable=False
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("feed_item_id", sa.Integer(), sa.ForeignKey("feed_items.id")),
        sa.Column("excerpt", sa.Text()),
        sa.Column("metadata_json", json),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_backlog_candidate_evidence_candidate", "backlog_candidate_evidence", ["candidate_id"]
    )
    op.create_table(
        "backlog_task_matches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "candidate_id", sa.Integer(), sa.ForeignKey("backlog_candidates.id"), nullable=False
        ),
        sa.Column("mining_run_id", sa.Integer(), sa.ForeignKey("backlog_mining_runs.id")),
        sa.Column("task_store", sa.Text(), nullable=False),
        sa.Column("external_task_key", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("closure_reason_text", sa.Text()),
        sa.Column("match_evidence", json),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_backlog_task_matches_candidate", "backlog_task_matches", ["candidate_id"])
    op.create_index(
        "idx_backlog_task_matches_external_key",
        "backlog_task_matches",
        ["task_store", "external_task_key"],
    )
    op.create_table(
        "backlog_candidate_recommendations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "candidate_id", sa.Integer(), sa.ForeignKey("backlog_candidates.id"), nullable=False
        ),
        sa.Column("task_store", sa.Text(), nullable=False),
        sa.Column("recommended_cc_task", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_backlog_candidate_recommendations_candidate",
        "backlog_candidate_recommendations",
        ["candidate_id"],
    )
    op.create_table(
        "backlog_candidate_dispositions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("frame_id", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("parent_frame_id", sa.Text()),
        sa.Column(
            "record_kind", sa.Text(), nullable=False, server_default=sa.text("'disposition'")
        ),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("target_candidate_id", sa.Integer(), sa.ForeignKey("backlog_candidates.id")),
        sa.Column("moved_frame_id", sa.Text()),
        sa.Column("affected_candidate_ids", json, nullable=False),
        sa.Column("preimage", json, nullable=False),
        sa.Column("postimage", json, nullable=False),
        sa.Column("action_payload", json),
        sa.Column("abandoned_frame_ids", json, nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("stack_state", sa.Text(), nullable=False, server_default=sa.text("'applied'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("frame_id", name="uq_backlog_candidate_dispositions_frame"),
    )
    op.create_index(
        "idx_backlog_candidate_dispositions_fingerprint_sequence",
        "backlog_candidate_dispositions",
        ["fingerprint", "sequence_no"],
    )
    op.create_index(
        "idx_backlog_candidate_dispositions_stack",
        "backlog_candidate_dispositions",
        ["fingerprint", "stack_state"],
    )
    op.create_table(
        "backlog_candidate_task_links",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("task_store", sa.Text(), nullable=False),
        sa.Column("external_task_key", sa.Text(), nullable=False),
        sa.Column(
            "origin_candidate_id",
            sa.Integer(),
            sa.ForeignKey("backlog_candidates.id"),
            nullable=False,
        ),
        sa.Column(
            "last_reconciled_candidate_id",
            sa.Integer(),
            sa.ForeignKey("backlog_candidates.id"),
            nullable=False,
        ),
        sa.Column("relationship_role", sa.Text()),
        sa.Column("first_operation_id", sa.Text()),
        sa.Column("last_operation_id", sa.Text()),
        sa.Column("last_observed_title", sa.Text()),
        sa.Column("last_observed_state", sa.Text()),
        sa.Column("last_observed_closure_reason", sa.Text()),
        sa.Column("closure_class", sa.Text(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("closure_class_source", sa.Text()),
        sa.Column("checked_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "fingerprint", "task_store", "external_task_key", name="uq_backlog_candidate_task_link"
        ),
    )
    op.create_index(
        "idx_backlog_candidate_task_links_fingerprint",
        "backlog_candidate_task_links",
        ["fingerprint"],
    )
    op.create_index(
        "idx_backlog_candidate_task_links_external_key",
        "backlog_candidate_task_links",
        ["task_store", "external_task_key"],
    )
    op.create_table(
        "backlog_candidate_promotions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("operation_id", sa.Text(), nullable=False),
        sa.Column(
            "candidate_id", sa.Integer(), sa.ForeignKey("backlog_candidates.id"), nullable=False
        ),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("primitives", json, nullable=False),
        sa.Column("selected_task_links", json, nullable=False),
        sa.Column("requested_payload", json, nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False, server_default=sa.text("'planned'")),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("step_receipts", json, nullable=False),
        sa.Column("correlation_markers", json, nullable=False),
        sa.Column("before_observation", json),
        sa.Column("after_observation", json),
        sa.Column("error_payload", json),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("operation_id", name="uq_backlog_candidate_promotions_operation"),
    )
    op.create_index(
        "idx_backlog_candidate_promotions_fingerprint",
        "backlog_candidate_promotions",
        ["fingerprint"],
    )
    op.create_index(
        "idx_backlog_candidate_promotions_state", "backlog_candidate_promotions", ["state"]
    )


def downgrade() -> None:
    for table in (
        "backlog_candidate_promotions",
        "backlog_candidate_task_links",
        "backlog_candidate_dispositions",
        "backlog_candidate_recommendations",
        "backlog_task_matches",
        "backlog_candidate_evidence",
        "backlog_candidates",
        "backlog_mining_runs",
    ):
        op.drop_table(table)
