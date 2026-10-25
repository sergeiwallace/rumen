"""Persist Phase-2 category-routing decisions for review and audit.

Revision ID: 0020
Revises: 0019
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "backlog_mapping_reviews",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "mining_run_id", sa.Integer(), sa.ForeignKey("backlog_mining_runs.id"), nullable=False
        ),
        sa.Column(
            "topic_entity_id", sa.Integer(), sa.ForeignKey("topic_entities.id"), nullable=False
        ),
        sa.Column("category", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("target_identity", sa.Text()),
        sa.Column("decision_inputs", _json(), nullable=False),
        sa.Column("notification_status", sa.Text()),
        sa.Column("notification_error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_backlog_mapping_reviews_run", "backlog_mapping_reviews", ["mining_run_id"])
    op.create_index(
        "idx_backlog_mapping_reviews_status",
        "backlog_mapping_reviews",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_backlog_mapping_reviews_status", table_name="backlog_mapping_reviews")
    op.drop_index("idx_backlog_mapping_reviews_run", table_name="backlog_mapping_reviews")
    op.drop_table("backlog_mapping_reviews")
