"""classifier metrics snapshots

Revision ID: 0010
Revises: 0009
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "classifier_metric_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("classification_pass_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="in_progress"),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("eligibility_high_water_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_item_cursor", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("accumulator_json", _json(), nullable=False),
        sa.Column("effective_config_json", _json(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("classification_pass_id", name="uq_classifier_metric_snapshots_pass"),
    )
    with op.batch_alter_table("classifier_metric_snapshots", schema=None) as batch_op:
        batch_op.create_index(
            "idx_classifier_metric_snapshots_status_started", ["status", "started_at"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("classifier_metric_snapshots", schema=None) as batch_op:
        batch_op.drop_index("idx_classifier_metric_snapshots_status_started")
    op.drop_table("classifier_metric_snapshots")
