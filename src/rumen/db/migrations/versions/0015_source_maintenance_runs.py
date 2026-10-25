"""Add source maintenance ledger.

Revision ID: 0015
Revises: 0014
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "source_maintenance_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("items_checked", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_purged", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reason_counts", json, nullable=False),
        sa.Column("error_message", sa.Text()),
    )
    op.create_index(
        "idx_source_maintenance_runs_source_started",
        "source_maintenance_runs",
        ["source", "started_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_source_maintenance_runs_source_started", table_name="source_maintenance_runs"
    )
    op.drop_table("source_maintenance_runs")
