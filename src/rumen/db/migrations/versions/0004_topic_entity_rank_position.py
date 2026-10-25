"""topic entity rank position

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("topic_entities", schema=None) as batch_op:
        batch_op.add_column(sa.Column("rank_position", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("is_exploration", sa.Integer(), nullable=True, server_default="0")
        )
        batch_op.create_index("idx_topic_entities_rank_position", ["rank_position"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("topic_entities", schema=None) as batch_op:
        batch_op.drop_index("idx_topic_entities_rank_position")
        batch_op.drop_column("is_exploration")
        batch_op.drop_column("rank_position")
