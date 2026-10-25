"""Add the Phase-2 pre-assessment candidate family key.

Revision ID: 0019
Revises: 0018
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("backlog_candidates", sa.Column("target_identity", sa.Text(), nullable=True))
    op.create_index(
        "idx_backlog_candidates_family",
        "backlog_candidates",
        ["topic_entity_id", "target_identity"],
    )
    # The application migration runner may supply the one Phase-1 target active
    # at upgrade time.  A deployment without that mapping intentionally retains
    # NULL: there is no per-row target data from which to infer one.  This remains
    # one set-based update because the expected Phase-1 table is small; larger
    # deployments should schedule this migration in a maintenance window rather
    # than guess at a dialect-specific batching/online-index strategy here.
    target_identity = op.get_context().config.attributes.get("backlog_target_identity")
    if isinstance(target_identity, str) and target_identity:
        op.execute(
            sa.text(
                "UPDATE backlog_candidates "
                "SET target_identity = :target_identity "
                "WHERE target_identity IS NULL"
            ).bindparams(target_identity=target_identity)
        )


def downgrade() -> None:
    op.drop_index("idx_backlog_candidates_family", table_name="backlog_candidates")
    op.drop_column("backlog_candidates", "target_identity")
