"""Add replay-oriented source state and attempt diagnostics to feed runs.

Revision ID: 0018
Revises: 0017
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("feed_runs", sa.Column("source_state", _json()))
    op.add_column("feed_runs", sa.Column("diagnostics", _json()))


def downgrade() -> None:
    op.drop_column("feed_runs", "diagnostics")
    op.drop_column("feed_runs", "source_state")
