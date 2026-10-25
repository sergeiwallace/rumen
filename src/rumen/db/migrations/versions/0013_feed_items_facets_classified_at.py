"""feed item facet classification marker

Revision ID: 0013
Revises: 0012
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "feed_items", sa.Column("facets_classified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.execute(
        sa.text(
            "UPDATE feed_items SET facets_classified_at = :timestamp "
            "WHERE classification_method IS NOT NULL"
        ).bindparams(timestamp=datetime.now(UTC))
    )


def downgrade() -> None:
    op.drop_column("feed_items", "facets_classified_at")
