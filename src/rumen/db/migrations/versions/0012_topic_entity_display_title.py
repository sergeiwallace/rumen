"""topic entity display title

Revision ID: 0012
Revises: 0011
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("topic_entities", sa.Column("display_title", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("topic_entities", "display_title")
