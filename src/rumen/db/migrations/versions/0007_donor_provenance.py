"""donor taxonomy provenance

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("skos_concepts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_uri", sa.Text(), nullable=True))
    with op.batch_alter_table("skos_relations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_scheme", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("source_version", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("target_scheme", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("target_version", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("reviewer", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("review_date", sa.Date(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("skos_relations", schema=None) as batch_op:
        batch_op.drop_column("review_date")
        batch_op.drop_column("reviewer")
        batch_op.drop_column("target_version")
        batch_op.drop_column("target_scheme")
        batch_op.drop_column("source_version")
        batch_op.drop_column("source_scheme")
    with op.batch_alter_table("skos_concepts", schema=None) as batch_op:
        batch_op.drop_column("source_uri")
