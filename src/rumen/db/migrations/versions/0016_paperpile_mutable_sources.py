"""add mutable source projection fields

Revision ID: 0016
Revises: 0015
"""

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("feed_items") as batch:
        batch.add_column(sa.Column("source_revision_digest", sa.Text()))
        batch.add_column(sa.Column("source_last_seen_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("source_deleted_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    with op.batch_alter_table("feed_items") as batch:
        batch.drop_column("source_deleted_at")
        batch.drop_column("source_last_seen_at")
        batch.drop_column("source_revision_digest")
