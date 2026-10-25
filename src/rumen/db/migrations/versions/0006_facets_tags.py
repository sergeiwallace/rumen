"""facets and tags

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "item_concepts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item_table", sa.Text(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("concept_id", sa.Integer(), nullable=False),
        sa.Column("facet", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["concept_id"], ["skos_concepts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "item_table", "item_id", "concept_id", name="uq_item_concepts_item_concept"
        ),
    )
    with op.batch_alter_table("item_concepts", schema=None) as batch_op:
        batch_op.create_index("idx_item_concepts_item", ["item_table", "item_id"], unique=False)
        batch_op.create_index("idx_item_concepts_facet", ["facet", "concept_id"], unique=False)
        batch_op.create_index("idx_item_concepts_concept", ["concept_id"], unique=False)
    for table in ("feed_items", "topic_entities"):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(sa.Column("tags", _json(), nullable=True))


def downgrade() -> None:
    for table in ("topic_entities", "feed_items"):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column("tags")
    with op.batch_alter_table("item_concepts", schema=None) as batch_op:
        batch_op.drop_index("idx_item_concepts_concept")
        batch_op.drop_index("idx_item_concepts_facet")
        batch_op.drop_index("idx_item_concepts_item")
    op.drop_table("item_concepts")
