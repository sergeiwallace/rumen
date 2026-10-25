"""taxonomy store

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "skos_concepts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("facet", sa.Text(), nullable=False),
        sa.Column("pref_label", sa.Text(), nullable=False),
        sa.Column("notation", sa.Text(), nullable=True),
        sa.Column("scheme", sa.Text(), nullable=False),
        sa.Column("scheme_version", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("skos_concepts", schema=None) as batch_op:
        batch_op.create_index("idx_skos_concepts_facet", ["facet", "status"], unique=False)

    op.create_table(
        "skos_relations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_concept_id", sa.Integer(), nullable=False),
        sa.Column("target_concept_id", sa.Integer(), nullable=False),
        sa.Column("relation_type", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["source_concept_id"], ["skos_concepts.id"]),
        sa.ForeignKeyConstraint(["target_concept_id"], ["skos_concepts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_concept_id", "target_concept_id", "relation_type", name="uq_skos_relations"
        ),
    )
    with op.batch_alter_table("skos_relations", schema=None) as batch_op:
        batch_op.create_index(
            "idx_skos_relations_source", ["source_concept_id", "relation_type"], unique=False
        )
        batch_op.create_index(
            "idx_skos_relations_target", ["target_concept_id", "relation_type"], unique=False
        )

    op.create_table(
        "taxonomy_revisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("audit_run_id", sa.Text(), nullable=False),
        sa.Column("op_type", sa.Text(), nullable=False),
        sa.Column("concept_ids", _json(), nullable=False),
        sa.Column("before_json", _json(), nullable=True),
        sa.Column("after_json", _json(), nullable=True),
        sa.Column("actor", sa.Text(), nullable=False, server_default="system:taxonomy-audit"),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("reversible", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reverts", sa.Integer(), nullable=True),
        sa.Column("reverted_by", sa.Integer(), nullable=True),
        sa.Column(
            "applied_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["reverted_by"], ["taxonomy_revisions.id"]),
        sa.ForeignKeyConstraint(["reverts"], ["taxonomy_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("taxonomy_revisions", schema=None) as batch_op:
        batch_op.create_index("idx_taxonomy_revisions_run", ["audit_run_id"], unique=False)


def downgrade() -> None:
    op.drop_table("taxonomy_revisions")
    op.drop_table("skos_relations")
    op.drop_table("skos_concepts")
