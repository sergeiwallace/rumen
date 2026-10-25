"""human classification corrections and gold-label provenance

Revision ID: 0009
Revises: 0008
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


_item_concepts = sa.table(
    "item_concepts",
    sa.column("item_table", sa.Text()),
    sa.column("item_id", sa.Integer()),
    sa.column("concept_id", sa.Integer()),
    sa.column("facet", sa.Text()),
    sa.column("method", sa.Text()),
)
_corrections = sa.table(
    "classification_corrections",
    sa.column("item_table", sa.Text()),
    sa.column("item_id", sa.Integer()),
    sa.column("facet", sa.Text()),
    sa.column("action", sa.Text()),
    sa.column("prior_concept_ids", _json()),
    sa.column("corrected_concept_ids", _json()),
    sa.column("classifier_context_json", _json()),
    sa.column("actor", sa.Text()),
    sa.column("source", sa.Text()),
    sa.column("rationale", sa.Text()),
    sa.column("status", sa.Text()),
    sa.column("decision_actor", sa.Text()),
    sa.column("decision_rationale", sa.Text()),
    sa.column("submitted_at", sa.DateTime(timezone=True)),
    sa.column("decided_at", sa.DateTime(timezone=True)),
)


def upgrade() -> None:
    op.create_table(
        "classification_corrections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item_table", sa.Text(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("facet", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("prior_concept_ids", _json(), nullable=False),
        sa.Column("corrected_concept_ids", _json(), nullable=False),
        sa.Column("classifier_context_json", _json(), nullable=True),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="proposed"),
        sa.Column("decision_actor", sa.Text(), nullable=True),
        sa.Column("decision_rationale", sa.Text(), nullable=True),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["supersedes"], ["classification_corrections.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("classification_corrections", schema=None) as batch_op:
        batch_op.create_index(
            "idx_classification_corrections_item_facet",
            ["item_table", "item_id", "facet", "submitted_at"],
            unique=False,
        )
        batch_op.create_index(
            "idx_classification_corrections_telemetry",
            ["status", "facet", "decided_at"],
            unique=False,
        )
    backfill_legacy_human_assignments(op.get_bind())


def downgrade() -> None:
    with op.batch_alter_table("classification_corrections", schema=None) as batch_op:
        batch_op.drop_index("idx_classification_corrections_telemetry")
        batch_op.drop_index("idx_classification_corrections_item_facet")
    op.drop_table("classification_corrections")


def backfill_legacy_human_assignments(bind: sa.Connection) -> None:
    """Document existing T-17 gold rows without changing their current projection."""
    rows = bind.execute(
        sa.select(
            _item_concepts.c.item_table,
            _item_concepts.c.item_id,
            _item_concepts.c.facet,
            _item_concepts.c.concept_id,
        )
        .where(_item_concepts.c.method.like("human:%"))
        .order_by(
            _item_concepts.c.item_table,
            _item_concepts.c.item_id,
            _item_concepts.c.facet,
            _item_concepts.c.concept_id,
        )
    ).all()
    grouped: dict[tuple[str, int, str], list[int]] = {}
    for row in rows:
        grouped.setdefault((row.item_table, row.item_id, row.facet), []).append(row.concept_id)
    now = datetime.now(UTC)
    events = [
        {
            "item_table": item_table,
            "item_id": item_id,
            "facet": facet,
            "action": "approval",
            "prior_concept_ids": concept_ids,
            "corrected_concept_ids": concept_ids,
            "classifier_context_json": {"legacy_provenance": "unavailable"},
            "actor": "system:legacy-human-import",
            "source": "migration",
            "rationale": "Original human-review provenance is unavailable; imported existing human labels.",
            "status": "accepted",
            "decision_actor": "system:legacy-human-import",
            "decision_rationale": "Original human-review provenance is unavailable.",
            "submitted_at": now,
            "decided_at": now,
        }
        for (item_table, item_id, facet), concept_ids in grouped.items()
    ]
    if events:
        bind.execute(_corrections.insert(), events)
