"""server defaults and bookmark link metadata.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _defaults(table: str, columns: dict[str, sa.DefaultClause | None]) -> None:
    """Use batch mode so SQLite recreates tables when ALTER DEFAULT is unavailable."""
    with op.batch_alter_table(table, schema=None) as batch_op:
        for name, default in columns.items():
            batch_op.alter_column(name, server_default=default)


def upgrade() -> None:
    now = sa.text("CURRENT_TIMESTAMP")
    _defaults(
        "topic_entities", {"source_count": sa.text("1"), "created_at": now, "updated_at": now}
    )
    _defaults("digests", {"generated_at": now})
    _defaults("feed_keywords", {"keyword_type": sa.text("'include'"), "created_at": now})
    _defaults("feed_views", {"is_default": sa.text("0"), "created_at": now})
    _defaults("feed_feedback", {"created_at": now})
    _defaults(
        "feed_runs",
        {
            "items_fetched": sa.text("0"),
            "items_passed_filter": sa.text("0"),
            "status": sa.text("'running'"),
        },
    )
    _defaults(
        "merge_log", {"actor": sa.text("'auto'"), "action": sa.text("'merge'"), "created_at": now}
    )
    _defaults("stable_id_alias", {"created_at": now})
    _defaults("merge_candidates", {"status": sa.text("'pending'"), "created_at": now})
    _defaults(
        "dropped_items", {"dropped_at": now, "reviewed": sa.text("0"), "rescued": sa.text("0")}
    )
    with op.batch_alter_table("feed_items", schema=None) as batch_op:
        batch_op.alter_column("ingested_at", server_default=now)
        batch_op.add_column(
            sa.Column(
                "link_metadata",
                sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
                nullable=True,
            )
        )


def downgrade() -> None:
    tables = {
        "topic_entities": ("source_count", "created_at", "updated_at"),
        "digests": ("generated_at",),
        "feed_keywords": ("keyword_type", "created_at"),
        "feed_views": ("is_default", "created_at"),
        "feed_feedback": ("created_at",),
        "feed_runs": ("items_fetched", "items_passed_filter", "status"),
        "merge_log": ("actor", "action", "created_at"),
        "stable_id_alias": ("created_at",),
        "merge_candidates": ("status", "created_at"),
        "dropped_items": ("dropped_at", "reviewed", "rescued"),
    }
    for table, columns in tables.items():
        _defaults(table, {column: None for column in columns})
    with op.batch_alter_table("feed_items", schema=None) as batch_op:
        batch_op.drop_column("link_metadata")
        batch_op.alter_column("ingested_at", server_default=None)
