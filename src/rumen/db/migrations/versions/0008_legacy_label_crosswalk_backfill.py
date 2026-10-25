"""backfill documented legacy category labels through the topic-spine crosswalk

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import sqlalchemy as sa
from alembic import op

from rumen.taxonomy.legacy_crosswalk import (
    LEGACY_CATEGORY_CROSSWALK,
    TOPIC_CONCEPTS,
    TOPIC_SPINE_SCHEME,
)

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BATCH_SIZE = 500
_crosswalk = sa.table(
    "legacy_category_crosswalk",
    sa.column("legacy_label", sa.Text()),
    sa.column("concept_id", sa.Integer()),
    sa.column("top_level_concept_id", sa.Integer()),
)
_concepts = sa.table(
    "skos_concepts",
    sa.column("id", sa.Integer()),
    sa.column("facet", sa.Text()),
    sa.column("pref_label", sa.Text()),
    sa.column("notation", sa.Text()),
    sa.column("scheme", sa.Text()),
    sa.column("scheme_version", sa.Text()),
    sa.column("source_uri", sa.Text()),
    sa.column("status", sa.Text()),
)
_item_concepts = sa.table(
    "item_concepts",
    sa.column("item_table", sa.Text()),
    sa.column("item_id", sa.Integer()),
    sa.column("concept_id", sa.Integer()),
    sa.column("facet", sa.Text()),
    sa.column("confidence", sa.Float()),
    sa.column("method", sa.Text()),
)


def upgrade() -> None:
    """Seed reviewed targets and backfill the documented labels in durable batches."""
    bind = op.get_bind()
    op.create_table(
        "legacy_category_crosswalk",
        sa.Column("legacy_label", sa.Text(), primary_key=True),
        sa.Column("concept_id", sa.Integer(), nullable=False),
        sa.Column("top_level_concept_id", sa.Integer(), nullable=False),
        prefixes=["TEMPORARY"],
    )
    _seed_concepts(bind)
    bind.execute(
        _crosswalk.insert(),
        [
            {
                "legacy_label": label,
                "concept_id": target.id,
                # The documented partial seed maps directly to spine roots.  Keeping
                # this column makes the backfill explicit about category derivation.
                "top_level_concept_id": target.id,
            }
            for label, target in LEGACY_CATEGORY_CROSSWALK.items()
        ],
    )
    backfill_legacy_categories(bind, commit_batch=_commit_migration_batch)
    op.drop_table("legacy_category_crosswalk")


def downgrade() -> None:
    """The backfill is intentionally non-destructive and has no schema to remove."""


def backfill_legacy_categories(
    bind: sa.Connection,
    *,
    batch_size: int = _BATCH_SIZE,
    commit_batch: Callable[[Callable[[], None]], None] | None = None,
) -> None:
    """Backfill mapped rows while retaining unmapped legacy categories unchanged.

    ``commit_batch`` is supplied by Alembic upgrade so each completed batch is durable.
    Tests and repair tooling can omit it to run inside their caller's transaction.
    Rows with an existing topic assignment are deliberately excluded, making a rerun
    safe after an interrupted migration.
    """
    targets = {
        label: (target.id, target.pref_label) for label, target in LEGACY_CATEGORY_CROSSWALK.items()
    }
    for item_table in ("feed_items", "topic_entities"):
        _backfill_table(
            bind,
            item_table=item_table,
            targets=targets,
            batch_size=batch_size,
            commit_batch=commit_batch,
        )


def _seed_concepts(bind: sa.Connection) -> None:
    existing_ids = set(bind.execute(sa.select(_concepts.c.id)).scalars())
    rows = [
        {
            "id": concept.id,
            "facet": "topic",
            "pref_label": concept.pref_label,
            "notation": None,
            "scheme": TOPIC_SPINE_SCHEME,
            "scheme_version": "1",
            "source_uri": str(concept.uri),
            "status": "active",
        }
        for concept in TOPIC_CONCEPTS
        if concept.id not in existing_ids
    ]
    if rows:
        bind.execute(_concepts.insert(), rows)


def _backfill_table(
    bind: sa.Connection,
    *,
    item_table: str,
    targets: dict[str, tuple[int, str]],
    batch_size: int,
    commit_batch: Callable[[Callable[[], None]], None] | None,
) -> None:
    table = sa.table(item_table, sa.column("id", sa.Integer()), sa.column("category", sa.Text()))
    last_id = 0
    warned_labels: set[str] = set()
    while True:
        existing_topic = sa.exists(
            sa.select(sa.literal(1)).where(
                _item_concepts.c.item_table == item_table,
                _item_concepts.c.item_id == table.c.id,
                _item_concepts.c.facet == "topic",
            )
        )
        rows = bind.execute(
            sa.select(table.c.id, table.c.category)
            .where(table.c.id > last_id, table.c.category.is_not(None), ~existing_topic)
            .order_by(table.c.id)
            .limit(batch_size)
        ).all()
        if not rows:
            return
        last_id = rows[-1].id
        assignments = []
        updates = []
        unmapped = set()
        for row in rows:
            target = targets.get(row.category)
            if target is None:
                unmapped.add(row.category)
                continue
            concept_id, top_level_label = target
            assignments.append(
                {
                    "item_table": item_table,
                    "item_id": row.id,
                    "concept_id": concept_id,
                    "facet": "topic",
                    "confidence": 1.0,
                    "method": "rule",
                }
            )
            updates.append({"row_id": row.id, "category": top_level_label})

        # _commit_migration_batch() always invokes write_batch() synchronously, before
        # assignments/updates could ever be rebound by a later loop iteration.
        def write_batch() -> None:
            if assignments:  # noqa: B023
                bind.execute(_item_concepts.insert(), assignments)  # noqa: B023
                bind.execute(
                    table.update().where(table.c.id == sa.bindparam("row_id")),
                    updates,  # noqa: B023
                )

        if assignments:
            if commit_batch is None:
                write_batch()
            else:
                commit_batch(write_batch)
        new_unmapped = unmapped - warned_labels
        if new_unmapped:
            warnings.warn(
                f"legacy category crosswalk has no mapping for {item_table}: "
                f"{', '.join(sorted(new_unmapped))}",
                UserWarning,
                stacklevel=2,
            )
            warned_labels.update(new_unmapped)


def _commit_migration_batch(write_batch: Callable[[], None]) -> None:
    """Commit one backfill batch without sacrificing Alembic's migration context."""
    with op.get_context().autocommit_block():
        write_batch()
