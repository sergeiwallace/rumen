"""T-14: documented legacy-label crosswalk and migration backfill coverage."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

import pytest
from alembic import command
from sqlalchemy import inspect, text

from rumen.db import migrate
from rumen.taxonomy.legacy_crosswalk import (
    ECONOMY,
    NEEDS_REVIEW,
    concept_for_legacy_label,
)

_migration = importlib.import_module(
    "rumen.db.migrations.versions.0008_legacy_label_crosswalk_backfill"
)


def test_documented_synonyms_and_junk_labels_resolve_to_reviewed_concepts() -> None:
    assert concept_for_legacy_label("econ").id == concept_for_legacy_label("economics").id
    assert concept_for_legacy_label("safe") == NEEDS_REVIEW
    assert concept_for_legacy_label("request clarification") == NEEDS_REVIEW


def test_legacy_crosswalk_migration_backfills_mapped_rows_and_keeps_unknown_rows(
    migrated_engine,
) -> None:
    # Seed at the pre-T-14 revision so this exercises migration data work, not only DDL.
    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0007")
    with migrated_engine.begin() as connection:
        _seed_legacy_rows(connection)

    with pytest.warns(UserWarning, match="mystery-label"), migrated_engine.connect() as connection:
        command.upgrade(migrate._config(connection), "head")

    with migrated_engine.connect() as connection:
        rows = connection.execute(
            text("SELECT source_id, category FROM feed_items ORDER BY source_id")
        ).all()
        assert rows == [
            ("econ", ECONOMY.pref_label),
            ("economics", ECONOMY.pref_label),
            ("safe", NEEDS_REVIEW.pref_label),
            ("unknown", "mystery-label"),
        ]
        assert (
            connection.execute(
                text("SELECT category FROM topic_entities WHERE canonical_title = 'legacy-ai'")
            ).scalar_one()
            == "Science and technology"
        )
        assignments = connection.execute(
            text(
                "SELECT item_table, item_id, concept_id, facet, confidence, method "
                "FROM item_concepts ORDER BY item_table, item_id"
            )
        ).all()
        assert len(assignments) == 4
        assert {
            (item_table, facet, confidence, method)
            for item_table, _item_id, _concept_id, facet, confidence, method in assignments
        } == {
            ("feed_items", "topic", 1.0, "rule"),
            ("topic_entities", "topic", 1.0, "rule"),
        }
        econ_concepts = (
            connection.execute(
                text(
                    "SELECT concept_id FROM item_concepts "
                    "WHERE item_table = 'feed_items' AND item_id IN "
                    "(SELECT id FROM feed_items WHERE source_id IN ('econ', 'economics'))"
                )
            )
            .scalars()
            .all()
        )
        assert econ_concepts == [ECONOMY.id, ECONOMY.id]
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM skos_concepts WHERE id = :id"), {"id": NEEDS_REVIEW.id}
            ).scalar_one()
            == 1
        )
        assert "legacy_category_crosswalk" not in inspect(connection).get_table_names()

    # A rerun after partial work must leave rows with topic assignments untouched.
    with migrated_engine.begin() as connection:
        connection.execute(
            text("UPDATE feed_items SET category = 'already-backfilled' WHERE source_id = 'econ'")
        )
        _migration.backfill_legacy_categories(connection, batch_size=1)
        assert (
            connection.execute(
                text("SELECT category FROM feed_items WHERE source_id = 'econ'")
            ).scalar_one()
            == "already-backfilled"
        )
        assert (
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM item_concepts WHERE item_table = 'feed_items' "
                    "AND item_id = (SELECT id FROM feed_items WHERE source_id = 'econ') AND facet = 'topic'"
                )
            ).scalar_one()
            == 1
        )

    # ``downgrade -1`` leaves the non-destructive data backfill in place; upgrading
    # again must therefore recognize existing topic assignments and leave them unique.
    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0007")
    with pytest.warns(UserWarning, match="mystery-label"), migrated_engine.connect() as connection:
        command.upgrade(migrate._config(connection), "head")
    with migrated_engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM item_concepts")).scalar_one() == 4


def _seed_legacy_rows(connection) -> None:
    now = datetime.now(UTC)
    for source_id, category in (
        ("econ", "econ"),
        ("economics", "economics"),
        ("safe", "safe"),
        ("unknown", "mystery-label"),
    ):
        connection.execute(
            text(
                "INSERT INTO feed_items "
                "(source, source_id, url, title, published_at, ingested_at, category) "
                "VALUES ('legacy', :source_id, :url, :title, :now, :now, :category)"
            ),
            {
                "source_id": source_id,
                "url": f"https://example.test/{source_id}",
                "title": source_id,
                "now": now,
                "category": category,
            },
        )
    connection.execute(
        text(
            "INSERT INTO topic_entities "
            "(canonical_title, canonical_url, earliest_seen, category) "
            "VALUES ('legacy-ai', 'https://example.test/ai', :now, 'ai-news')"
        ),
        {"now": now},
    )
