"""T-02: curation schema — 12 models + initial migration. ACs from the Task Breakdown.

Every test runs against both dialects via the ``session`` / ``migrated_engine``
fixtures (SQLite always; Postgres when ``RUMEN_TEST_PG_DSN`` is set)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alembic import command
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from rumen.db import migrate
from rumen.db.migrate import downgrade_to_base
from rumen.models import (
    Digest,
    DigestEntity,
    DroppedItem,
    FeedFeedback,
    FeedItem,
    FeedRun,
    FeedView,
    MergeCandidate,
    MergeLog,
    StableIdAlias,
    TopicEntity,
)

_ALL_TABLES = {
    "feed_items",
    "topic_entities",
    "digest_entities",
    "feed_keywords",
    "feed_views",
    "digests",
    "feed_feedback",
    "feed_runs",
    "merge_log",
    "stable_id_alias",
    "merge_candidates",
    "dropped_items",
}

_NAMED_INDEXES = {
    "idx_feed_items_stable_id",
    "idx_feed_items_canonical_url",
    "idx_topic_entities_stable_id",
    "idx_merge_candidates_status",
    "idx_dropped_items_review",
    "idx_digest_entities_digest",
}


def _now() -> datetime:
    return datetime.now(UTC)


def _entity(session, **overrides) -> TopicEntity:
    e = TopicEntity(
        canonical_title=overrides.get("canonical_title", "Example"),
        canonical_url=overrides.get("canonical_url", "https://example.com"),
        earliest_seen=overrides.get("earliest_seen", _now()),
        **{
            k: v
            for k, v in overrides.items()
            if k not in {"canonical_title", "canonical_url", "earliest_seen"}
        },
    )
    session.add(e)
    session.flush()
    return e


def test_migration_creates_all_twelve_tables_and_named_indexes(migrated_engine) -> None:
    insp = inspect(migrated_engine)
    tables = set(insp.get_table_names())
    assert _ALL_TABLES <= tables, f"missing tables: {_ALL_TABLES - tables}"

    all_indexes = set()
    for table in _ALL_TABLES:
        all_indexes.update(ix["name"] for ix in insp.get_indexes(table))
    assert _NAMED_INDEXES <= all_indexes, f"missing indexes: {_NAMED_INDEXES - all_indexes}"


def test_downgrade_drops_all_tables(migrated_engine) -> None:
    downgrade_to_base(migrated_engine)
    insp = inspect(migrated_engine)
    remaining = set(insp.get_table_names()) & _ALL_TABLES
    assert remaining == set(), f"tables not dropped: {remaining}"


def test_feed_item_status_migration_round_trip(migrated_engine) -> None:
    with migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO feed_items "
                "(source, source_id, url, title, published_at, ingested_at) "
                "VALUES ('migration', 'status-default', 'https://example.com', 'item', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        assert (
            connection.execute(
                text("SELECT status FROM feed_items WHERE source_id = 'status-default'")
            ).scalar_one()
            == "active"
        )

    assert "status" in {
        column["name"] for column in inspect(migrated_engine).get_columns("feed_items")
    }

    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0002")

    assert "status" not in {
        column["name"] for column in inspect(migrated_engine).get_columns("feed_items")
    }


def test_facets_classified_marker_migration_stamps_only_previously_classified_rows(
    migrated_engine,
) -> None:
    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0012")
        connection.execute(
            text(
                "INSERT INTO feed_items "
                "(source, source_id, url, title, published_at, ingested_at, classification_method) "
                "VALUES ('migration', 'classified', 'https://example.com/one', 'one', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'rule'), "
                "('migration', 'stale', 'https://example.com/two', 'two', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)"
            )
        )
        command.upgrade(migrate._config(connection), "head")
        rows = connection.execute(
            text(
                "SELECT source_id, facets_classified_at FROM feed_items "
                "WHERE source = 'migration' ORDER BY source_id"
            )
        ).all()

    assert rows[0].facets_classified_at is not None
    assert rows[1].facets_classified_at is None
    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0012")
    assert "facets_classified_at" not in {
        column["name"] for column in inspect(migrated_engine).get_columns("feed_items")
    }


def test_topic_entity_rank_position_migration_round_trip(migrated_engine) -> None:
    columns_at_head = {
        column["name"] for column in inspect(migrated_engine).get_columns("topic_entities")
    }
    assert {"rank_position", "is_exploration"} <= columns_at_head
    assert "idx_topic_entities_rank_position" in {
        index["name"] for index in inspect(migrated_engine).get_indexes("topic_entities")
    }
    with migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO topic_entities "
                "(canonical_title, canonical_url, earliest_seen, source_count, created_at, updated_at) "
                "VALUES ('migration-rank', 'https://example.com/rank', CURRENT_TIMESTAMP, 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        assert connection.execute(
            text(
                "SELECT rank_position, is_exploration FROM topic_entities "
                "WHERE canonical_title = 'migration-rank'"
            )
        ).one() == (None, 0)

    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0003")

    columns_after_downgrade = {
        column["name"] for column in inspect(migrated_engine).get_columns("topic_entities")
    }
    assert columns_after_downgrade == columns_at_head - {
        "display_title",
        "rank_position",
        "is_exploration",
        "tags",
    }
    assert "idx_topic_entities_rank_position" not in {
        index["name"] for index in inspect(migrated_engine).get_indexes("topic_entities")
    }


def test_server_defaults_migration_round_trip(migrated_engine) -> None:
    with migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO feed_runs (source, started_at) VALUES ('defaults', CURRENT_TIMESTAMP)"
            )
        )
        assert connection.execute(
            text(
                "SELECT items_fetched, items_passed_filter, status FROM feed_runs WHERE source='defaults'"
            )
        ).one() == (0, 0, "running")
    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0004")
    defaults = {
        column["name"]: column["default"]
        for column in inspect(migrated_engine).get_columns("feed_runs")
    }
    assert defaults["items_fetched"] is None


def test_facets_tags_migration_creates_and_drops_schema(migrated_engine) -> None:
    inspector = inspect(migrated_engine)
    assert "item_concepts" in inspector.get_table_names()
    assert {"tags"} <= {column["name"] for column in inspector.get_columns("feed_items")}
    assert {"tags"} <= {column["name"] for column in inspector.get_columns("topic_entities")}
    assert ("item_table", "item_id", "concept_id") in {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("item_concepts")
    }

    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0005")

    inspector = inspect(migrated_engine)
    assert "item_concepts" not in inspector.get_table_names()
    assert "tags" not in {column["name"] for column in inspector.get_columns("feed_items")}
    assert "tags" not in {column["name"] for column in inspector.get_columns("topic_entities")}


def test_donor_provenance_migration_creates_and_drops_columns(migrated_engine) -> None:
    inspector = inspect(migrated_engine)
    assert "source_uri" in {column["name"] for column in inspector.get_columns("skos_concepts")}
    assert {
        "source_scheme",
        "source_version",
        "target_scheme",
        "target_version",
        "reviewer",
        "review_date",
    } <= {column["name"] for column in inspector.get_columns("skos_relations")}

    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0006")

    inspector = inspect(migrated_engine)
    assert "source_uri" not in {column["name"] for column in inspector.get_columns("skos_concepts")}
    assert not {
        "source_scheme",
        "source_version",
        "target_scheme",
        "target_version",
        "reviewer",
        "review_date",
    } & {column["name"] for column in inspector.get_columns("skos_relations")}


def test_digest_entities_unique_and_rank_roundtrip(session) -> None:
    entity = _entity(session)
    digest = Digest(
        period_start=_now(),
        period_end=_now(),
        content_json={"active_tool_updates": []},
        content_markdown="# digest",
    )
    session.add(digest)
    session.flush()
    session.add(DigestEntity(digest_id=digest.id, topic_entity_id=entity.id, rank=3))
    session.commit()

    stored = session.query(DigestEntity).one()
    assert stored.rank == 3

    # UNIQUE(digest_id, topic_entity_id)
    session.add(DigestEntity(digest_id=digest.id, topic_entity_id=entity.id, rank=4))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_topic_entity_embedding_roundtrip_and_updated_at_autoset(session, migrated_engine) -> None:
    blob = b"\x00\x01\x02\x03" * 4
    entity = _entity(session, embedding=blob)
    session.commit()
    assert entity.created_at is not None
    assert entity.updated_at is not None
    first_updated = entity.updated_at

    reloaded = session.get(TopicEntity, entity.id)
    assert reloaded.embedding == blob

    # onupdate bumps updated_at on mutation.
    reloaded.source_count = 5
    session.commit()
    assert reloaded.updated_at >= first_updated


def test_feed_items_d8_d12_columns_and_unique_source(session) -> None:
    item = FeedItem(
        source="github_releases",
        source_id="v1.0.0",
        url="https://github.com/o/r/releases/tag/v1.0.0",
        canonical_url="https://github.com/o/r",
        stable_id_kind="github_repo",
        stable_id_value="o/r",
        title="Release",
        published_at=_now(),
        match_method="stable_id",
        match_confidence=1.0,
        classification_confidence=0.9,
        classification_method="rule",
    )
    session.add(item)
    session.commit()
    stored = session.query(FeedItem).one()
    assert stored.stable_id_value == "o/r"
    assert stored.classification_method == "rule"

    # UNIQUE(source, source_id)
    session.add(
        FeedItem(
            source="github_releases", source_id="v1.0.0", url="x", title="dup", published_at=_now()
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_feed_items_json_list_columns_roundtrip(session) -> None:
    item = FeedItem(
        source="bookmarks",
        source_id="t1",
        url="https://x.com/a/status/1",
        title="tweet",
        published_at=_now(),
        categories=["ai", "tooling"],
        domains=["github.com", "arxiv.org"],
        matched_keywords=["llm"],
        tags=["release", "open-source"],
    )
    session.add(item)
    session.commit()
    reloaded = session.get(FeedItem, item.id)
    assert reloaded.categories == ["ai", "tooling"]
    assert reloaded.domains == ["github.com", "arxiv.org"]
    assert reloaded.matched_keywords == ["llm"]
    assert reloaded.tags == ["release", "open-source"]


def test_topic_entity_tags_roundtrip(session) -> None:
    entity = _entity(session, tags=["ai", "research"])
    session.commit()

    assert session.get(TopicEntity, entity.id).tags == ["ai", "research"]


def test_feed_view_filter_json_roundtrip_and_is_default_defaults_false(session) -> None:
    view = FeedView(name="AI/ML", filter_json={"category": ["tool", "paper"]})
    session.add(view)
    session.commit()
    reloaded = session.get(FeedView, view.id)
    assert reloaded.filter_json == {"category": ["tool", "paper"]}
    assert reloaded.is_default in (0, False)


def test_merge_log_roundtrip(session) -> None:
    entity = _entity(session)
    item = FeedItem(
        source="hn",
        source_id="h1",
        url="https://news.ycombinator.com/item?id=1",
        title="post",
        published_at=_now(),
    )
    session.add(item)
    session.flush()
    session.add(
        MergeLog(
            topic_entity_id=entity.id,
            feed_item_id=item.id,
            match_method="stable_id",
            match_confidence=1.0,
            actor="auto",
            evidence={"stable_id": "o/r", "competitor": None},
        )
    )
    session.commit()
    stored = session.query(MergeLog).one()
    assert stored.action == "merge"
    assert stored.evidence == {"stable_id": "o/r", "competitor": None}


def test_stable_id_alias_roundtrip(session) -> None:
    session.add(
        StableIdAlias(
            kind="doi",
            from_value="10.preprint/x",
            to_value="10.published/x",
            reason="preprint_published",
        )
    )
    session.commit()
    stored = session.query(StableIdAlias).one()
    assert stored.to_value == "10.published/x"


def test_merge_candidates_defaults_pending(session) -> None:
    a = _entity(session, canonical_url="https://a.example")
    b = _entity(session, canonical_url="https://b.example")
    session.add(
        MergeCandidate(entity_a_id=a.id, entity_b_id=b.id, similarity=0.85, reason="embedding_band")
    )
    session.commit()
    stored = session.query(MergeCandidate).one()
    assert stored.status == "pending"


def test_dropped_items_roundtrip(session) -> None:
    session.add(
        DroppedItem(
            source="hf_daily",
            source_id="p1",
            url="https://arxiv.org/abs/2501.1",
            title="paper",
            drop_reason="classified_out_of_scope",
            drop_confidence=0.4,
        )
    )
    session.commit()
    stored = session.query(DroppedItem).one()
    assert stored.drop_reason == "classified_out_of_scope"
    assert stored.drop_confidence == 0.4
    assert stored.reviewed in (0, False)


def test_feed_run_roundtrip(session) -> None:
    session.add(
        FeedRun(
            source="github_releases",
            started_at=_now(),
            status="completed",
            high_water_mark="2026-07-16T00:00:00Z",
        )
    )
    session.commit()
    assert session.query(FeedRun).one().status == "completed"


def test_feed_run_source_state_migration_round_trip(migrated_engine) -> None:
    columns_at_head = {
        column["name"] for column in inspect(migrated_engine).get_columns("feed_runs")
    }
    assert {"source_state", "diagnostics"} <= columns_at_head
    with migrated_engine.begin() as connection:
        command.downgrade(migrate._config(connection), "0014")
    columns_after_downgrade = {
        column["name"] for column in inspect(migrated_engine).get_columns("feed_runs")
    }
    assert {"source_state", "diagnostics"}.isdisjoint(columns_after_downgrade)
    with migrated_engine.begin() as connection:
        command.upgrade(migrate._config(connection), "head")


def test_feed_feedback_fk_enforced(session) -> None:
    # Failure path: FK to a non-existent topic_entity_id must raise (SQLite pragma on).
    session.add(FeedFeedback(topic_entity_id=999999, rating=1))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_sqlite_foreign_keys_pragma_active_on_migrated_engine(migrated_engine) -> None:
    if migrated_engine.dialect.name != "sqlite":
        pytest.skip("SQLite-only pragma check")
    with migrated_engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
