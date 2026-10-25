"""Shared pytest fixtures for rumen.

Isolates the engine singleton per test and provides an optional Postgres engine
(active only when ``RUMEN_TEST_PG_DSN`` points at a reachable database), plus a
helper to run Alembic migrations against an arbitrary connection.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from rumen import embeddings as embeddings_mod
from rumen.db import engine as engine_mod
from rumen.db.migrate import downgrade_to_base, upgrade_to_head

__all__ = ["downgrade_to_base", "upgrade_to_head"]


@pytest.fixture(autouse=True)
def _isolate_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Reset the engine singleton and point the (opt-in) SQLite DSN at a temp file so
    tests never touch a real database. Postgres tests override ``RUMEN_DB_DSN`` themselves.

    Also pin the deterministic hashing embedder as the default provider — the suite must
    be deterministic + offline regardless of whether the MiniLM ``embeddings`` extra is
    installed (production auto-selects MiniLM via ``get_default_provider``)."""
    engine_mod.reset_engine()
    monkeypatch.delenv("RUMEN_DB_SCHEMA", raising=False)
    monkeypatch.setenv("RUMEN_DB_DSN", f"sqlite:///{tmp_path / 'rumen_home' / 'curation.db'}")
    embeddings_mod.set_default_provider(embeddings_mod.HashingEmbeddingProvider())
    yield
    embeddings_mod.set_default_provider(None)
    engine_mod.reset_engine()


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    """A fresh SQLite engine (with the FK pragma) on the isolated RUMEN_HOME."""
    engine_mod.reset_engine()
    engine = engine_mod.get_engine()
    assert engine.dialect.name == "sqlite"
    yield engine


@pytest.fixture
def pg_dsn() -> str:
    dsn = os.environ.get("RUMEN_TEST_PG_DSN")
    if not dsn:
        pytest.skip("RUMEN_TEST_PG_DSN not set — Postgres-dialect test skipped")
    return dsn


@pytest.fixture
def pg_engine(pg_dsn: str) -> Iterator[Engine]:
    """A Postgres engine with a clean schema per test (drop+recreate ``public``)."""
    engine = create_engine(engine_mod.normalize_pg_driver(pg_dsn))
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    yield engine
    engine.dispose()


@pytest.fixture(params=["sqlite", "postgres"])
def migrated_engine(request: pytest.FixtureRequest) -> Iterator[Engine]:
    """A migrated (``alembic upgrade head``) engine, run once per dialect.

    Postgres is skipped when ``RUMEN_TEST_PG_DSN`` is unset, so the SQLite variant
    always runs and CI without Postgres degrades gracefully.
    """
    if request.param == "sqlite":
        engine_mod.reset_engine()
        engine = engine_mod.get_engine()
        assert engine.dialect.name == "sqlite"
        upgrade_to_head(engine)
        yield engine
        return

    dsn = os.environ.get("RUMEN_TEST_PG_DSN")
    if not dsn:
        pytest.skip("RUMEN_TEST_PG_DSN not set — Postgres-dialect test skipped")
    engine = create_engine(engine_mod.normalize_pg_driver(dsn))
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    upgrade_to_head(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(migrated_engine: Engine) -> Iterator[Session]:
    """An ORM session on the migrated engine (both dialects via ``migrated_engine``)."""
    with Session(migrated_engine) as s:
        yield s
