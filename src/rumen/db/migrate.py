"""Programmatic Alembic runner (design T-01).

Lets the engine, jobs, and tests apply migrations against a live connection
without shelling out to the ``alembic`` CLI. Dialect-agnostic — the injected
connection carries the dialect.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import Engine

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _config(connection, *, backlog_target_identity: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.attributes["connection"] = connection
    if backlog_target_identity is not None:
        cfg.attributes["backlog_target_identity"] = backlog_target_identity
    return cfg


def upgrade_to_head(engine: Engine, *, backlog_target_identity: str | None = None) -> None:
    """Apply all migrations up to ``head``.

    Migrations own their transaction boundaries so a data backfill can make durable
    progress batch-by-batch (D-14) instead of being trapped in this helper's outer
    transaction.
    """
    with engine.connect() as connection:
        command.upgrade(
            _config(connection, backlog_target_identity=backlog_target_identity), "head"
        )
    if backlog_target_identity:
        # Alembic executes a revision only once.  Repeating this idempotent post-upgrade
        # completion step lets a later configured invocation repair a prior config-less
        # upgrade without inventing per-row provenance for Phase-1 candidates.
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE backlog_candidates "
                    "SET target_identity = :target_identity "
                    "WHERE target_identity IS NULL"
                ),
                {"target_identity": backlog_target_identity},
            )


def downgrade_to_base(engine: Engine) -> None:
    """Revert all migrations back to the empty schema."""
    with engine.begin() as connection:
        command.downgrade(_config(connection), "base")
