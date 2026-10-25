"""Alembic migration environment: dual-dialect (Postgres + SQLite).

The URL comes from the same resolver the engine uses (an explicit ``RUMEN_DB_DSN``), so migrations and the runtime engine always target the same database.
Tests inject a live connection via ``config.attributes['connection']``.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from rumen.db.engine import db_schema, db_url, get_engine

config = context.config

# Keep the version table in the configured schema so independent applications
# can share a Postgres instance without sharing migration state. Pin the schema
# explicitly rather than relying on search_path. On SQLite db_schema() is None,
# which selects Alembic's default behavior.

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate. Imported lazily/guarded so this env module
# loads even before models.py exists (T-01 lands before T-02).
try:
    from rumen.models import Base

    target_metadata = Base.metadata
except ImportError:
    target_metadata = None


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url") or db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        version_table_schema=db_schema(),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Support an injected connection (from tests / programmatic migrate).
    injected = config.attributes.get("connection")
    if injected is not None:
        context.configure(
            connection=injected,
            target_metadata=target_metadata,
            render_as_batch=True,
            version_table_schema=db_schema(),
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    # Reuse the runtime engine factory so the SQLite parent dir is created and
    # the FK pragma is applied, exactly as at runtime.
    engine = get_engine()
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            version_table_schema=db_schema(),
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
