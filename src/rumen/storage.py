"""Storage-backend preflight for standalone CLI and job entrypoints.

The SQLAlchemy engine remains the only implemented backend. Unsupported backend
selections fail before a database connection is opened.
"""

from __future__ import annotations

import os

from sqlalchemy.engine import Engine

from rumen.config import PipelineConfig
from rumen.db.engine import get_engine


def preflight_storage(config: PipelineConfig) -> None:
    """Reject unavailable storage selections before a caller opens a store."""
    if config.storage.backend != "sqlalchemy":
        raise ValueError("storage.backend must be 'sqlalchemy'")


def resolve_database_dsn(config: PipelineConfig) -> str | None:
    """Resolve the documented DSN precedence: environment, then config, then error later."""
    return os.environ.get("RUMEN_DB_DSN") or config.database.dsn


def open_storage(config: PipelineConfig) -> Engine:
    """Preflight and open the configured SQLAlchemy engine.

    ``RUMEN_DB_DSN`` wins over ``database.dsn``.  The engine itself retains the
    explicit-DSN error when neither source is configured.
    """
    preflight_storage(config)
    if not os.environ.get("RUMEN_DB_DSN") and config.database.dsn is not None:
        os.environ["RUMEN_DB_DSN"] = config.database.dsn
    return get_engine()
