"""Self-contained persistence layer (design D-1 / D-7 / D-13).

A single SQLAlchemy engine singleton uses an explicit DSN: ``RUMEN_DB_DSN`` takes
precedence, and a pipeline configuration may provide ``database.dsn`` for an
entrypoint to apply before opening the engine. SQLite is an explicit local
option, never a fallback; absent configuration raises a clear error. Schema is
versioned with Alembic. The engine owns its storage and migration state.
"""

from rumen.db.engine import db_url, get_engine, reset_engine

__all__ = ["db_url", "get_engine", "reset_engine"]
