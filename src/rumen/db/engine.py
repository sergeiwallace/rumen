"""Shared SQLAlchemy engine for rumen (design D-7 / T-01).

Single engine singleton used by the models, the store backend, and Alembic
migrations. The DSN is **explicit and required** — there is no implicit SQLite
fallback (decided 2026-07-16):

- ``RUMEN_DB_DSN`` is required. Use a Postgres URL or explicitly opt into SQLite
  with ``RUMEN_DB_DSN=sqlite:///path``. No SQLite file is created without a DSN.
- ``RUMEN_DB_SCHEMA`` (Postgres only) isolates tables in a named schema inside a
  shared instance. The schema is created if absent and ``search_path`` is pinned.

An unreachable Postgres DSN raises ``OperationalError`` on first connect — it does
NOT silently fall back (a silent fallback would mask a misconfigured database).
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

_engine: Engine | None = None


@contextmanager
def batched_write_mode(session: Session):
    """Keep loaded objects live across mid-loop commits during a batched-write stage.

    With the default ``expire_on_commit=True``, every batched ``session.commit()`` expires
    all loaded ORM objects, so the *next* attribute access re-fetches that row one at a time
    — a catastrophic N+1 over a remote DB (measured ~150 ms/row over a VPN link, which turned
    a 2-second stage into ~350 seconds). Disabling expiry (and autoflush, so writes batch at
    the commit boundaries we choose) collapses it back to a couple of bulk round-trips. Every
    batched-commit pipeline stage (grouping, keyword_filter, classify, rank) runs inside this.
    Restores the session's prior flags on exit."""
    prior_expire = session.expire_on_commit
    prior_autoflush = session.autoflush
    session.expire_on_commit = False
    session.autoflush = False
    try:
        yield
    finally:
        session.expire_on_commit = prior_expire
        session.autoflush = prior_autoflush


_SQLITE_PREFIX = "sqlite:///"
_VALID_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_NO_DSN_MESSAGE = (
    "RUMEN_DB_DSN is not set. rumen requires an explicit database DSN. "
    "Point it at your Postgres instance, e.g. RUMEN_DB_DSN=postgresql://host/db with "
    "RUMEN_DB_SCHEMA=curation. For a zero-dependency local trial (the open-source "
    "on-ramp) set an explicit SQLite DSN: RUMEN_DB_DSN=sqlite:///rumen.db. There is no "
    "implicit SQLite fallback."
)


def normalize_pg_driver(dsn: str) -> str:
    """Pin the psycopg3 driver (``psycopg[binary]``, our declared dep).

    SQLAlchemy defaults a bare ``postgresql://`` URL to psycopg2, which we do not
    install. Rewrite it to ``postgresql+psycopg://`` unless a driver is already named.
    """
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://") :]
    if dsn.startswith("postgres://"):
        return "postgresql+psycopg://" + dsn[len("postgres://") :]
    return dsn


def db_url() -> str:
    """Return the configured DB URL. Raises ``RuntimeError`` if ``RUMEN_DB_DSN`` is unset."""
    dsn = os.environ.get("RUMEN_DB_DSN")
    if not dsn:
        raise RuntimeError(_NO_DSN_MESSAGE)
    return normalize_pg_driver(dsn)


def db_schema() -> str | None:
    """Return the configured Postgres schema (``RUMEN_DB_SCHEMA``), or ``None``.

    Validated as a bare SQL identifier so it can be safely interpolated into the
    ``search_path`` / ``CREATE SCHEMA`` statements.
    """
    schema = os.environ.get("RUMEN_DB_SCHEMA")
    if not schema:
        return None
    if not _VALID_IDENT.match(schema):
        raise ValueError(f"RUMEN_DB_SCHEMA must be a simple SQL identifier; got {schema!r}")
    return schema


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite ships with FK enforcement OFF per-connection — turn it on for every
    connection so ``REFERENCES`` constraints raise ``IntegrityError`` (design T-02)."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _enable_sqlite_transactional_savepoints(engine: Engine) -> None:
    """Let SQLAlchemy control SQLite transaction boundaries and SAVEPOINTs.

    pysqlite's transaction emulation can otherwise start a SAVEPOINT before the
    outer transaction, making its release durable rather than nested in the
    eventual batch rollback.
    """

    @event.listens_for(engine, "connect")
    def _disable_pysqlite_transaction_emulation(dbapi_connection, _connection_record):
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _emit_sqlite_begin(connection):
        connection.exec_driver_sql("BEGIN")


def _ensure_pg_schema(engine: Engine, schema: str) -> None:
    """Create the configured schema only when it is absent.

    A restricted role may own a pre-provisioned schema without database-level
    CREATE privileges. Even ``CREATE SCHEMA IF NOT EXISTS`` checks those privileges,
    so query for the schema first. The connection pins ``search_path`` separately.
    """
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :s"),
            {"s": schema},
        ).scalar()
        if exists is None:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            conn.commit()


def _connection_config(
    is_sqlite: bool, schema: str | None
) -> tuple[dict[str, object], dict[str, object]]:
    """Build ``(connect_args, engine_kwargs)`` for ``create_engine``.

    SQLite needs neither. For Postgres, guard against stale pooled connections
    and half-open network connections:

    - ``pool_pre_ping`` validates a pooled connection before reuse (catches a stale one).
    - TCP **keepalives** make the OS probe a live-but-in-use socket and surface a dead link as
      an error instead of an infinite hang.
    - ``statement_timeout`` caps any single runaway statement.

    A broken connection then fails fast and — because every stage commits in batches and is
    idempotent (D-14) — a re-run resumes cleanly. ``search_path`` is pinned via the same libpq
    startup option (a transactional ``SET`` would be undone by the pool's ROLLBACK-on-return).
    Overridable via ``RUMEN_DB_KEEPALIVE_IDLE`` / ``RUMEN_DB_STATEMENT_TIMEOUT_MS`` (``0`` disables
    the timeout).
    """
    if is_sqlite:
        return {}, {}
    connect_args: dict[str, object] = {
        "keepalives": 1,
        "keepalives_idle": int(os.environ.get("RUMEN_DB_KEEPALIVE_IDLE", "30")),
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
    options = []
    if schema:
        options.append(f"-csearch_path={schema}")
    stmt_timeout_ms = os.environ.get("RUMEN_DB_STATEMENT_TIMEOUT_MS", "300000")
    if stmt_timeout_ms and stmt_timeout_ms != "0":
        options.append(f"-cstatement_timeout={stmt_timeout_ms}")
    if options:
        connect_args["options"] = " ".join(options)
    return connect_args, {"pool_pre_ping": True}


def get_engine() -> Engine:
    """Return the shared engine singleton, creating it on first call.

    For an explicit ``sqlite://`` DSN the parent directory is created and FK
    enforcement enabled. For Postgres, ``RUMEN_DB_SCHEMA`` (if set) is provisioned and
    pinned on the ``search_path``.
    """
    global _engine
    if _engine is not None:
        return _engine
    url = db_url()
    is_sqlite = url.startswith(_SQLITE_PREFIX)
    if is_sqlite:
        path = url[len(_SQLITE_PREFIX) :]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    schema = None if is_sqlite else db_schema()
    connect_args, engine_kwargs = _connection_config(is_sqlite, schema)
    engine = create_engine(url, connect_args=connect_args, **engine_kwargs)
    if is_sqlite:
        _enable_sqlite_foreign_keys(engine)
        _enable_sqlite_transactional_savepoints(engine)
    elif schema:
        _ensure_pg_schema(engine, schema)
    _engine = engine
    return _engine


def reset_engine() -> None:
    """Dispose and reset the engine singleton (for tests / DSN changes)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
