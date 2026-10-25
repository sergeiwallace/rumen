"""rumen — self-contained rumen engine.

Ingest → classify → keyword-tag → group (dedup by identity) → rank → digest.
Storage is self-contained (own SQLAlchemy models + SQLite/Postgres) behind the
``CurationStore`` port (design D-13); additional backends may be provided by optional adapters.
"""

__version__ = "0.1.0"
