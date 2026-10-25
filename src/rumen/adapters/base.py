"""Source-adapter protocol + the normalized ``RawItem`` (design Stage 1 / T-03)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class RawItem:
    """A normalized pointer emitted by a source adapter, before persistence.

    Maps onto the ``feed_items`` columns an adapter can populate at ingest. Grouping,
    classification, and embedding enrich the row later; adapters fill only what the
    source natively provides. Bookmarks arrive pre-classified (``category``/``categories``).
    """

    source: str
    source_id: str
    url: str
    title: str
    published_at: datetime
    content: str | None = None
    author: str | None = None
    category: str | None = None
    categories: list[str] | None = None
    domains: list[str] | None = None
    # Adapter-specific extras carried through onto the item (e.g. github_urls,
    # expanded_text_links) without a dedicated column.
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceChange:
    """One replayable mutation from an opt-in mutable source."""

    operation: str
    source_id: str
    occurred_at: datetime
    revision_digest: str
    item: RawItem | None = None

    def __post_init__(self) -> None:
        if self.operation not in {"upsert", "tombstone"}:
            raise ValueError("SourceChange operation must be 'upsert' or 'tombstone'")
        if not self.source_id or not self.revision_digest:
            raise ValueError("SourceChange source_id and revision_digest are required")
        if (self.operation == "upsert") != (self.item is not None):
            raise ValueError("upserts require an item and tombstones must not include one")


@dataclass(slots=True)
class FetchResult:
    """Items plus opaque, durable state returned by replay-oriented adapters."""

    items: list[RawItem]
    next_state: dict
    diagnostics: dict


class StatefulFetchError(RuntimeError):
    """An adapter failure whose safe diagnostic summary may be persisted."""

    def __init__(self, message: str, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@runtime_checkable
class SourceAdapter(Protocol):
    """Common contract for every ingestion source.

    ``fetch_new(since)`` returns candidates at or after ``since`` (all items when
    ``since`` is ``None``). Adapters whose upstream cursor is inclusive may replay the
    boundary cohort; ingestion deduplicates by ``(source, source_id)``. ``source_name()``
    is the stable ``feed_items.source`` tag.
    """

    def fetch_new(self, since: datetime | None) -> list[RawItem]: ...

    def source_name(self) -> str: ...


@runtime_checkable
class ChangeSourceAdapter(Protocol):
    """Capability protocol deliberately separate from append-only adapters."""

    def source_name(self) -> str: ...

    def fetch_changes(self, since: datetime | None) -> list[SourceChange]: ...


@runtime_checkable
class StatefulSourceAdapter(SourceAdapter, Protocol):
    """Optional capability for adapters that need completed-run state."""

    def fetch_new_with_state(self, since: datetime | None, state: dict | None) -> FetchResult: ...
