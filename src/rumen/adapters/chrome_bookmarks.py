"""Chrome Bookmarks JSON adapter."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from rumen.adapters.base import RawItem

logger = logging.getLogger(__name__)

_CHROME_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)


class ChromeBookmarksError(RuntimeError):
    """Configured Chrome bookmarks file cannot be parsed."""


class ChromeBookmarksAdapter:
    """Read one explicitly configured Chrome profile's Bookmarks JSON file.

    When ``folder_name`` is set, only bookmarks below the first matching folder
    are read. Matches use a stable depth-first traversal of roots and children;
    duplicate folder names therefore select the first match.
    """

    def __init__(self, bookmarks_path: str, profile: str, folder_name: str | None = None) -> None:
        self.bookmarks_path = bookmarks_path
        self.profile = profile
        self.folder_name = folder_name

    def source_name(self) -> str:
        return f"chrome_bookmarks:{self.profile}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        roots = self._load_roots()
        items: list[RawItem] = []
        nodes = self._scoped_nodes(roots)
        for node in nodes:
            if node.get("type") != "url":
                continue
            item = self._map_node(node)
            if item is not None and (since is None or item.published_at > since):
                items.append(item)
        return items

    def _scoped_nodes(self, roots: dict):
        if self.folder_name is None:
            for root in roots.values():
                yield from self._walk(root)
            return

        wanted_name = self.folder_name.casefold()
        for root in roots.values():
            for node in self._walk(root):
                name = node.get("name")
                if (
                    node.get("type") == "folder"
                    and isinstance(name, str)
                    and name.casefold() == wanted_name
                ):
                    yield from self._walk(node)
                    return

        # A configured scope that is absent is a configuration error, unlike a malformed bookmark.
        raise ChromeBookmarksError(f"configured folder {self.folder_name!r} was not found")

    def _load_roots(self) -> dict:
        try:
            payload = json.loads(Path(self.bookmarks_path).read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChromeBookmarksError(
                f"cannot read Chrome bookmarks file {self.bookmarks_path!r}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("roots"), dict):
            raise ChromeBookmarksError(
                "Chrome bookmarks file has an invalid top-level roots structure"
            )
        return payload["roots"]

    def _walk(self, node: object):
        if not isinstance(node, dict):
            return
        yield node
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                yield from self._walk(child)

    def _map_node(self, node: dict) -> RawItem | None:
        guid = node.get("guid")
        url = node.get("url")
        date_added = self._parse_date_added(node.get("date_added"))
        try:
            valid_guid = isinstance(guid, str) and bool(guid) and str(UUID(guid)) == guid.lower()
        except (TypeError, ValueError, AttributeError):
            valid_guid = False
        if not valid_guid or not isinstance(url, str) or not url.strip() or date_added is None:
            logger.warning("chrome_bookmarks: skipping malformed bookmark node")
            return None
        name = node.get("name")
        return RawItem(
            source=self.source_name(),
            source_id=guid,
            url=url,
            title=name if isinstance(name, str) else "",
            published_at=date_added,
        )

    def _parse_date_added(self, value: object) -> datetime | None:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None
        try:
            return _CHROME_EPOCH + timedelta(microseconds=int(value))
        except (OverflowError, TypeError, ValueError):
            return None
