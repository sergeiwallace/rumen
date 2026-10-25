"""npm registry adapter — polls ``registry.npmjs.org/<package>`` (design Stage 1 / T-03, AD-3).

Claude Code has no centralized changelog; the npm registry (``@anthropic-ai/claude-code``)
is its source of truth (R-49 §2). Each new version in the registry's ``time`` map becomes a
``RawItem``. Package → tool-name mapping is config, never hardcoded.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from datetime import datetime

from rumen.net_safety import open_public_request

Fetcher = Callable[[str], str]

# Non-version keys in the registry's ``time`` object.
_NON_VERSION_KEYS = frozenset({"created", "modified"})


def _http_get(url: str) -> str:
    with open_public_request(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


class NpmRegistryAdapter:
    """One adapter instance per tracked npm package."""

    def __init__(self, package: str, tool_name: str, *, fetcher: Fetcher | None = None) -> None:
        self.package = package
        self.tool_name = tool_name
        self._fetcher = fetcher or _http_get

    def source_name(self) -> str:
        # Per-package identity, NOT the bare type — same reason as the GitHub adapter: this is
        # the high-water-mark cursor key and dedup namespace, so each tracked package must be
        # distinct (a shared "npm_registry" would share one cursor across all packages and
        # collide on identical version tags).
        return f"npm_registry:{self.package}"

    def registry_url(self) -> str:
        # Registry paths keep the '@scope/name' unencoded except the '/'.
        return f"https://registry.npmjs.org/{urllib.parse.quote(self.package, safe='@/')}"

    def fetch_new(self, since: datetime | None):
        from rumen.adapters.base import RawItem

        doc = json.loads(self._fetcher(self.registry_url()))
        times: dict = doc.get("time", {})
        versions: dict = doc.get("versions", {})
        items: list[RawItem] = []
        for version, published_text in times.items():
            if version in _NON_VERSION_KEYS:
                continue
            published_at = datetime.fromisoformat(published_text)
            if since is not None and published_at <= since:
                continue
            description = None
            version_meta = versions.get(version)
            if isinstance(version_meta, dict):
                description = version_meta.get("description")
            items.append(
                RawItem(
                    source=self.source_name(),
                    source_id=version,
                    url=f"https://www.npmjs.com/package/{self.package}/v/{version}",
                    title=f"{self.tool_name} {version}",
                    published_at=published_at,
                    content=description,
                    metadata={"package": self.package, "tool_name": self.tool_name},
                )
            )
        return items
