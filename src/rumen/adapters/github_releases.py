"""GitHub Releases adapter — polls a repo's ``releases.atom`` feed (design Stage 1 / T-03)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from xml.etree import ElementTree

from rumen.net_safety import open_public_request

_ATOM = "{http://www.w3.org/2005/Atom}"

Fetcher = Callable[[str], str]


def _http_get(url: str) -> str:
    with open_public_request(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


class GitHubReleasesAdapter:
    """One adapter instance per tracked repo (``owner/repo``)."""

    def __init__(self, repo: str, *, fetcher: Fetcher | None = None) -> None:
        self.repo = repo
        self._fetcher = fetcher or _http_get

    def source_name(self) -> str:
        # Per-repo identity, NOT the bare type: source_name is the high-water-mark cursor key
        # AND the dedup namespace, so every tracked repo must be distinct. A shared
        # "github_releases" would make all repos share one cursor (dropping later repos'
        # releases) and collide on identical tags (e.g. two repos shipping "v1.0.0").
        return f"github_releases:{self.repo}"

    def feed_url(self) -> str:
        return f"https://github.com/{self.repo}/releases.atom"

    def fetch_new(self, since: datetime | None):
        from rumen.adapters.base import RawItem

        root = ElementTree.fromstring(self._fetcher(self.feed_url()))
        items: list[RawItem] = []
        for entry in root.findall(f"{_ATOM}entry"):
            updated_text = entry.findtext(f"{_ATOM}updated")
            if not updated_text:
                continue
            published_at = datetime.fromisoformat(updated_text)
            if since is not None and published_at <= since:
                continue
            tag = (entry.findtext(f"{_ATOM}title") or "").strip()
            link_el = entry.find(f"{_ATOM}link")
            url = link_el.get("href") if link_el is not None else self.feed_url()
            content = entry.findtext(f"{_ATOM}content")
            author = None
            author_el = entry.find(f"{_ATOM}author")
            if author_el is not None:
                author = author_el.findtext(f"{_ATOM}name")
            items.append(
                RawItem(
                    source=self.source_name(),
                    source_id=tag,
                    url=url,
                    title=f"{self.repo} {tag}".strip(),
                    published_at=published_at,
                    content=content,
                    author=author,
                )
            )
        return items
