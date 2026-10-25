"""Hacker News Algolia adapter (design Stage 1 / T-03, T-32).

Uses the public ``search_by_date`` endpoint for configured keyword profiles.  ``points`` and
``num_comments`` are discovery-time snapshots: ingest deliberately does not update an existing
item's metadata on later runs.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from rumen.adapters.base import RawItem
from rumen.net_safety import open_public_request

Fetcher = Callable[[str], str]

_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
_TAGS = "story"
_HITS_PER_PAGE = 100


class HackerNewsFetchError(RuntimeError):
    """The provider returned a malformed or incomplete search window."""


def _http_get(url: str) -> str:
    with open_public_request(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class HackerNewsAdapter:
    """Fetch one atomic, keyword-defined HN profile through Algolia."""

    def __init__(
        self,
        profile: str,
        queries: list[str],
        *,
        bootstrap_lookback_hours: int = 72,
        indexing_overlap_minutes: int = 15,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.profile = profile
        self.queries = queries
        self.bootstrap_lookback_hours = bootstrap_lookback_hours
        self.indexing_overlap_minutes = indexing_overlap_minutes
        self._fetcher = fetcher or _http_get

    def source_name(self) -> str:
        return f"hn:{self.profile}"

    def search_url(
        self, query: str, lower_bound: datetime, upper_bound: datetime, page: int
    ) -> str:
        numeric_filters = (
            f"created_at_i>{int(lower_bound.timestamp())},"
            f"created_at_i<={int(upper_bound.timestamp())}"
        )
        params = urllib.parse.urlencode(
            {
                "query": query,
                "tags": _TAGS,
                "numericFilters": numeric_filters,
                "hitsPerPage": _HITS_PER_PAGE,
                "page": page,
            }
        )
        return f"{_SEARCH_URL}?{params}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        # Both boundaries are fixed before the first request, holding the offset-paginated
        # result set stable when Algolia honors the server-side numeric filters.
        upper_bound = _utcnow()
        lower_bound = (
            upper_bound - timedelta(hours=self.bootstrap_lookback_hours)
            if since is None
            else since - timedelta(minutes=self.indexing_overlap_minutes)
        )

        hits: list[dict] = []
        for query in self.queries:
            hits.extend(self._fetch_query(query, lower_bound, upper_bound))

        seen: set[str] = set()
        items: list[RawItem] = []
        for hit in hits:
            item = self._map_hit(hit)
            if item.published_at <= lower_bound or item.published_at > upper_bound:
                continue
            if item.source_id in seen:
                continue
            seen.add(item.source_id)
            items.append(item)
        return items

    def _fetch_query(self, query: str, lower_bound: datetime, upper_bound: datetime) -> list[dict]:
        first = self._load_page(query, lower_bound, upper_bound, 0)
        page_count = first["nbPages"]
        if page_count == 0:
            return []

        hits = list(first["hits"])
        for page in range(1, page_count):
            envelope = self._load_page(query, lower_bound, upper_bound, page)
            if envelope["nbPages"] != page_count:
                raise HackerNewsFetchError(
                    f"Hacker News query {query!r} changed nbPages during pagination"
                )
            hits.extend(envelope["hits"])
        return hits

    def _load_page(
        self, query: str, lower_bound: datetime, upper_bound: datetime, page: int
    ) -> dict:
        payload = json.loads(self._fetcher(self.search_url(query, lower_bound, upper_bound, page)))
        if not isinstance(payload, dict):
            raise HackerNewsFetchError(
                f"Hacker News query {query!r} returned a non-object envelope"
            )
        if "message" in payload or "error" in payload:
            raise self._truncation_error(query)
        hits = payload.get("hits")
        page_count = payload.get("nbPages")
        hit_count = payload.get("nbHits")
        if (
            not isinstance(hits, list)
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count < 0
            or isinstance(hit_count, bool)
            or not isinstance(hit_count, int)
            or hit_count < 0
        ):
            raise HackerNewsFetchError(f"Hacker News query {query!r} returned an invalid envelope")
        if hit_count > page_count * _HITS_PER_PAGE or (page < page_count and not hits):
            raise self._truncation_error(query)
        for hit in hits:
            self._validate_hit(query, hit)
        return {"hits": hits, "nbPages": page_count}

    def _truncation_error(self, query: str) -> HackerNewsFetchError:
        return HackerNewsFetchError(
            f"Hacker News query {query!r} appears truncated; "
            "lower bootstrap_lookback_hours or narrow the query"
        )

    @staticmethod
    def _validate_hit(query: str, hit: object) -> None:
        if not isinstance(hit, dict):
            raise HackerNewsFetchError(f"Hacker News query {query!r} contains a non-object hit")
        required = ("objectID", "title", "created_at_i")
        if any(key not in hit for key in required):
            raise HackerNewsFetchError(f"Hacker News query {query!r} contains a malformed hit")
        if not isinstance(hit["objectID"], str) or not isinstance(hit["title"], str):
            raise HackerNewsFetchError(f"Hacker News query {query!r} contains a malformed hit")
        timestamp = hit["created_at_i"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise HackerNewsFetchError(f"Hacker News query {query!r} contains a malformed hit")

    def _map_hit(self, hit: dict) -> RawItem:
        object_id = hit["objectID"]
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        return RawItem(
            source=self.source_name(),
            source_id=object_id,
            url=url,
            title=hit["title"],
            published_at=datetime.fromtimestamp(hit["created_at_i"], UTC),
            author=hit.get("author"),
            metadata={"points": hit.get("points"), "num_comments": hit.get("num_comments")},
        )
