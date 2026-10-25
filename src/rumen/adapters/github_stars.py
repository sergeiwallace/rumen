"""GitHub starred-repository adapter."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

from rumen.adapters.base import RawItem
from rumen.net_safety import open_public_request

Fetcher = Callable[[urllib.request.Request], str]

_STARS_URL = "https://api.github.com/user/starred"
_GRAPHQL_URL = "https://api.github.com/graphql"
_ACCEPT = "application/vnd.github.star+json"
_GRAPHQL_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2026-03-10"
_PAGE_SIZE = 100

_FIND_LIST_QUERY = """
query FindGitHubStarsList {
  viewer {
    lists(first: 100) {
      nodes {
        id
        name
      }
    }
  }
}
"""

_LIST_ITEMS_QUERY = """
query GitHubStarsListItems($listId: ID!, $cursor: String) {
  node(id: $listId) {
    ... on UserList {
      items(first: 100, after: $cursor) {
        nodes {
          ... on Repository {
            databaseId
            name
            nameWithOwner
            url
            description
            owner {
              login
            }
            primaryLanguage {
              name
            }
            repositoryTopics(first: 100) {
              nodes {
                topic {
                  name
                }
              }
            }
            stargazerCount
            pushedAt
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""


class GitHubStarsFetchError(RuntimeError):
    """GitHub stars response was malformed or unavailable."""


def _http_get(request: urllib.request.Request) -> str:
    with open_public_request(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise GitHubStarsFetchError(f"GitHub stars response has an invalid {field}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GitHubStarsFetchError(f"GitHub stars response has an invalid {field}") from exc
    if parsed.tzinfo is None:
        raise GitHubStarsFetchError(f"GitHub stars response has a naive {field}")
    return parsed.astimezone(UTC)


class GitHubStarredAdapter:
    """Fetch the authenticated user's stars with a high-water-mark cutoff."""

    def __init__(self, token_env: str, *, fetcher: Fetcher | None = None) -> None:
        self.token_env = token_env
        self._fetcher = fetcher or _http_get

    def source_name(self) -> str:
        return "github_stars"

    def stars_url(self, page: int) -> str:
        query = urllib.parse.urlencode(
            {"sort": "created", "direction": "desc", "per_page": _PAGE_SIZE, "page": page}
        )
        return f"{_STARS_URL}?{query}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        token = os.environ.get(self.token_env)
        if not token:
            raise GitHubStarsFetchError(
                f"GitHub token environment variable {self.token_env!r} is unset"
            )

        items: list[RawItem] = []
        seen: set[str] = set()
        page = 1
        while True:
            payload = self._fetch_page(page, token)
            if not payload:
                return items
            for entry in payload:
                starred_at, item = self._map_entry(entry)
                if since is not None and starred_at <= since:
                    return items
                if item.source_id not in seen:
                    seen.add(item.source_id)
                    items.append(item)
            page += 1

    def _fetch_page(self, page: int, token: str) -> list:
        request = urllib.request.Request(
            self.stars_url(page),
            headers={
                "Accept": _ACCEPT,
                "X-GitHub-Api-Version": _API_VERSION,
                "Authorization": f"Bearer {token}",
            },
        )
        try:
            payload = json.loads(self._fetcher(request))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise GitHubStarsFetchError("GitHub starred repositories request failed") from exc
        if not isinstance(payload, list):
            raise GitHubStarsFetchError("GitHub starred repositories response is not a list")
        return payload

    def _map_entry(self, entry: object) -> tuple[datetime, RawItem]:
        if not isinstance(entry, dict):
            raise GitHubStarsFetchError("GitHub stars response contains a non-object entry")
        starred_at = _parse_datetime(entry.get("starred_at"), "starred_at")
        repo = entry.get("repo")
        if not isinstance(repo, dict):
            raise GitHubStarsFetchError("GitHub stars response has an invalid repo")
        repo_id = repo.get("id")
        name = repo.get("name")
        full_name = repo.get("full_name")
        url = repo.get("html_url")
        owner = repo.get("owner")
        owner_login = owner.get("login") if isinstance(owner, dict) else None
        if (
            isinstance(repo_id, bool)
            or not isinstance(repo_id, int)
            or not isinstance(name, str)
            or not name
            or not isinstance(full_name, str)
            or not full_name
            or not isinstance(url, str)
            or not url
            or not isinstance(owner_login, str)
            or not owner_login
        ):
            raise GitHubStarsFetchError("GitHub stars response has malformed repository metadata")
        topics = repo.get("topics", [])
        if not isinstance(topics, list) or not all(isinstance(topic, str) for topic in topics):
            raise GitHubStarsFetchError("GitHub stars response has malformed repository topics")
        star_count = repo.get("stargazers_count")
        if isinstance(star_count, bool) or not isinstance(star_count, int):
            raise GitHubStarsFetchError("GitHub stars response has malformed repository star count")
        language = repo.get("language")
        if language is not None and not isinstance(language, str):
            raise GitHubStarsFetchError("GitHub stars response has malformed repository language")
        description = repo.get("description")
        if description is not None and not isinstance(description, str):
            raise GitHubStarsFetchError(
                "GitHub stars response has malformed repository description"
            )
        return starred_at, RawItem(
            source=self.source_name(),
            source_id=str(repo_id),
            url=url,
            title=full_name,
            published_at=starred_at,
            content=description,
            author=owner_login,
            metadata={
                "name": name,
                "full_name": full_name,
                "owner": owner_login,
                "language": language,
                "topics": topics,
                "star_count": star_count,
            },
        )


class GitHubStarredListAdapter:
    """Fetch repositories from one named GitHub Stars List via GraphQL."""

    def __init__(self, token_env: str, list_name: str, *, fetcher: Fetcher | None = None) -> None:
        self.token_env = token_env
        self.list_name = list_name
        self._fetcher = fetcher or _http_get

    def source_name(self) -> str:
        return f"github_stars_list:{self.list_name}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        token = os.environ.get(self.token_env)
        if not token:
            raise GitHubStarsFetchError(
                f"GitHub token environment variable {self.token_env!r} is unset"
            )

        list_id = self._find_list_id(token)
        cursor: str | None = None
        items: list[RawItem] = []
        seen: set[str] = set()
        while True:
            entries, has_next_page, end_cursor = self._fetch_list_page(token, list_id, cursor)
            for entry in entries:
                pushed_at, item = self._map_entry(entry)
                # List membership has no timestamp. ``pushedAt`` is repository activity, not the
                # time the repository was added to this list, so this cutoff is lossy/best-effort.
                # Downstream (source, source_id) deduplication prevents repeated ingestion.
                if since is not None and pushed_at <= since:
                    continue
                if item.source_id not in seen:
                    seen.add(item.source_id)
                    items.append(item)
            if not has_next_page:
                return items
            if not isinstance(end_cursor, str) or not end_cursor:
                raise GitHubStarsFetchError("GitHub Stars List response has an invalid end cursor")
            cursor = end_cursor

    def _find_list_id(self, token: str) -> str:
        payload = self._post_graphql(_FIND_LIST_QUERY, {}, token)
        viewer = payload.get("viewer")
        lists = viewer.get("lists") if isinstance(viewer, dict) else None
        nodes = lists.get("nodes") if isinstance(lists, dict) else None
        if not isinstance(nodes, list):
            raise GitHubStarsFetchError("GitHub Stars List response has invalid lists")
        for node in nodes:
            if not isinstance(node, dict) or node.get("name") != self.list_name:
                continue
            list_id = node.get("id")
            if isinstance(list_id, str) and list_id:
                return list_id
            raise GitHubStarsFetchError("GitHub Stars List response has an invalid list id")
        raise GitHubStarsFetchError(f"GitHub Stars List {self.list_name!r} was not found")

    def _fetch_list_page(
        self, token: str, list_id: str, cursor: str | None
    ) -> tuple[list, bool, str | None]:
        payload = self._post_graphql(
            _LIST_ITEMS_QUERY, {"listId": list_id, "cursor": cursor}, token
        )
        node = payload.get("node")
        items = node.get("items") if isinstance(node, dict) else None
        entries = items.get("nodes") if isinstance(items, dict) else None
        page_info = items.get("pageInfo") if isinstance(items, dict) else None
        if not isinstance(entries, list) or not isinstance(page_info, dict):
            raise GitHubStarsFetchError("GitHub Stars List response has invalid items")
        has_next_page = page_info.get("hasNextPage")
        if not isinstance(has_next_page, bool):
            raise GitHubStarsFetchError("GitHub Stars List response has invalid pagination")
        end_cursor = page_info.get("endCursor")
        if end_cursor is not None and not isinstance(end_cursor, str):
            raise GitHubStarsFetchError("GitHub Stars List response has an invalid end cursor")
        return entries, has_next_page, end_cursor

    def _post_graphql(self, query: str, variables: dict, token: str) -> dict:
        request = urllib.request.Request(
            _GRAPHQL_URL,
            data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
            headers={
                "Accept": _GRAPHQL_ACCEPT,
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": _API_VERSION,
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        try:
            response = json.loads(self._fetcher(request))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise GitHubStarsFetchError("GitHub Stars List request failed") from exc
        if not isinstance(response, dict) or response.get("errors"):
            raise GitHubStarsFetchError("GitHub Stars List request failed")
        payload = response.get("data")
        if not isinstance(payload, dict):
            raise GitHubStarsFetchError("GitHub Stars List response has invalid data")
        return payload

    def _map_entry(self, entry: object) -> tuple[datetime, RawItem]:
        if not isinstance(entry, dict):
            raise GitHubStarsFetchError("GitHub Stars List response contains a non-object entry")
        pushed_at = _parse_datetime(entry.get("pushedAt"), "pushedAt")
        repo_id = entry.get("databaseId")
        name = entry.get("name")
        full_name = entry.get("nameWithOwner")
        url = entry.get("url")
        owner = entry.get("owner")
        owner_login = owner.get("login") if isinstance(owner, dict) else None
        if (
            isinstance(repo_id, bool)
            or not isinstance(repo_id, int)
            or not isinstance(name, str)
            or not name
            or not isinstance(full_name, str)
            or not full_name
            or not isinstance(url, str)
            or not url
            or not isinstance(owner_login, str)
            or not owner_login
        ):
            raise GitHubStarsFetchError(
                "GitHub Stars List response has malformed repository metadata"
            )
        description = entry.get("description")
        if description is not None and not isinstance(description, str):
            raise GitHubStarsFetchError(
                "GitHub Stars List response has malformed repository description"
            )
        language_data = entry.get("primaryLanguage")
        if language_data is None:
            language = None
        elif isinstance(language_data, dict) and isinstance(language_data.get("name"), str):
            language = language_data["name"]
        else:
            raise GitHubStarsFetchError(
                "GitHub Stars List response has malformed repository language"
            )
        topics_data = entry.get("repositoryTopics")
        topic_nodes = topics_data.get("nodes") if isinstance(topics_data, dict) else None
        if not isinstance(topic_nodes, list):
            raise GitHubStarsFetchError(
                "GitHub Stars List response has malformed repository topics"
            )
        topics: list[str] = []
        for topic_node in topic_nodes:
            topic = topic_node.get("topic") if isinstance(topic_node, dict) else None
            topic_name = topic.get("name") if isinstance(topic, dict) else None
            if not isinstance(topic_name, str):
                raise GitHubStarsFetchError(
                    "GitHub Stars List response has malformed repository topics"
                )
            topics.append(topic_name)
        star_count = entry.get("stargazerCount")
        if isinstance(star_count, bool) or not isinstance(star_count, int):
            raise GitHubStarsFetchError(
                "GitHub Stars List response has malformed repository star count"
            )
        return pushed_at, RawItem(
            source=self.source_name(),
            source_id=str(repo_id),
            url=url,
            title=full_name,
            published_at=pushed_at,
            content=description,
            author=owner_login,
            metadata={
                "name": name,
                "full_name": full_name,
                "owner": owner_login,
                "language": language,
                "topics": topics,
                "star_count": star_count,
            },
        )
