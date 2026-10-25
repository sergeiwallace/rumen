"""Configured GitHub repository adapter."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

from rumen.adapters.base import RawItem
from rumen.net_safety import open_public_request

Fetcher = Callable[[urllib.request.Request], str]

_API_VERSION = "2026-03-10"


class GitHubRepositoryFetchError(RuntimeError):
    """GitHub repository response was malformed or unavailable."""


def _http_get(request: urllib.request.Request) -> str:
    with open_public_request(request, timeout=30) as response:
        return response.read().decode("utf-8")


class GitHubRepositoryAdapter:
    """Track the current metadata of one configured ``owner/repo`` repository."""

    def __init__(self, repo: str, token_env: str, *, fetcher: Fetcher | None = None) -> None:
        self.repo = repo
        self.token_env = token_env
        self._fetcher = fetcher or _http_get

    def source_name(self) -> str:
        return f"github_repositories:{self.repo}"

    def repository_url(self) -> str:
        return f"https://api.github.com/repos/{self.repo}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        token = os.environ.get(self.token_env)
        if not token:
            raise GitHubRepositoryFetchError(
                f"GitHub token environment variable {self.token_env!r} is unset"
            )
        request = urllib.request.Request(
            self.repository_url(),
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": _API_VERSION,
                "Authorization": f"Bearer {token}",
            },
        )
        try:
            payload = json.loads(self._fetcher(request))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise GitHubRepositoryFetchError("GitHub repository request failed") from exc
        item = self._map_repository(payload)
        return [item] if since is None or item.published_at > since else []

    def _map_repository(self, payload: object) -> RawItem:
        if not isinstance(payload, dict):
            raise GitHubRepositoryFetchError("GitHub repository response is not an object")
        repo_id = payload.get("id")
        name = payload.get("name")
        full_name = payload.get("full_name")
        url = payload.get("html_url")
        pushed_at = self._parse_pushed_at(payload.get("pushed_at"))
        owner = payload.get("owner")
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
            raise GitHubRepositoryFetchError(
                "GitHub repository response has malformed required fields"
            )
        topics = payload.get("topics", [])
        if not isinstance(topics, list) or not all(isinstance(topic, str) for topic in topics):
            raise GitHubRepositoryFetchError("GitHub repository response has malformed topics")
        star_count = payload.get("stargazers_count")
        if isinstance(star_count, bool) or not isinstance(star_count, int):
            raise GitHubRepositoryFetchError("GitHub repository response has malformed star count")
        language = payload.get("language")
        if language is not None and not isinstance(language, str):
            raise GitHubRepositoryFetchError("GitHub repository response has malformed language")
        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            raise GitHubRepositoryFetchError("GitHub repository response has malformed description")
        return RawItem(
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
                "topics": topics,
                "language": language,
                "star_count": star_count,
                "pushed_at": payload["pushed_at"],
            },
        )

    def _parse_pushed_at(self, value: object) -> datetime:
        if not isinstance(value, str) or not value:
            raise GitHubRepositoryFetchError("GitHub repository response has an invalid pushed_at")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise GitHubRepositoryFetchError(
                "GitHub repository response has an invalid pushed_at"
            ) from exc
        if parsed.tzinfo is None:
            raise GitHubRepositoryFetchError("GitHub repository response has a naive pushed_at")
        return parsed.astimezone(UTC)
