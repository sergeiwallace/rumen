"""Official OAuth Reddit subreddit discovery adapter.

The PRAW boundary is deliberately small so all normal tests can use an injected
gateway without credentials or a network connection.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from rumen.adapters.base import RawItem


class RedditFetchError(RuntimeError):
    """A sanitized Reddit source failure."""


class RedditGateway(Protocol):
    def listing(self, subreddit: str, mode: str, *, limit: int) -> Iterable[object]: ...

    def revalidate(self, fullnames: list[str]) -> Iterable[object]: ...


@dataclass(frozen=True, slots=True)
class RedditSubredditSettings:
    profile: str
    subreddits: tuple[str, ...]
    modes: tuple[str, ...]
    client_id_env: str
    client_secret_env: str
    user_agent: str
    approval_reference: str
    bootstrap_lookback_hours: int = 72
    replay_lookback_hours: int = 24
    retention_hours: int = 48
    listing_item_ceiling: int = 1000
    request_timeout_seconds: int = 30
    run_timeout_seconds: int = 300


class PrawRedditGateway:
    """Lazy application-only PRAW gateway; never expose its client or credentials."""

    def __init__(self, settings: RedditSubredditSettings) -> None:
        self._settings = settings
        self._reddit = None

    def _client(self):
        if self._reddit is not None:
            return self._reddit
        client_id = os.environ.get(self._settings.client_id_env)
        client_secret = os.environ.get(self._settings.client_secret_env)
        if not client_id or not client_secret:
            raise RedditFetchError(
                "Reddit OAuth credentials are unavailable; set configured variables"
            )
        try:
            import praw
        except ImportError as exc:
            raise RedditFetchError("Reddit support requires `pip install rumen[reddit]`") from exc
        try:
            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=self._settings.user_agent,
                read_only=True,
                requestor_kwargs={"timeout": self._settings.request_timeout_seconds},
            )
        except Exception as exc:  # PRAW can include headers/configuration in its errors.
            raise RedditFetchError("unable to construct Reddit OAuth client") from exc
        return self._reddit

    def listing(self, subreddit: str, mode: str, *, limit: int) -> Iterable[object]:
        try:
            community = self._client().subreddit(subreddit)
            return (
                community.new(limit=limit)
                if mode == "submissions"
                else community.comments(limit=limit)
            )
        except RedditFetchError:
            raise
        except Exception as exc:
            raise RedditFetchError("Reddit listing request failed") from exc

    def revalidate(self, fullnames: list[str]) -> Iterable[object]:
        try:
            return self._client().info(fullnames=fullnames)
        except RedditFetchError:
            raise
        except Exception as exc:
            raise RedditFetchError("Reddit revalidation request failed") from exc


class RedditSubredditAdapter:
    def __init__(
        self,
        settings: RedditSubredditSettings,
        *,
        gateway: RedditGateway | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self._gateway = gateway or PrawRedditGateway(settings)
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def source_name(self) -> str:
        return f"reddit:{self.settings.profile}"

    @property
    def gateway(self) -> RedditGateway:
        return self._gateway

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        now = self._now_fn()
        cutoff = now - timedelta(hours=self.settings.retention_hours)
        if since is None:
            cutoff = max(cutoff, now - timedelta(hours=self.settings.bootstrap_lookback_hours))
        else:
            cutoff = max(
                cutoff,
                since.astimezone(UTC) - timedelta(hours=self.settings.replay_lookback_hours),
            )
        seen: set[str] = set()
        result: list[RawItem] = []
        deadline = time.monotonic() + self.settings.run_timeout_seconds
        try:
            for subreddit in self.settings.subreddits:
                for mode in self.settings.modes:
                    stream_count = 0
                    reached_cutoff = False
                    for obj in self._gateway.listing(
                        subreddit, mode, limit=min(100, self.settings.listing_item_ceiling)
                    ):
                        if time.monotonic() > deadline:
                            raise RedditFetchError("Reddit fetch exceeded configured run timeout")
                        stream_count += 1
                        if stream_count > self.settings.listing_item_ceiling:
                            raise RedditFetchError("Reddit listing window was truncated")
                        item = self._map(obj, mode)
                        if item.published_at <= cutoff:
                            reached_cutoff = True
                            break
                        if item.source_id not in seen:
                            seen.add(item.source_id)
                            result.append(item)
                    if stream_count >= self.settings.listing_item_ceiling and not reached_cutoff:
                        raise RedditFetchError("Reddit listing window was truncated")
        except RedditFetchError:
            raise
        except Exception as exc:
            raise RedditFetchError("Reddit fetch failed") from exc
        return result

    def _map(self, obj: object, mode: str) -> RawItem:
        fullname = getattr(obj, "name", None)
        created = getattr(obj, "created_utc", None)
        permalink = getattr(obj, "permalink", None)
        subreddit = getattr(getattr(obj, "subreddit", None), "display_name", None)
        if (
            not isinstance(fullname, str)
            or not fullname.startswith("t3_" if mode == "submissions" else "t1_")
            or isinstance(created, bool)
            or not isinstance(created, (int, float))
            or not isinstance(permalink, str)
            or not permalink.startswith("/")
            or not isinstance(subreddit, str)
            or not subreddit
        ):
            raise RedditFetchError("Reddit listing contains malformed required fields")
        published_at = datetime.fromtimestamp(created, UTC)
        author_obj = getattr(obj, "author", None)
        author = getattr(author_obj, "name", None) if author_obj is not None else None
        author = author if isinstance(author, str) else None
        absolute_permalink = f"https://www.reddit.com{permalink}"
        if mode == "submissions":
            title = getattr(obj, "title", None)
            if not isinstance(title, str) or not title:
                raise RedditFetchError("Reddit submission contains malformed title")
            selftext = getattr(obj, "selftext", None)
            content = (
                selftext
                if isinstance(selftext, str) and selftext not in {"[removed]", "[deleted]"}
                else None
            )
            external = getattr(obj, "url", None)
            is_self = getattr(obj, "is_self", None)
            if not isinstance(is_self, bool):
                raise RedditFetchError("Reddit submission contains malformed is_self")
            if not is_self and (not isinstance(external, str) or not external):
                raise RedditFetchError("Reddit submission contains malformed url")
            return RawItem(
                source=self.source_name(),
                source_id=fullname,
                url=absolute_permalink if is_self else external,
                title=title,
                content=content,
                author=author,
                published_at=published_at,
                metadata={
                    "kind": "submission",
                    "subreddit": subreddit,
                    "permalink": permalink,
                    "flair": getattr(obj, "link_flair_text", None),
                    "nsfw": bool(getattr(obj, "over_18", False)),
                    "score": getattr(obj, "score", None),
                    "comment_count": getattr(obj, "num_comments", None),
                },
            )
        body = getattr(obj, "body", None)
        content = body if isinstance(body, str) and body not in {"[removed]", "[deleted]"} else None
        link_fullname = getattr(obj, "link_id", None)
        parent_fullname = getattr(obj, "parent_id", None)
        if not isinstance(link_fullname, str) or not isinstance(parent_fullname, str):
            raise RedditFetchError("Reddit comment contains malformed parent fields")
        return RawItem(
            source=self.source_name(),
            source_id=fullname,
            url=absolute_permalink,
            title=f"Comment in r/{subreddit}",
            content=content,
            author=author,
            published_at=published_at,
            metadata={
                "kind": "comment",
                "subreddit": subreddit,
                "permalink": permalink,
                "link_fullname": link_fullname,
                "parent_fullname": parent_fullname,
            },
        )
