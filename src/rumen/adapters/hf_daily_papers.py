"""Hugging Face Daily Papers adapter (design Stage 1 / T-03, T-33).

``upvotes`` and ``num_comments`` are discovery-time snapshots: ingest deliberately
does not update metadata for an already-persisted paper.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

from rumen.adapters.base import RawItem
from rumen.net_safety import open_public_request

Fetcher = Callable[[str], str]

_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
_PAGE_LIMIT = 100
_MAX_PAGES_PER_DAY = 20
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")


class HfDailyPapersFetchError(RuntimeError):
    """The provider returned a malformed or incomplete per-day result."""


def _http_get(url: str) -> str:
    with open_public_request(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class HfDailyPapersAdapter:
    """Fetch the one global Daily Papers feed through complete per-day snapshots."""

    def __init__(
        self,
        *,
        bootstrap_lookback_days: int = 7,
        daily_recheck_days: int = 2,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.bootstrap_lookback_days = bootstrap_lookback_days
        self.daily_recheck_days = daily_recheck_days
        self._fetcher = fetcher or _http_get

    def source_name(self) -> str:
        return "hf_daily"

    def day_url(self, day: date, page: int = 0) -> str:
        params = urllib.parse.urlencode({"date": day.isoformat(), "limit": _PAGE_LIMIT, "p": page})
        return f"{_DAILY_PAPERS_URL}?{params}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        run_date = _utcnow().date()
        start_day = (
            run_date - timedelta(days=self.bootstrap_lookback_days)
            if since is None
            else since.astimezone(UTC).date() - timedelta(days=self.daily_recheck_days)
        )

        items: list[RawItem] = []
        seen: set[str] = set()
        day = start_day
        while day <= run_date:
            for item in self._fetch_day(day):
                if item.source_id not in seen:
                    seen.add(item.source_id)
                    items.append(item)
            day += timedelta(days=1)
        return items

    def _fetch_day(self, day: date) -> list[RawItem]:
        items: list[RawItem] = []
        day_ids: set[str] = set()
        for page in range(_MAX_PAGES_PER_DAY):
            payload = json.loads(self._fetcher(self.day_url(day, page)))
            if not isinstance(payload, list):
                raise HfDailyPapersFetchError(
                    f"Hugging Face day {day} returned a non-list envelope"
                )
            for entry in payload:
                item = self._map_entry(entry, day)
                if item.source_id in day_ids:
                    raise HfDailyPapersFetchError(
                        f"Hugging Face day {day} contains repeated paper id {item.source_id!r}"
                    )
                day_ids.add(item.source_id)
                items.append(item)
            if len(payload) < _PAGE_LIMIT:
                return items
        raise HfDailyPapersFetchError(
            f"Hugging Face day {day} reached {_MAX_PAGES_PER_DAY} pages without a short page"
        )

    def _map_entry(self, entry: object, requested_day: date) -> RawItem:
        if not isinstance(entry, dict):
            raise HfDailyPapersFetchError("Hugging Face response contains a non-object entry")
        paper = entry.get("paper")
        if not isinstance(paper, dict):
            raise HfDailyPapersFetchError("Hugging Face entry has a malformed paper")

        paper_id = paper.get("id")
        title = paper.get("title")
        summary = paper.get("summary")
        submitted_at = self._parse_datetime(paper.get("submittedOnDailyAt"), "submittedOnDailyAt")
        self._parse_datetime(paper.get("publishedAt"), "publishedAt")
        if not isinstance(paper_id, str) or not paper_id:
            raise HfDailyPapersFetchError("Hugging Face entry has a malformed paper id")
        if not isinstance(title, str) or not title:
            raise HfDailyPapersFetchError("Hugging Face entry has a malformed paper title")
        if not isinstance(summary, str):
            raise HfDailyPapersFetchError("Hugging Face entry has a malformed paper summary")
        if submitted_at.date() != requested_day:
            raise HfDailyPapersFetchError(
                f"Hugging Face day {requested_day} contains submittedOnDailyAt "
                f"{paper['submittedOnDailyAt']!r}"
            )

        authors = paper.get("authors")
        if not isinstance(authors, list) or not authors or not isinstance(authors[0], dict):
            raise HfDailyPapersFetchError("Hugging Face entry has malformed paper authors")
        author = authors[0].get("name")
        if not isinstance(author, str) or not author:
            raise HfDailyPapersFetchError("Hugging Face entry has malformed paper authors")
        upvotes = paper.get("upvotes")
        num_comments = entry.get("numComments")
        if isinstance(upvotes, bool) or not isinstance(upvotes, int):
            raise HfDailyPapersFetchError("Hugging Face entry has malformed paper upvotes")
        if isinstance(num_comments, bool) or not isinstance(num_comments, int):
            raise HfDailyPapersFetchError("Hugging Face entry has malformed numComments")

        metadata = {
            "upvotes": upvotes,
            "num_comments": num_comments,
            "discussion_id": paper.get("discussionId"),
            "authors": authors,
            "original_published_at": paper["publishedAt"],
        }
        optional_values = {
            "organization": entry.get("organization", paper.get("organization")),
            "github_repo": paper.get("githubRepo"),
            "github_stars": paper.get("githubStars"),
            "project_page": paper.get("projectPage"),
            "media_urls": entry.get("mediaUrls", paper.get("mediaUrls")),
        }
        metadata.update({key: value for key, value in optional_values.items() if value is not None})
        return RawItem(
            source=self.source_name(),
            source_id=paper_id,
            url=self._paper_url(paper_id),
            title=title,
            content=summary,
            published_at=submitted_at,
            author=author,
            metadata=metadata,
        )

    @staticmethod
    def _parse_datetime(value: object, field: str) -> datetime:
        if not isinstance(value, str):
            raise HfDailyPapersFetchError(f"Hugging Face entry has malformed {field}")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise HfDailyPapersFetchError(f"Hugging Face entry has malformed {field}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise HfDailyPapersFetchError(f"Hugging Face entry has malformed {field}")
        return parsed.astimezone(UTC)

    @staticmethod
    def _paper_url(paper_id: str) -> str:
        if _ARXIV_ID_RE.fullmatch(paper_id):
            return f"https://arxiv.org/abs/{paper_id}"
        return f"https://huggingface.co/papers/{urllib.parse.quote(paper_id, safe='')}"
