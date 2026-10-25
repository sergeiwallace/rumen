"""Read completed publisher transcripts from the private Spotify-podcast cache."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from rumen.adapters.base import RawItem
from rumen.podcasts.cache import CacheError, open_reader, ready_rows


class SpotifyPodcastCacheError(CacheError):
    """A configured Spotify-podcast cache cannot satisfy the reader contract."""


class SpotifyPodcastAdapter:
    def __init__(
        self,
        db_path: str,
        profile: str,
        *,
        staleness_threshold: timedelta | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path
        self.profile = profile
        self.staleness_threshold = staleness_threshold
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def source_name(self) -> str:
        return f"spotify_podcasts:{self.profile}"

    def fetch_new(self, since: datetime | None) -> list[RawItem]:
        del (
            since
        )  # The cache is intentionally full-scanned; ingestion owns persistence deduplication.
        try:
            conn = open_reader(self.db_path)
            try:
                rows = ready_rows(conn)
            finally:
                conn.close()
        except CacheError as exc:
            raise SpotifyPodcastCacheError(str(exc)) from exc
        self._validate_staleness(rows)
        return [
            RawItem(
                source=self.source_name(),
                source_id=row["episode_key"],
                url=row["link"] or row["enclosure_url"],
                title=row["title"],
                published_at=datetime.fromisoformat(row["published_at"]),
                content=row["plain_text"],
                author=row["show_title"],
                metadata={
                    "show_title": row["show_title"],
                    "feed_url": row["feed_url"],
                    "guid": row["guid"],
                    "enclosure_mime": row["enclosure_mime"],
                    "spotify_episode_uris": sorted(
                        filter(None, (row["spotify_uris"] or "").split(","))
                    ),
                    "play_count": row["play_count"],
                    "first_played_at": row["first_played_at"],
                    "last_played_at": row["last_played_at"],
                    "transcript": {
                        "source_class": row["source_class"],
                        "url": row["source_url"],
                        "mime_type": row["mime_type"],
                        "language": row["language"],
                        "retrieved_at": row["retrieved_at"],
                        "content_hash": row["content_hash"],
                        "generator_identifier": row["generator_identifier"],
                    },
                    "segment_count": row["segment_count"],
                    "cache_schema_version": 1,
                },
            )
            for row in rows
        ]

    def _validate_staleness(self, rows) -> None:
        if self.staleness_threshold is None or not rows:
            return
        newest = max(datetime.fromisoformat(row["retrieved_at"]) for row in rows)
        if self._now_fn() - newest > self.staleness_threshold:
            raise SpotifyPodcastCacheError(
                f"spotify podcast cache is stale: newest transcript {newest.isoformat()}"
            )
