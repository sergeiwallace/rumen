"""Source adapters — one per ingestion source (design Stage 1 / T-03).

Each adapter implements the ``SourceAdapter`` protocol: ``fetch_new(since)`` returns
new ``RawItem``s since the last high-water mark, ``source_name()`` names the source.
Adapters normalize; ``ingest.run_ingestion`` handles high-water-mark bookkeeping.
"""

from rumen.adapters.base import (
    ChangeSourceAdapter,
    RawItem,
    SourceAdapter,
    SourceChange,
)
from rumen.adapters.chrome_reading_list import (
    ChromeReadingListAdapter,
    ChromeReadingListError,
)
from rumen.adapters.spotify_podcasts import (
    SpotifyPodcastAdapter,
    SpotifyPodcastCacheError,
)

__all__ = [
    "ChangeSourceAdapter",
    "ChromeReadingListAdapter",
    "ChromeReadingListError",
    "RawItem",
    "SourceAdapter",
    "SourceChange",
    "SpotifyPodcastAdapter",
    "SpotifyPodcastCacheError",
]
