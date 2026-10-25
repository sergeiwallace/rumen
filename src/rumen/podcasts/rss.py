"""RSS parsing and conservative Spotify-history-to-publisher identity resolution."""

from __future__ import annotations

import hashlib
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime

from rumen.podcasts.transcripts import TranscriptCandidate


class RssError(ValueError):
    """A publisher feed does not meet the constrained RSS contract."""


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _tag(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if child.tag.rsplit("}", 1)[-1] == name), None)


def _text(element: ET.Element, name: str) -> str | None:
    found = _tag(element, name)
    return found.text.strip() if found is not None and found.text else None


@dataclass(frozen=True)
class FeedEpisode:
    title: str
    guid: str | None
    link: str | None
    enclosure_url: str | None
    enclosure_mime: str | None
    published_at: str
    show_title: str
    transcripts: list[TranscriptCandidate] = field(default_factory=list)


def parse_feed(body: bytes | str) -> list[FeedEpisode]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise RssError("publisher feed is not valid XML") from exc
    channel = next((node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "channel"), None)
    if channel is None:
        raise RssError("publisher feed has no RSS channel")
    show_title = _text(channel, "title")
    if not show_title:
        raise RssError("publisher feed channel has no title")
    episodes: list[FeedEpisode] = []
    for item in (node for node in channel if node.tag.rsplit("}", 1)[-1] == "item"):
        title = _text(item, "title")
        pub_date = _text(item, "pubDate")
        if not title or not pub_date:
            continue
        try:
            published_at = parsedate_to_datetime(pub_date).astimezone(UTC).isoformat()
        except (TypeError, ValueError) as exc:
            raise RssError("publisher episode has invalid pubDate") from exc
        enclosure = _tag(item, "enclosure")
        candidates = [
            TranscriptCandidate(
                child.attrib.get("url", ""),
                child.attrib.get("type", ""),
                child.attrib.get("language"),
                child.attrib.get("rel", "").casefold() == "generated",
            )
            for child in item
            if child.tag.rsplit("}", 1)[-1] == "transcript"
        ]
        episodes.append(
            FeedEpisode(
                title,
                _text(item, "guid"),
                _text(item, "link"),
                enclosure.attrib.get("url") if enclosure is not None else None,
                enclosure.attrib.get("type") if enclosure is not None else None,
                published_at,
                show_title,
                candidates,
            )
        )
    return episodes


def episode_key(feed_url: str, episode: FeedEpisode) -> str | None:
    identity = episode.guid or episode.enclosure_url or episode.link
    if not identity:
        return None
    return hashlib.sha256(f"{feed_url}\0{identity}".encode()).hexdigest()


@dataclass(frozen=True)
class Resolution:
    episode: FeedEpisode | None
    method: str | None
    evidence: str
    reason: str | None = None


def resolve_episode(
    spotify_uri: str,
    title: str,
    feed_url: str,
    episodes: list[FeedEpisode],
    aliases: dict[str, str],
) -> Resolution:
    alias_guid = aliases.get(spotify_uri)
    if alias_guid is not None:
        matching = [episode for episode in episodes if episode.guid == alias_guid]
        if len(matching) == 1 and episode_key(feed_url, matching[0]):
            return Resolution(matching[0], "explicit-alias", f"guid:{alias_guid}")
        return Resolution(None, None, f"guid:{alias_guid}", "alias-missing-or-unidentifiable")
    matching = [
        episode for episode in episodes if normalize_name(episode.title) == normalize_name(title)
    ]
    if len(matching) != 1:
        return Resolution(
            None,
            None,
            ", ".join(episode.title for episode in matching),
            "zero-or-ambiguous-title-match",
        )
    if episode_key(feed_url, matching[0]) is None:
        return Resolution(None, None, matching[0].title, "episode-has-no-canonical-identity")
    return Resolution(matching[0], "exact-normalized-title", matching[0].title)
