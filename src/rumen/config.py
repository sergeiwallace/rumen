"""Pipeline config loader — adapters + LLM command from a JSON file (config over code).

Per-tool source mapping is config, not code (design Stage 1): each tracked source
declares its adapter type + locator. Keeps the open-core engine free of personal
specifics (paths, package lists) — those live in the operator's config file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from rumen.adapters.base import ChangeSourceAdapter, SourceAdapter
from rumen.adapters.bookmarks import BookmarksAdapter
from rumen.adapters.chrome_bookmarks import ChromeBookmarksAdapter
from rumen.adapters.chrome_reading_list import ChromeReadingListAdapter
from rumen.adapters.github_releases import GitHubReleasesAdapter
from rumen.adapters.github_repository import GitHubRepositoryAdapter
from rumen.adapters.github_stars import GitHubStarredAdapter, GitHubStarredListAdapter
from rumen.adapters.hacker_news import HackerNewsAdapter
from rumen.adapters.hf_daily_papers import HfDailyPapersAdapter
from rumen.adapters.npm_registry import NpmRegistryAdapter
from rumen.adapters.paperpile import PaperpileAdapter
from rumen.adapters.reddit import RedditSavedAdapter
from rumen.adapters.reddit_subreddits import (
    RedditSubredditAdapter,
    RedditSubredditSettings,
)
from rumen.adapters.spotify_podcasts import SpotifyPodcastAdapter
from rumen.adapters.youtube import YouTubePlaylistAdapter
from rumen.chrome_reading_list import (
    MINIMUM_CHROME_VERSION,
    ChromeReadingListSourceConfig,
)
from rumen.llm import FallbackLlmCommand, LlmCommand, SubprocessLlmCommand
from rumen.source_maintenance import SourceMaintainer


@dataclass(frozen=True, slots=True)
class TargetTemplate:
    """Validated, application-owned target/template selection data."""

    beads_repo_identity: str
    task_store: str
    template_revision: str

    @property
    def target_identity(self) -> str:
        return f"{self.beads_repo_identity}::{self.template_revision}".lower()


@dataclass(frozen=True, slots=True)
class MappingWriteResult:
    """Outcome of a locked mapping write, including a concurrent-writer resolution."""

    target: TargetTemplate
    created: bool


@dataclass(frozen=True, slots=True)
class CcTaskCreationConfig:
    """Explicit opt-in for session-local candidate draft projection."""

    enabled: bool = False


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Optional explicit database DSN supplied by a sanitized pipeline config."""

    dsn: str | None = None


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Persistence backend selection; only the SQLAlchemy backend is available today."""

    backend: str = "sqlalchemy"


@dataclass(slots=True)
class PipelineConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    adapters: list[SourceAdapter | ChangeSourceAdapter] = field(default_factory=list)
    source_maintainers: list[SourceMaintainer] = field(default_factory=list)
    # Retained for connector diagnostics and native-host installation tooling; enabled
    # sources also create a normal SourceAdapter below.
    chrome_reading_list_sources: list[ChromeReadingListSourceConfig] = field(default_factory=list)
    llm_command: LlmCommand | None = None
    # D-16/T-21: resolved only from AI_HOST; never inferred from a machine name.
    ai_host: str | None = None
    # Validated host-local Ollama command specs, retained for opt-in digest polish.
    local_llm_by_host: dict[str, dict] = field(default_factory=dict)
    out_of_scope: frozenset[str] = frozenset()
    digest_top_n: int = 20
    # D-17/T-19 extractive digest: lower-cased top-level topic ``pref_label``→section overrides
    # (falling back to the same-shaped legacy category keys) layered over built-in defaults.
    digest_section_map: dict[str, str] = field(default_factory=dict)
    # Safeguard ceiling on Tier-2 LLM calls per run (Tier-2 should fire rarely; a spike
    # means a routing bug — cap it rather than drain the backend / quota).
    max_llm_calls: int = 200
    # Incremental-durability batch size (design D-14): every pipeline stage commits after
    # this many rows so an interrupted run keeps completed batches and resumes (idempotent).
    commit_batch_size: int = 500
    # Per-row write retries give transient failures a bounded chance to recover without
    # abandoning the remainder of the classification pass.
    classify_write_max_attempts: int = 3
    classify_write_retry_backoff_seconds: float = 1.0
    # LLM subprocess timeout (seconds) for the config-driven command (Tier-2 + digest).
    llm_timeout: float = 120.0
    # Path to the persisted per-facet HMC bundle directory from `rumen fit-classifier`. An absent
    # bundle intentionally leaves facets to Tier-0 plus their constrained Tier-2 fallback.
    classifier_path: str | None = None
    # D-15 operator alarm threshold for a completed facet-decision escalation rate.
    classifier_escalation_alarm_threshold: float = 0.25
    # CAP's published data is CC BY-NC-SA 4.0.  Keep its labels out of an OSS
    # bundle until an operator has separately confirmed redistribution rights.
    rights_confirmed: bool = False
    # Donor entries are deliberately data/config, not a hard-coded import list.
    # ``graft_donor`` refuses entries without documented reuse verification.
    taxonomy_donors: list[dict] = field(default_factory=list)
    # --- Tunable pipeline thresholds (design D-2/D-8/D-12 — "config-driven, tuned later by
    # the feedback loop"). All heuristics from the parameter research; override per deployment.
    tau_class: float = 0.75  # D-12 Tier-1 confidence gate → below this, escalate to Tier-2 LLM
    tau_high: float = 0.92  # D-8 residue cosine auto-merge threshold
    tau_low: float = 0.80  # D-8 residue review-band floor (tau_low..tau_high → review queue)
    mmr_lambda: float = 0.5  # D-2 MMR relevance-vs-diversity tradeoff
    exploration_fraction: float = 0.12  # D-2 slots reserved for lower-relevance/novel entities
    # D-2 per-category time-decay half-lives (days); unlisted categories use the fallback.
    half_life_days: dict[str, float] = field(default_factory=dict)
    default_half_life_days: float = 14.0
    residue_window_days: int = 30
    # D-23 scheduled maintenance audit guardrails.  They are config, not hidden
    # constants, because operators need to tune conservatism to corpus size.
    taxonomy_audit_crowded_item_threshold: int = 100
    taxonomy_audit_sparse_item_threshold: int = 0
    taxonomy_audit_duplicate_similarity_threshold: float = 0.92
    # Category routing is configuration, never model/source data.
    category_to_target_template: dict[str, TargetTemplate] = field(default_factory=dict)
    backlog_mining_enabled: bool = False
    cc_task_creation: CcTaskCreationConfig = field(default_factory=CcTaskCreationConfig)
    auto_create_mapping: bool = False
    auto_create_mapping_confidence_threshold: float = 1.0
    auto_mappable_categories: tuple[str, ...] = ()
    default_target_template: TargetTemplate | None = None
    # The actionability judge uses the existing command/fallback seam, independently
    # configurable from classification/digest when an operator needs that separation.
    actionability_command_factory: Callable[[], LlmCommand | None] | None = None
    actionability_max_calls: int = 200
    backlog_max_age_days: int = 30
    backlog_max_rank_position: int = 20
    config_path: Path | None = None


def _target_template(value: object, *, field_name: str) -> TargetTemplate:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    expected = {"beads_repo_identity", "task_store", "template_revision"}
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise ValueError(f"{field_name} has {'; '.join(details)} fields")
    values = {key: value[key] for key in expected}
    if not all(isinstance(item, str) and item.strip() for item in values.values()):
        raise ValueError(f"{field_name} fields must be non-empty strings")
    if "::" in values["beads_repo_identity"] or "::" in values["template_revision"]:
        raise ValueError(
            f"{field_name} beads_repo_identity/template_revision must not contain '::'"
        )
    return TargetTemplate(**values)


def _target_templates(data: dict) -> dict[str, TargetTemplate]:
    raw = data.get("category_to_target_template", {})
    if not isinstance(raw, dict):
        raise TypeError("category_to_target_template must be a category-to-target object")
    templates: dict[str, TargetTemplate] = {}
    for category, target in raw.items():
        if not isinstance(category, str) or not category:
            raise ValueError(
                "category_to_target_template keys must be non-empty canonical categories"
            )
        templates[category] = _target_template(
            target, field_name=f"category_to_target_template[{category!r}]"
        )
    return templates


def mapping_confidence(config: PipelineConfig, category: str | None) -> float:
    """The deliberately closed missing-mapping confidence function (T-2.3)."""
    if category is None or category in config.category_to_target_template:
        return 0.0
    return 1.0 if category in config.auto_mappable_categories else 0.0


def create_category_mapping_result(config: PipelineConfig, category: str) -> MappingWriteResult:
    """Persist one deterministic mapping with a read-before-write conflict check.

    Tests and embedded callers without a config file still get the same in-memory
    state transition; file-backed configurations are replaced atomically.
    """
    if category in config.category_to_target_template:
        return MappingWriteResult(config.category_to_target_template[category], created=False)
    if config.default_target_template is None:
        raise ValueError("cannot create a category mapping without default_target_template")
    target = config.default_target_template
    if config.config_path is not None:
        # A same-directory advisory lock serializes writers.  Re-reading only after
        # acquiring it makes the conflict check cover the bytes that are replaced.
        import fcntl

        lock_path = config.config_path.with_suffix(f"{config.config_path.suffix}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                raw = json.loads(config.config_path.read_text())
                mappings = raw.setdefault("category_to_target_template", {})
                if category in mappings:
                    resolved = _target_templates(raw)[category]
                    config.category_to_target_template[category] = resolved
                    return MappingWriteResult(resolved, created=False)
                mappings[category] = {
                    "beads_repo_identity": target.beads_repo_identity,
                    "task_store": target.task_store,
                    "template_revision": target.template_revision,
                }
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=config.config_path.parent, delete=False
                ) as temporary:
                    json.dump(raw, temporary, indent=2, sort_keys=True)
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, config.config_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    config.category_to_target_template[category] = target
    return MappingWriteResult(target, created=True)


def create_category_mapping(config: PipelineConfig, category: str) -> TargetTemplate:
    """Persist a mapping and return its target (compatibility helper)."""
    return create_category_mapping_result(config, category).target


def migration_target_identity(path: str | Path) -> str | None:
    """Resolve the one Phase-1 target used by the family-key migration, if any."""
    data = json.loads(Path(path).read_text())
    targets = {target.target_identity for target in _target_templates(data).values()}
    if not targets and data.get("default_target_template") is not None:
        targets.add(
            _target_template(
                data["default_target_template"], field_name="default_target_template"
            ).target_identity
        )
    return next(iter(targets)) if len(targets) == 1 else None


def _build_adapter(spec: dict) -> SourceAdapter | ChangeSourceAdapter:
    kind = spec.get("type")
    if kind == "github_releases":
        return GitHubReleasesAdapter(spec["repo"])
    if kind == "npm_registry":
        return NpmRegistryAdapter(spec["package"], spec["tool_name"])
    if kind == "bookmarks":
        return BookmarksAdapter(
            spec["db_path"],
            staleness_threshold=timedelta(hours=spec.get("staleness_hours", 48)),
        )
    if kind == "reddit_saved":
        _validate_reddit_saved_spec(spec)
        return RedditSavedAdapter(
            spec["db_path"],
            staleness_threshold=timedelta(hours=spec.get("staleness_hours", 48)),
        )
    if kind == "reddit_subreddits":
        return RedditSubredditAdapter(_validate_reddit_subreddits_spec(spec))
    if kind == "paperpile":
        _validate_paperpile_spec(spec)
        return PaperpileAdapter(
            spec["profile"],
            spec["db_path"],
            staleness_threshold=timedelta(hours=spec.get("staleness_hours", 48)),
        )
    if kind == "github_stars":
        _validate_github_stars_spec(spec)
        return GitHubStarredAdapter(spec["token_env"])
    if kind == "github_stars_list":
        _validate_github_stars_list_spec(spec)
        return GitHubStarredListAdapter(spec["token_env"], spec["list_name"])
    if kind == "chrome_bookmarks":
        _validate_chrome_bookmarks_spec(spec)
        return ChromeBookmarksAdapter(
            spec["bookmarks_path"], spec["profile"], spec.get("folder_name")
        )
    if kind == "github_repository":
        _validate_github_repository_spec(spec)
        return GitHubRepositoryAdapter(spec["repo"], spec["token_env"])
    if kind == "hacker_news":
        return HackerNewsAdapter(
            _validate_hn_profile(spec),
            _validate_hn_queries(spec),
            bootstrap_lookback_hours=_validate_int(
                spec, "hacker_news", "bootstrap_lookback_hours", default=72, minimum=1, maximum=168
            ),
            indexing_overlap_minutes=_validate_int(
                spec, "hacker_news", "indexing_overlap_minutes", default=15, minimum=0, maximum=1440
            ),
        )
    if kind == "hf_daily_papers":
        _validate_hf_daily_papers_spec(spec)
        return HfDailyPapersAdapter(
            bootstrap_lookback_days=_validate_int(
                spec, "hf_daily_papers", "bootstrap_lookback_days", default=7, minimum=1, maximum=90
            ),
            daily_recheck_days=_validate_int(
                spec, "hf_daily_papers", "daily_recheck_days", default=2, minimum=0, maximum=30
            ),
        )
    if kind == "youtube_playlist":
        _validate_youtube_playlist_spec(spec)
        return YouTubePlaylistAdapter(
            spec["playlist_id"],
            spec["api_key_env"],
            spec["max_api_units_per_run"],
            metadata_refresh_days=spec.get("metadata_refresh_days", 28),
            transcript_mode=spec.get("transcript_mode", "off"),
            transcript_languages=spec.get("transcript_languages", []),
            prefer_manual_transcript=spec.get("prefer_manual_transcript", True),
        )
    if kind == "rss_atom":
        _validate_rss_atom_spec(spec)
        # This import is deliberately branch-local: a base installation can still
        # import and run Rumen until an operator opts into an RSS/Atom source.
        from rumen.adapters.rss_atom import RssAtomAdapter

        return RssAtomAdapter(
            spec["profile"], spec["feed_url"], spec.get("min_poll_interval_seconds", 900)
        )
    if kind == "spotify_podcasts":
        _validate_spotify_podcasts_spec(spec)
        return SpotifyPodcastAdapter(
            spec["db_path"],
            spec["profile"],
            staleness_threshold=(
                timedelta(hours=spec["staleness_hours"]) if "staleness_hours" in spec else None
            ),
        )
    raise ValueError(f"unknown source type: {kind!r}")


_HN_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HN_KEYS = frozenset(
    {"type", "host", "profile", "queries", "bootstrap_lookback_hours", "indexing_overlap_minutes"}
)
_HF_DAILY_PAPERS_KEYS = frozenset({"type", "host", "bootstrap_lookback_days", "daily_recheck_days"})
_GITHUB_STARS_KEYS = frozenset({"type", "host", "token_env"})
_GITHUB_STARS_LIST_KEYS = frozenset({"type", "host", "token_env", "list_name"})
_CHROME_BOOKMARKS_KEYS = frozenset({"type", "host", "bookmarks_path", "profile", "folder_name"})
_GITHUB_REPOSITORY_KEYS = frozenset({"type", "host", "repo", "token_env"})
_REDDIT_SAVED_KEYS = frozenset({"type", "host", "db_path", "staleness_hours"})
_REDDIT_SUBREDDITS_KEYS = frozenset(
    {
        "type",
        "host",
        "enabled",
        "profile",
        "subreddits",
        "modes",
        "client_id_env",
        "client_secret_env",
        "user_agent",
        "approval_reference",
        "bootstrap_lookback_hours",
        "replay_lookback_hours",
        "retention_hours",
        "listing_item_ceiling",
        "request_timeout_seconds",
        "run_timeout_seconds",
    }
)
_PAPERPILE_KEYS = frozenset({"type", "host", "profile", "db_path", "staleness_hours"})
_SPOTIFY_PODCASTS_KEYS = frozenset({"type", "host", "profile", "db_path", "staleness_hours"})
_CHROME_READING_LIST_KEYS = frozenset(
    {"type", "host", "enabled", "profile", "inbox_path", "minimum_chrome_version"}
)
_YOUTUBE_PLAYLIST_KEYS = frozenset(
    {
        "type",
        "host",
        "playlist_id",
        "api_key_env",
        "max_api_units_per_run",
        "metadata_refresh_days",
        "transcript_mode",
        "transcript_languages",
        "prefer_manual_transcript",
    }
)
_RSS_ATOM_KEYS = frozenset({"type", "host", "profile", "feed_url", "min_poll_interval_seconds"})
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GITHUB_REPOSITORY_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


def _validate_hn_profile(spec: dict) -> str:
    profile = spec.get("profile")
    if not isinstance(profile, str) or not _HN_PROFILE_RE.fullmatch(profile):
        raise ValueError("hacker_news profile must match ^[a-z0-9][a-z0-9_-]*$")
    return profile


def _validate_rss_atom_spec(spec: dict) -> None:
    _validate_source_keys(spec, "rss_atom", _RSS_ATOM_KEYS)
    profile = spec.get("profile")
    if not isinstance(profile, str) or not _HN_PROFILE_RE.fullmatch(profile):
        raise ValueError("rss_atom profile must match ^[a-z0-9][a-z0-9_-]*$")
    feed_url = spec.get("feed_url")
    parsed = urlsplit(feed_url) if isinstance(feed_url, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError(
            "rss_atom feed_url must be an absolute HTTPS URL without credentials or fragment"
        )
    _validate_int(
        spec, "rss_atom", "min_poll_interval_seconds", default=900, minimum=60, maximum=86400
    )


def _validate_spotify_podcasts_spec(spec: dict) -> None:
    _validate_source_keys(spec, "spotify_podcasts", _SPOTIFY_PODCASTS_KEYS)
    profile = spec.get("profile")
    if not isinstance(profile, str) or not _HN_PROFILE_RE.fullmatch(profile):
        raise ValueError("spotify_podcasts profile must match ^[a-z0-9][a-z0-9_-]*$")
    if not isinstance(spec.get("db_path"), str) or not spec["db_path"].strip():
        raise ValueError("spotify_podcasts db_path must be a non-empty string")
    if "staleness_hours" in spec:
        value = spec["staleness_hours"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("spotify_podcasts staleness_hours must be a positive integer")


def _validate_spotify_podcast_profiles(specs: list[dict]) -> None:
    profiles: set[str] = set()
    for spec in specs:
        if spec.get("type") != "spotify_podcasts":
            continue
        profile = spec.get("profile")
        if profile in profiles:
            raise ValueError(f"duplicate host-applicable spotify_podcasts profile: {profile!r}")
        profiles.add(profile)


def _validate_hn_queries(spec: dict) -> list[str]:
    queries = spec.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("hacker_news queries must be a non-empty list")
    canonical: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if not isinstance(query, str):
            raise TypeError("hacker_news queries must contain only strings")
        normalized = " ".join(unicodedata.normalize("NFKC", query).strip().split())
        if not normalized:
            raise ValueError("hacker_news queries must not be empty")
        key = normalized.casefold()
        if key in seen:
            raise ValueError("hacker_news queries must not contain canonical duplicates")
        seen.add(key)
        canonical.append(normalized)
    unknown = set(spec) - _HN_KEYS
    if unknown:
        raise ValueError(f"unknown hacker_news source keys: {', '.join(sorted(unknown))}")
    return canonical


def _validate_int(
    spec: dict, kind: str, key: str, *, default: int, minimum: int, maximum: int
) -> int:
    value = spec.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{kind} {key} must be an integer in [{minimum}, {maximum}]")
    return value


def _validate_hf_daily_papers_spec(spec: dict) -> None:
    unknown = set(spec) - _HF_DAILY_PAPERS_KEYS
    if unknown:
        raise ValueError(f"unknown hf_daily_papers source keys: {', '.join(sorted(unknown))}")


def _validate_github_stars_spec(spec: dict) -> None:
    _validate_source_keys(spec, "github_stars", _GITHUB_STARS_KEYS)
    _validate_token_env(spec, "github_stars")


def _validate_github_stars_list_spec(spec: dict) -> None:
    _validate_source_keys(spec, "github_stars_list", _GITHUB_STARS_LIST_KEYS)
    _validate_token_env(spec, "github_stars_list")
    list_name = spec.get("list_name")
    if not isinstance(list_name, str) or not list_name:
        raise ValueError("github_stars_list list_name must be a non-empty string")


def _validate_chrome_bookmarks_spec(spec: dict) -> None:
    _validate_source_keys(spec, "chrome_bookmarks", _CHROME_BOOKMARKS_KEYS)
    path = spec.get("bookmarks_path")
    if not isinstance(path, str) or not path:
        raise ValueError("chrome_bookmarks bookmarks_path must be a non-empty string")
    profile = spec.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("chrome_bookmarks profile must be a non-empty string")
    folder_name = spec.get("folder_name")
    if folder_name is not None and (not isinstance(folder_name, str) or not folder_name):
        raise ValueError("chrome_bookmarks folder_name must be a non-empty string when set")


def _validate_github_repository_spec(spec: dict) -> None:
    _validate_source_keys(spec, "github_repository", _GITHUB_REPOSITORY_KEYS)
    repo = spec.get("repo")
    if not isinstance(repo, str) or not _GITHUB_REPOSITORY_RE.fullmatch(repo):
        raise ValueError("github_repository repo must be exactly owner/repo")
    _validate_token_env(spec, "github_repository")


def _validate_reddit_saved_spec(spec: dict) -> None:
    _validate_source_keys(spec, "reddit_saved", _REDDIT_SAVED_KEYS)
    path = spec.get("db_path")
    if not isinstance(path, str) or not path:
        raise ValueError("reddit_saved db_path must be a non-empty string")


def _validate_reddit_subreddits_spec(spec: dict) -> RedditSubredditSettings:
    _validate_source_keys(spec, "reddit_subreddits", _REDDIT_SUBREDDITS_KEYS)
    enabled = spec.get("enabled")
    if not isinstance(enabled, bool):
        raise TypeError("reddit_subreddits enabled must be a boolean")
    if not enabled:
        raise ValueError("disabled reddit_subreddits sources must not be built")
    profile = spec.get("profile")
    if not isinstance(profile, str) or not _HN_PROFILE_RE.fullmatch(profile):
        raise ValueError("reddit_subreddits profile must match ^[a-z0-9][a-z0-9_-]*$")
    subreddits = spec.get("subreddits")
    if not isinstance(subreddits, list) or not subreddits:
        raise ValueError("reddit_subreddits subreddits must be a non-empty list")
    canonical: list[str] = []
    seen: set[str] = set()
    for subreddit in subreddits:
        if not isinstance(subreddit, str):
            raise TypeError("reddit_subreddits subreddits must contain only strings")
        value = subreddit.strip()
        if (
            not value
            or value.casefold().startswith("r/")
            or "/" in value
            or any(c.isspace() for c in value)
        ):
            raise ValueError("reddit_subreddits subreddit is invalid")
        if value.casefold() in seen:
            raise ValueError("reddit_subreddits subreddits must not contain canonical duplicates")
        seen.add(value.casefold())
        canonical.append(value)
    modes = spec.get("modes", ["submissions"])
    if (
        not isinstance(modes, list)
        or not modes
        or any(mode not in {"submissions", "comments"} for mode in modes)
    ):
        raise ValueError("reddit_subreddits modes must contain submissions and optional comments")
    if len(set(modes)) != len(modes):
        raise ValueError("reddit_subreddits modes must not contain duplicates")
    if "comments" in modes and not spec.get("approval_reference"):
        raise ValueError("reddit_subreddits comments require approval_reference")
    for key in ("client_id_env", "client_secret_env"):
        value = spec.get(key)
        if not isinstance(value, str) or not _ENV_NAME_RE.fullmatch(value):
            raise ValueError(f"reddit_subreddits {key} must be an environment-variable name")
    for key in ("user_agent", "approval_reference"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"reddit_subreddits {key} must be a non-empty string")
    values = {
        "bootstrap_lookback_hours": _validate_int(
            spec, "reddit_subreddits", "bootstrap_lookback_hours", default=48, minimum=1, maximum=48
        ),
        "replay_lookback_hours": _validate_int(
            spec, "reddit_subreddits", "replay_lookback_hours", default=24, minimum=0, maximum=48
        ),
        "retention_hours": _validate_int(
            spec, "reddit_subreddits", "retention_hours", default=48, minimum=1, maximum=48
        ),
        "listing_item_ceiling": _validate_int(
            spec,
            "reddit_subreddits",
            "listing_item_ceiling",
            default=1000,
            minimum=1,
            maximum=10000,
        ),
        "request_timeout_seconds": _validate_int(
            spec, "reddit_subreddits", "request_timeout_seconds", default=30, minimum=1, maximum=300
        ),
        "run_timeout_seconds": _validate_int(
            spec, "reddit_subreddits", "run_timeout_seconds", default=300, minimum=1, maximum=3600
        ),
    }
    if (
        values["bootstrap_lookback_hours"] > values["retention_hours"]
        or values["replay_lookback_hours"] > values["retention_hours"]
    ):
        raise ValueError(
            "reddit_subreddits bootstrap/replay lookback cannot exceed retention_hours"
        )
    return RedditSubredditSettings(
        profile,
        tuple(canonical),
        tuple(modes),
        spec["client_id_env"],
        spec["client_secret_env"],
        spec["user_agent"],
        spec["approval_reference"],
        **values,
    )


def _validate_paperpile_spec(spec: dict) -> None:
    _validate_source_keys(spec, "paperpile", _PAPERPILE_KEYS)
    profile = spec.get("profile")
    if not isinstance(profile, str) or not _HN_PROFILE_RE.fullmatch(profile):
        raise ValueError("paperpile profile must match ^[a-z0-9][a-z0-9_-]*$")
    path = spec.get("db_path")
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError("paperpile db_path must be an absolute non-empty string")
    _validate_int(spec, "paperpile", "staleness_hours", default=48, minimum=1, maximum=8760)


def _validate_youtube_playlist_spec(spec: dict) -> None:
    _validate_source_keys(spec, "youtube_playlist", _YOUTUBE_PLAYLIST_KEYS)
    playlist_id = spec.get("playlist_id")
    if not isinstance(playlist_id, str) or not playlist_id.strip():
        raise ValueError("youtube_playlist playlist_id must be a non-empty string")
    api_key_env = spec.get("api_key_env")
    if not isinstance(api_key_env, str) or not _ENV_NAME_RE.fullmatch(api_key_env):
        raise ValueError("youtube_playlist api_key_env must be an environment-variable name")
    _validate_int(
        spec, "youtube_playlist", "max_api_units_per_run", default=0, minimum=1, maximum=100000
    )
    _validate_int(
        spec, "youtube_playlist", "metadata_refresh_days", default=28, minimum=1, maximum=28
    )
    mode = spec.get("transcript_mode", "off")
    if mode not in {"off", "unofficial_captions"}:
        raise ValueError("youtube_playlist transcript_mode must be 'off' or 'unofficial_captions'")
    languages = spec.get("transcript_languages", [])
    if not isinstance(languages, list) or any(
        not isinstance(value, str) or not value for value in languages
    ):
        raise ValueError("youtube_playlist transcript_languages must be non-empty strings")
    if (mode == "off" and languages) or (mode == "unofficial_captions" and not languages):
        raise ValueError("youtube_playlist transcript mode and languages are incompatible")
    prefer_manual = spec.get("prefer_manual_transcript", True)
    if not isinstance(prefer_manual, bool):
        raise ValueError("youtube_playlist prefer_manual_transcript must be a boolean")


def _build_chrome_reading_list_source(spec: dict) -> ChromeReadingListSourceConfig | None:
    _validate_source_keys(spec, "chrome_reading_list", _CHROME_READING_LIST_KEYS)
    enabled = spec.get("enabled")
    if not isinstance(enabled, bool):
        raise TypeError("chrome_reading_list enabled must be a boolean")
    if not enabled:
        return None
    profile = spec.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("chrome_reading_list profile must be a non-empty string")
    inbox_path = spec.get("inbox_path")
    if not isinstance(inbox_path, str) or not inbox_path or not Path(inbox_path).is_absolute():
        raise ValueError("chrome_reading_list inbox_path must be an absolute non-empty string")
    minimum_version = spec.get("minimum_chrome_version")
    if (
        isinstance(minimum_version, bool)
        or not isinstance(minimum_version, int)
        or minimum_version < MINIMUM_CHROME_VERSION
    ):
        raise ValueError(
            f"chrome_reading_list minimum_chrome_version must be an integer >= "
            f"{MINIMUM_CHROME_VERSION}"
        )
    return ChromeReadingListSourceConfig(profile, inbox_path, minimum_version)


def _validate_source_keys(spec: dict, kind: str, allowed: frozenset[str]) -> None:
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown {kind} source keys: {', '.join(sorted(unknown))}")


def _validate_token_env(spec: dict, kind: str) -> None:
    token_env = spec.get("token_env")
    if not isinstance(token_env, str) or not _ENV_NAME_RE.fullmatch(token_env):
        raise ValueError(f"{kind} token_env must be an environment-variable name")


def _validate_hn_profiles(specs: list[dict]) -> None:
    profiles: set[str] = set()
    for spec in specs:
        if spec.get("type") != "hacker_news":
            continue
        profile = spec.get("profile")
        if profile in profiles:
            raise ValueError(f"duplicate host-applicable hacker_news profile: {profile!r}")
        profiles.add(profile)


def _validate_paperpile_profiles(specs: list[dict]) -> None:
    profiles: set[str] = set()
    for spec in specs:
        if spec.get("type") != "paperpile":
            continue
        profile = spec.get("profile")
        if profile in profiles:
            raise ValueError(f"duplicate host-applicable paperpile profile: {profile!r}")
        profiles.add(profile)


def _validate_hf_daily_papers_specs(specs: list[dict]) -> None:
    if sum(spec.get("type") == "hf_daily_papers" for spec in specs) > 1:
        raise ValueError("multiple host-applicable hf_daily_papers sources are not allowed")


def _validate_reddit_subreddit_profiles(specs: list[dict]) -> None:
    profiles: set[str] = set()
    for spec in specs:
        if spec.get("type") != "reddit_subreddits" or not spec.get("enabled"):
            continue
        profile = spec.get("profile")
        if profile in profiles:
            raise ValueError(f"duplicate host-applicable reddit_subreddits profile: {profile!r}")
        profiles.add(profile)


def _validate_disabled_reddit_subreddit_specs(specs: list[dict]) -> None:
    for spec in specs:
        if spec.get("type") != "reddit_subreddits" or spec.get("enabled") is True:
            continue
        _validate_source_keys(spec, "reddit_subreddits", _REDDIT_SUBREDDITS_KEYS)
        if spec.get("enabled") is not False:
            raise TypeError("reddit_subreddits enabled must be a boolean")


def _validate_youtube_playlist_specs(specs: list[dict]) -> None:
    playlist_ids: set[str] = set()
    for spec in specs:
        if spec.get("type") != "youtube_playlist":
            continue
        playlist_id = spec.get("playlist_id")
        if playlist_id in playlist_ids:
            raise ValueError(
                f"duplicate host-applicable youtube_playlist playlist_id: {playlist_id!r}"
            )
        playlist_ids.add(playlist_id)


def _validate_rss_atom_profiles(specs: list[dict]) -> None:
    profiles: set[str] = set()
    for spec in specs:
        if spec.get("type") != "rss_atom":
            continue
        profile = spec.get("profile")
        if profile in profiles:
            raise ValueError(f"duplicate host-applicable rss_atom profile: {profile!r}")
        profiles.add(profile)


def _source_applies_to_host(spec: dict, ai_host: str | None) -> bool:
    host = spec.get("host")
    return host is None or host == ai_host


_LOCAL_RUNTIMES = frozenset({"ollama", "llama-server", "llama.cpp", "mlx_lm", "mlx-lm"})
_OLLAMA_PREFIX = ("env", "OLLAMA_HOST=127.0.0.1:11434", "ollama", "run")


def _uses_local_runtime(argv: object) -> bool:
    return isinstance(argv, list) and any(
        isinstance(part, str) and part.rsplit("/", maxsplit=1)[-1] in _LOCAL_RUNTIMES
        for part in argv
    )


def _validate_local_llm_by_host(data: object) -> dict[str, dict]:
    """Validate the only supported host-local transport before resolving any command."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("local_llm_by_host must be a host-to-command object")
    validated: dict[str, dict] = {}
    for host, spec in data.items():
        if not isinstance(host, str) or not isinstance(spec, dict):
            raise TypeError("local_llm_by_host entries must be host names mapped to objects")
        command = spec.get("command")
        if spec.get("backend") != "ollama" or not isinstance(command, list):
            raise ValueError(
                f"local_llm_by_host[{host!r}] must use backend='ollama' with a loopback command"
            )
        if (
            tuple(command[:4]) != _OLLAMA_PREFIX
            or len(command) != 5
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ValueError(
                f"local_llm_by_host[{host!r}].command must be exactly "
                "['env', 'OLLAMA_HOST=127.0.0.1:11434', 'ollama', 'run', '<model>']"
            )
        if not isinstance(spec.get("model_id"), str) or not spec["model_id"]:
            raise ValueError(f"local_llm_by_host[{host!r}] requires a non-empty model_id")
        validated[host] = dict(spec)
        validated[host]["command"] = list(command)
    return validated


def _validate_shared_commands(data: dict) -> None:
    commands = data.get("llm_commands") or (
        [] if not data.get("llm_command") else [data["llm_command"]]
    )
    for command in commands:
        if _uses_local_runtime(command):
            raise ValueError(
                "local runtime commands must be moved into local_llm_by_host; "
                "shared llm_command/llm_commands tiers must be host-agnostic"
            )


def _build_llm_command(
    data: dict, timeout: float, *, ai_host: str | None, local_llm_by_host: dict[str, dict]
) -> LlmCommand | None:
    """Resolve the D-3 ladder, prepending only this invocation host's validated local tier."""
    _validate_shared_commands(data)
    ladder = list(data.get("llm_commands") or [])
    local = local_llm_by_host.get(ai_host) if ai_host is not None else None
    if local is not None:
        # A legacy singular llm_command has no ladder to prepend onto — fold it in as the
        # one shared fallback tier so it is not silently dropped when a local tier exists.
        if not ladder and data.get("llm_command"):
            ladder = [data["llm_command"]]
        ladder.insert(0, local["command"])
    if ladder:
        return FallbackLlmCommand([SubprocessLlmCommand(argv, timeout=timeout) for argv in ladder])
    if data.get("llm_command"):
        return SubprocessLlmCommand(data["llm_command"], timeout=timeout)
    return None


def _database_config(data: dict) -> DatabaseConfig:
    raw = data.get("database", {})
    if not isinstance(raw, dict):
        raise TypeError("database must be an object")
    unknown = set(raw) - {"dsn"}
    if unknown:
        raise ValueError(f"database has unknown fields: {', '.join(sorted(unknown))}")
    dsn = raw.get("dsn")
    if dsn is not None and (not isinstance(dsn, str) or not dsn.strip()):
        raise ValueError("database.dsn must be a non-empty connection string")
    return DatabaseConfig(dsn=dsn)


def _storage_config(data: dict) -> StorageConfig:
    raw = data.get("storage", {})
    if not isinstance(raw, dict):
        raise TypeError("storage must be an object")
    unknown = set(raw) - {"backend"}
    if unknown:
        raise ValueError(f"storage has unknown fields: {', '.join(sorted(unknown))}")
    backend = raw.get("backend", "sqlalchemy")
    if backend != "sqlalchemy":
        raise ValueError("storage.backend must be 'sqlalchemy'")
    return StorageConfig(backend=backend)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a pipeline config from a JSON file. Every tunable has a default and any knob can be
    overridden, with one required exception: ``classifier_escalation_alarm_threshold`` (D-15) must
    be explicitly set in ``[0, 1]`` so an operator makes a conscious alarm-sensitivity choice
    rather than silently inheriting a default."""
    config_path = Path(path)
    data = json.loads(config_path.read_text())
    database = _database_config(data)
    storage = _storage_config(data)
    alarm_threshold = data.get("classifier_escalation_alarm_threshold")
    if (
        isinstance(alarm_threshold, bool)
        or not isinstance(alarm_threshold, (int, float))
        or not 0 <= alarm_threshold <= 1
    ):
        raise ValueError("classifier_escalation_alarm_threshold must be a number in [0, 1]")
    llm_timeout = data.get("llm_timeout", 120.0)
    ai_host = os.environ.get("AI_HOST") or None
    local_llm_by_host = _validate_local_llm_by_host(data.get("local_llm_by_host"))
    target_templates = _target_templates(data)
    auto_categories = data.get("auto_mappable_categories", [])
    if not isinstance(auto_categories, list) or not all(
        isinstance(category, str) and category for category in auto_categories
    ):
        raise ValueError(
            "auto_mappable_categories must contain non-empty canonical category strings"
        )
    if len(auto_categories) != len(set(auto_categories)):
        raise ValueError("auto_mappable_categories must not contain duplicates")
    auto_create = data.get("auto_create_mapping", False)
    if not isinstance(auto_create, bool):
        raise TypeError("auto_create_mapping must be a boolean")
    mining_enabled = data.get("backlog_mining_enabled", False)
    if not isinstance(mining_enabled, bool):
        raise TypeError("backlog_mining_enabled must be a boolean")
    cc_task_creation_data = data.get("cc_task_creation", {})
    if not isinstance(cc_task_creation_data, dict):
        raise TypeError("cc_task_creation must be an object")
    cc_task_creation_enabled = cc_task_creation_data.get("enabled", False)
    if not isinstance(cc_task_creation_enabled, bool):
        raise TypeError("cc_task_creation.enabled must be a boolean")
    mapping_threshold = data.get("auto_create_mapping_confidence_threshold", 1.0)
    if (
        isinstance(mapping_threshold, bool)
        or not isinstance(mapping_threshold, (int, float))
        or not 0 <= mapping_threshold <= 1
    ):
        raise ValueError("auto_create_mapping_confidence_threshold must be a number in [0, 1]")
    default_target = data.get("default_target_template")
    default_template = (
        _target_template(default_target, field_name="default_target_template")
        if default_target is not None
        else None
    )
    if auto_create and default_template is None:
        raise ValueError("auto_create_mapping requires default_target_template")
    actionability_data = dict(data)
    if "actionability_llm_commands" in data:
        actionability_data["llm_commands"] = data["actionability_llm_commands"]
    elif "actionability_llm_command" in data:
        actionability_data["llm_command"] = data["actionability_llm_command"]
    applicable_sources = [
        spec for spec in data.get("sources", []) if _source_applies_to_host(spec, ai_host)
    ]
    chrome_reading_list_sources = [
        source
        for spec in applicable_sources
        if spec.get("type") == "chrome_reading_list"
        if (source := _build_chrome_reading_list_source(spec)) is not None
    ]
    _validate_hn_profiles(applicable_sources)
    _validate_paperpile_profiles(applicable_sources)
    _validate_hf_daily_papers_specs(applicable_sources)
    _validate_disabled_reddit_subreddit_specs(applicable_sources)
    _validate_reddit_subreddit_profiles(applicable_sources)
    reddit_specs = [
        spec
        for spec in applicable_sources
        if spec.get("type") == "reddit_subreddits" and spec.get("enabled")
    ]
    from rumen.reddit_retention import RedditRetentionMaintainer

    reddit_adapters = [
        RedditSubredditAdapter(_validate_reddit_subreddits_spec(spec)) for spec in reddit_specs
    ]
    _validate_youtube_playlist_specs(applicable_sources)
    _validate_rss_atom_profiles(applicable_sources)
    _validate_spotify_podcast_profiles(applicable_sources)
    return PipelineConfig(
        database=database,
        storage=storage,
        adapters=[
            *[
                _build_adapter(spec)
                for spec in applicable_sources
                if spec.get("type") not in {"chrome_reading_list", "reddit_subreddits"}
            ],
            *reddit_adapters,
            *[
                ChromeReadingListAdapter(source.inbox_path, source.profile)
                for source in chrome_reading_list_sources
            ],
        ],
        source_maintainers=[
            RedditRetentionMaintainer(adapter.settings, adapter.gateway)
            for adapter in reddit_adapters
        ],
        chrome_reading_list_sources=chrome_reading_list_sources,
        llm_command=_build_llm_command(
            data, llm_timeout, ai_host=ai_host, local_llm_by_host=local_llm_by_host
        ),
        actionability_command_factory=lambda: _build_llm_command(
            actionability_data,
            llm_timeout,
            ai_host=ai_host,
            local_llm_by_host=local_llm_by_host,
        ),
        actionability_max_calls=_validate_int(
            data,
            "backlog_mining",
            "actionability_max_calls",
            default=200,
            minimum=1,
            maximum=100000,
        ),
        ai_host=ai_host,
        local_llm_by_host=local_llm_by_host,
        out_of_scope=frozenset(data.get("out_of_scope", [])),
        digest_top_n=data.get("digest_top_n", 20),
        digest_section_map=data.get("digest_section_map", {}),
        max_llm_calls=data.get("max_llm_calls", 200),
        commit_batch_size=data.get("commit_batch_size", 500),
        classify_write_max_attempts=data.get("classify_write_max_attempts", 3),
        classify_write_retry_backoff_seconds=data.get("classify_write_retry_backoff_seconds", 1.0),
        llm_timeout=llm_timeout,
        classifier_path=data.get("classifier_path"),
        classifier_escalation_alarm_threshold=float(alarm_threshold),
        rights_confirmed=data.get("rights_confirmed", False),
        taxonomy_donors=data.get("taxonomy_donors", []),
        tau_class=data.get("tau_class", 0.75),
        tau_high=data.get("tau_high", 0.92),
        tau_low=data.get("tau_low", 0.80),
        mmr_lambda=data.get("mmr_lambda", 0.5),
        exploration_fraction=data.get("exploration_fraction", 0.12),
        half_life_days=data.get("half_life_days", {}),
        default_half_life_days=data.get("default_half_life_days", 14.0),
        residue_window_days=data.get("residue_window_days", 30),
        taxonomy_audit_crowded_item_threshold=data.get(
            "taxonomy_audit_crowded_item_threshold", 100
        ),
        taxonomy_audit_sparse_item_threshold=data.get("taxonomy_audit_sparse_item_threshold", 0),
        taxonomy_audit_duplicate_similarity_threshold=data.get(
            "taxonomy_audit_duplicate_similarity_threshold", 0.92
        ),
        category_to_target_template=target_templates,
        backlog_mining_enabled=mining_enabled,
        cc_task_creation=CcTaskCreationConfig(enabled=cc_task_creation_enabled),
        auto_create_mapping=auto_create,
        auto_create_mapping_confidence_threshold=float(mapping_threshold),
        auto_mappable_categories=tuple(auto_categories),
        default_target_template=default_template,
        backlog_max_age_days=_validate_int(
            data, "backlog_mining", "backlog_max_age_days", default=30, minimum=1, maximum=3650
        ),
        backlog_max_rank_position=_validate_int(
            data,
            "backlog_mining",
            "backlog_max_rank_position",
            default=20,
            minimum=0,
            maximum=100000,
        ),
        config_path=config_path,
    )
