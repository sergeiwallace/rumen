# Configuration reference

Start with [config.example.json](../config.example.json). Pass its copy with
`rumen run --config config.local.json`, or set `RUMEN_CONFIG` for other commands.
Paths are relative to the process working directory, unless an adapter requires
an absolute path. JSON has no comments; `_comment` in the example is ignored.
Unknown top-level keys are not generally rejected, so check spelling carefully.

The tables cover all 31 direct `data.get` keys in `load_config`, plus keys read
through helpers, indexed access and validation functions. Types describe the
intended JSON input. “Unchecked” means the loader passes the value through; it
does not promise that downstream code accepts arbitrary values. Numeric examples
in unchecked rows are conservative operating values, not enforced ranges.

## Database and storage

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `database` | object | `{}` | Only `dsn` is accepted. Connection settings. | `{"dsn":"sqlite:///rumen.db"}` |
| `database.dsn` | string or null | `null` | Nonempty if supplied. Explicit SQLite or Postgres URL; `RUMEN_DB_DSN` overrides it. No implicit database. | `"sqlite:///rumen.db"` |
| `storage` | object | `{}` | Only `backend` is accepted. | `{"backend":"sqlalchemy"}` |
| `storage.backend` | string | `"sqlalchemy"` | Only `sqlalchemy` is supported. | `"sqlalchemy"` |
| `commit_batch_size` | integer | `500` | Unchecked; use a positive count. Rows committed per stage batch for resumability. | `100` |

SQLite needs no server. Postgres uses psycopg3; bare `postgresql://` and
`postgres://` URLs are normalized to that driver. See the
[environment reference](environment.md) for schema and timeout controls.

## Sources

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `sources` | array of objects | `[]` | Each entry selects one adapter. See settings below. | `[{"type":"hf_daily_papers"}]` |
| `sources[].type` | string | required | One of the 16 types in the [README](../README.md#adapters). | `"rss_atom"` |
| `sources[].host` | string or null | `null` | Optional exact match against `AI_HOST`; absent/null applies everywhere. Nonmatching entries are skipped before adapter validation. | `"workstation"` |

Host pins match `AI_HOST` exactly. A mismatch, including an unset or empty
`AI_HOST`, skips the pinned source; unpinned sources still apply everywhere.
Each skip emits a WARNING naming the source type, pin and current host, and
`rumen run` includes `skipped_for_host` in its result JSON as a list of
`{"type":"hf_daily_papers","host":"old-host"}` entries (or `[]` when none).

There is no universal `enabled` flag. Only `chrome_reading_list` and
`reddit_subreddits` support it, and require an explicit boolean. Their shipped
`false` templates need no other fields. Other sources are enabled by adding them
to `sources`. Most adapters reject unknown fields; releases, npm and bookmarks
currently do not. Required fields below have no default.

| Type | Required settings | Optional settings, defaults and limits |
| --- | --- | --- |
| `github_releases` | `repo`: string, e.g. `"example-owner/example-repository"` | None beyond `host`; repo format is not validated by the builder. Reads the release Atom feed. |
| `npm_registry` | `package`: string, e.g. `"@example/package"`; `tool_name`: string, e.g. `"Example Tool"` | No builder range checks. Reads registry releases. |
| `bookmarks` | `db_path`: string, e.g. `"data/bookmarks.db"` | `staleness_hours`: number, default `48`, unchecked. Reads an existing bookmark cache; does not create an upstream sync. |
| `reddit_saved` | `db_path`: nonempty string, e.g. `"data/reddit.db"` | `staleness_hours`: number, default `48`, unchecked. Reads an existing saved-items cache. |
| `github_repository` | `repo`: exactly `owner/repo`; `token_env`: environment name, e.g. `"EXAMPLE_GITHUB_TOKEN"` | Reads repository metadata. |
| `github_stars` | `token_env`: environment name, e.g. `"EXAMPLE_GITHUB_TOKEN"` | Reads authenticated user's stars. |
| `github_stars_list` | `token_env`: environment name; `list_name`: nonempty string, e.g. `"Example List"` | Reads one named star list. |
| `chrome_bookmarks` | `bookmarks_path`: nonempty string, e.g. `"data/Bookmarks"`; `profile`: nonempty string, e.g. `"Default"` | `folder_name`: nonempty string when supplied, default `null` (all folders). Reads a local browser JSON file. |
| `chrome_reading_list` | `enabled`: boolean; if true: `profile`: nonempty string; `inbox_path`: absolute nonempty path; `minimum_chrome_version`: integer >= 120 | Reads a local native-connector snapshot. Example: `{"type":"chrome_reading_list","enabled":false}`. The browser extension and native-host installation are not part of the public package. |
| `hacker_news` | `profile`: lowercase identifier; `queries`: nonempty array of nonblank strings, e.g. `["compiler"]` | `bootstrap_lookback_hours`: integer `72`, range 1..168; `indexing_overlap_minutes`: integer `15`, range 0..1440. Queries normalize whitespace and Unicode and reject case-insensitive duplicates. |
| `hf_daily_papers` | None | `bootstrap_lookback_days`: integer `7`, range 1..90; `daily_recheck_days`: integer `2`, range 0..30. At most one applicable source. |
| `rss_atom` | `profile`: lowercase identifier; `feed_url`: absolute HTTPS URL without credentials or fragment, e.g. `"https://feeds.example.com/updates.xml"` | `min_poll_interval_seconds`: integer `900`, range 60..86400. Requires `feeds` extra. |
| `paperpile` | `profile`: lowercase identifier; `db_path`: absolute nonempty path | `staleness_hours`: integer `48`, range 1..8760. Reads an existing Paperpile cache. |
| `spotify_podcasts` | `profile`: lowercase identifier; `db_path`: nonempty string | `staleness_hours`: positive integer if supplied, default absent (no staleness warning). Reads an existing podcast cache. |
| `youtube_playlist` | `playlist_id`: nonempty string; `api_key_env`: environment name; `max_api_units_per_run`: integer 1..100000 | `metadata_refresh_days`: integer `28`, range 1..28; `transcript_mode`: `"off"` (default) or `"unofficial_captions"`; `transcript_languages`: array of nonempty strings, default `[]`, must be empty for off and nonempty for captions; `prefer_manual_transcript`: boolean `true`. Caption mode requires `youtube-transcripts` extra. |
| `reddit_subreddits` | `enabled`: boolean; enabled settings below | Requires `reddit` extra when fetching. Example: `{"type":"reddit_subreddits","enabled":false}`. |

A lowercase profile matches `^[a-z0-9][a-z0-9_-]*$`. Applicable profiles must be
unique within Hacker News, RSS, Paperpile, Spotify and enabled Reddit sources.
YouTube playlist IDs must be unique. Environment names match
`^[A-Za-z_][A-Za-z0-9_]*$`; put credential values in that variable, never JSON.

Enabled `reddit_subreddits` settings:

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `profile` | string | required | Lowercase identifier. | `"example"` |
| `subreddits` | string array | required | Nonempty, case-insensitively unique names without `r/`, slash or whitespace. | `["example"]` |
| `modes` | string array | `["submissions"]` | Nonempty unique subset of `submissions`, `comments`. Comments require approval reference. | `["submissions"]` |
| `client_id_env` | string | required | Environment variable name. | `"EXAMPLE_REDDIT_CLIENT_ID"` |
| `client_secret_env` | string | required | Environment variable name. | `"EXAMPLE_REDDIT_CLIENT_SECRET"` |
| `user_agent` | string | required | Nonblank API user agent. | `"example-reader/1.0"` |
| `approval_reference` | string | required | Nonblank evidence of permission to ingest. | `"operator-approved"` |
| `bootstrap_lookback_hours` | integer | `48` | 1..48, not above retention. | `24` |
| `replay_lookback_hours` | integer | `24` | 0..48, not above retention. | `12` |
| `retention_hours` | integer | `48` | 1..48, lifecycle retention window. | `48` |
| `listing_item_ceiling` | integer | `1000` | 1..10000, listing cap. | `100` |
| `request_timeout_seconds` | integer | `30` | 1..300 per request. | `30` |
| `run_timeout_seconds` | integer | `300` | 1..3600 per run. | `300` |

## Merge a template

Run this from the repository root after installation. It creates a new config
without overwriting an existing one, appends source entries rather than replacing
them, and retains the base database settings and required alarm threshold.
The disabled Reading List template makes this a fully offline loading example.

```bash
python - <<'PYTHON'
import json
from pathlib import Path
from rumen.config import load_config

base = json.loads(Path("config.example.json").read_text())
template = json.loads(Path("config/templates/chrome_reading_list.json").read_text())
base["sources"].extend(template["sources"])
with Path("config.merged.json").open("x") as output:
    json.dump(base, output, indent=2)
config = load_config("config.merged.json")
assert config.database.dsn == "sqlite:///rumen.db"
assert config.classifier_escalation_alarm_threshold == 0.25
assert config.adapters == []
print("Merged configuration loaded")
PYTHON
```

For a live source, select its template from the [adapter table](../README.md#adapters),
replace placeholders and install its extra before loading. For example, merge the
RSS template and replace `feed_url` with your own public HTTPS feed. Loading a
config constructs adapters; fetching occurs when the pipeline runs.
Cache adapters require compatible existing upstream databases, not arbitrary SQLite files.

## Classifier and grouping thresholds

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `classifier_escalation_alarm_threshold` | number | **required** | Explicit number in [0,1], boolean rejected. Alarm sensitivity for completed facet escalation rate. | `0.25` |
| `classifier_path` | string or null | `null` | Unchecked path to fitted per-facet classifier bundle. Missing bundle leaves deterministic rules and constrained LLM fallback. | `"artifacts/classifier"` |
| `classify_write_max_attempts` | integer | `3` | Unchecked; use positive count. Per-row transient write retries. | `3` |
| `classify_write_retry_backoff_seconds` | number | `1.0` | Unchecked; use nonnegative seconds between retries. | `1.0` |
| `tau_class` | number | `0.75` | Unchecked; confidence gate for classifier acceptance; use [0,1]. | `0.8` |
| `tau_high` | number | `0.92` | Unchecked; cosine residue auto-merge threshold. Use [0,1] and >= tau_low. | `0.95` |
| `tau_low` | number | `0.80` | Unchecked; review-band floor below tau_high. | `0.85` |
| `residue_window_days` | integer | `30` | Unchecked; recent-entity comparison window in days. | `14` |
| `out_of_scope` | string array | `[]` | Converted to a set; excluded category labels. | `["unwanted"]` |

## Ranking and digest

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `digest_top_n` | integer | `20` | Unchecked; top ranked entities considered for digest. | `10` |
| `digest_section_map` | string-to-string object | `{}` | Unchecked; lowercase topic label (legacy category fallback) to section override. | `{"tools":"active_tool_updates"}` |
| `mmr_lambda` | number | `0.5` | Unchecked; relevance/diversity tradeoff, use [0,1]. | `0.6` |
| `exploration_fraction` | number | `0.12` | Unchecked; share of ranking slots for exploration, use [0,1]. | `0.1` |
| `half_life_days` | string-to-number object | `{}` | Unchecked; per-category decay half-lives, use positive days. | `{"tools":7.0}` |
| `default_half_life_days` | number | `14.0` | Unchecked; positive fallback decay half-life. | `14.0` |

Digest sections are `active_tool_updates`, `immediate_techniques` and
`horizon_developments`; use one of these as a section-map value.

## LLM commands

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `llm_command` | string array | absent | Shared subprocess argv, no shell. Local runtime binaries rejected here. | `["python","model_bridge.py"]` |
| `llm_commands` | array of string arrays | `[]` | Ordered shared fallback ladder; nonempty ladder overrides singular command. | `[["python","model_bridge.py"]]` |
| `llm_timeout` | number | `120.0` | Unchecked; seconds per subprocess attempt. | `60.0` |
| `max_llm_calls` | integer | `200` | Unchecked; budget for unexpected Tier-2 routes per run. Expected missing-rule routes are counted separately and are not capped by this value. Consecutive failure breaker still applies. | `100` |
| `local_llm_by_host` | object or null | `{}` | Maps an `AI_HOST` label to the validated local command below. | See below. |

Local entries require `backend: "ollama"`, a nonempty `model_id` string and
exactly `command: ["env","OLLAMA_HOST=127.0.0.1:11434","ollama","run","<model>"]`.
Install the runtime and obtain the model separately. The local tier for the
current `AI_HOST` is prepended to shared fallback commands; no matching label
means no local tier. The label is operator chosen, not inferred from the machine.

```json
{"local_llm_by_host":{"workstation":{"backend":"ollama","model_id":"llama3.1","command":["env","OLLAMA_HOST=127.0.0.1:11434","ollama","run","llama3.1"]}}}
```

A subprocess reads the full prompt on stdin and emits its response on stdout.
Nonzero exit or timeout tries the next fallback; all failures propagate to the
caller. Commands inherit the process environment. Install and authorize any
provider command separately; the extra `local-llm` installs no runtime.
See [extending](extending.md#llm-command) for a tested offline bridge.

## Embeddings

There is no embedding-selection JSON key. Without `embeddings`, the default is
an offline hashing bag-of-words provider. Installing `embeddings` automatically
selects SentenceTransformer `all-MiniLM-L6-v2`; first use can download model weights.
Both produce 384-dimensional float32 vectors. Embedded callers can supply an
`EmbeddingProvider` through `set_default_provider` in
[embeddings.py](../src/rumen/embeddings.py). Keep one embedding space per corpus;
installing a new provider does not re-embed stored vectors automatically.

## Taxonomy

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `rights_confirmed` | boolean | `false` | Unchecked; retained for caller use by the restricted-donor export gate. The loader does not apply it or grant redistribution permission. | `false` |
| `taxonomy_donors` | object array | `[]` | Unchecked at load; donor grafting requires verified reuse terms. | `[]` |
| `taxonomy_audit_crowded_item_threshold` | integer | `100` | Unchecked; crowded concept audit threshold. | `100` |
| `taxonomy_audit_sparse_item_threshold` | integer | `0` | Unchecked; sparse concept audit threshold. | `0` |
| `taxonomy_audit_duplicate_similarity_threshold` | number | `0.92` | Unchecked; duplicate concept similarity audit threshold. | `0.95` |

Donor reuse validation is in [graft.py](../src/rumen/taxonomy/donors/graft.py):
`DonorConfig` requires `name`, `reuse_terms_verified: true` and `reuse_terms_url`.
The loader retains `taxonomy_donors` as data; it does not automatically fetch or
materialize those entries. Use the [taxonomy API](extending.md#taxonomy).

## Optional backlog controls

Post-pipeline mining and session-task projection are disabled by default. The
engine includes a read-only Beads reader and explicit reviewer-driven promotion
through the external `bd` CLI; configure and verify the intended task-store and
repository context before using them. The bundled actionability judge also needs
a configured trusted command backend. Session projection returns a gated plan;
the invoking session, not rumen, executes that plan.

| Key | Type | Default | Validation and purpose | Example |
| --- | --- | --- | --- | --- |
| `backlog_mining_enabled` | boolean | `false` | Explicit opt-in to post-ranking candidate mining. | `false` |
| `category_to_target_template` | object | `{}` | Nonempty category names mapped to target objects below. | `{}` |
| `auto_create_mapping` | boolean | `false` | Requires default_target_template when true. | `false` |
| `auto_create_mapping_confidence_threshold` | number | `1.0` | [0,1], boolean rejected; missing-mapping admission threshold. | `1.0` |
| `auto_mappable_categories` | string array | `[]` | Unique nonempty categories eligible for deterministic auto-mapping. | `["tools"]` |
| `default_target_template` | object or null | `null` | Target used for automatic mapping. | See below. |
| `cc_task_creation` | object | `{}` | `enabled`: boolean, default false; session-local projection opt-in. | `{"enabled":false}` |
| `actionability_llm_command` | string array | inherited `llm_command` | Actionability-specific singular override; shared ladder still has precedence. | `["python","model_bridge.py"]` |
| `actionability_llm_commands` | array of string arrays | inherited `llm_commands` | Actionability-specific fallback override; same local tier and timeout. | `[["python","model_bridge.py"]]` |
| `actionability_max_calls` | integer | `200` | 1..100000, independent judging budget. | `100` |
| `backlog_max_age_days` | integer | `30` | 1..3650, candidate age window. | `30` |
| `backlog_max_rank_position` | integer | `20` | 0..100000, rank eligibility cutoff. | `20` |

A target object has exactly three nonempty string fields: `beads_repo_identity`,
`task_store`, `template_revision`. Identity and revision cannot contain `::`.
A synthetic example is `{"beads_repo_identity":"example","task_store":"queue",
"template_revision":"v1"}`. These fields identify the target; they do not switch
the working directory or route `bd` to another repository or remote store. Verify
the external task-store context and permissions before enabling mining or invoking
promotion. Custom task-store readers and notification delivery can be supplied by
an embedding application.
