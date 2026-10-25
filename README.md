# rumen

![CI](https://github.com/sergeiwallace/rumen/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)

Self-contained rumen engine for turning content feeds into a useful,
ranked digest. It owns its SQLAlchemy storage and runs with SQLite or Postgres.

## Contents

- [What it does](#what-it-does)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick start: SQLite](#quick-start-sqlite)
- [Configuration, databases, and adapters](#configuration-databases-and-adapters)
- [Adapters](#adapters)
- [CLI](#cli)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Configuration reference](docs/configuration.md)
- [Environment reference](docs/environment.md)
- [Extending rumen](docs/extending.md)
- [Data model](docs/data-model.md)
- [Limitations](#limitations)
- [Security](#security)
- [Contributing](#contributing)
- [Provenance](#provenance)
- [License](#license)

## What it does

rumen processes configured sources through one pipeline:

1. ingest content from source adapters;
2. classify it and apply keyword tags;
3. group items by stable identity and near-duplicate relationships;
4. rank the resulting topic entities; and
5. produce periodic digests.

The engine includes adapters and configuration templates for browser bookmarks,
GitHub, Hacker News, Hugging Face Daily Papers, npm, Reddit, RSS/Atom, YouTube,
and local cache-based sources. Sources are opt-in: the shipped example config has
none enabled.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A SQLite file for a local trial, or a reachable Postgres database

## Installation

Clone the repository once, then choose an installation method. These commands
use a POSIX shell; on Windows use the virtual environment's activation command.

```bash
git clone https://github.com/sergeiwallace/rumen.git
cd rumen
```

With uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
rumen --help
```

With Python's pip, in a fresh environment:

```bash
python -m venv .venv-pip
source .venv-pip/bin/activate
python -m pip install -e .
rumen --help
```

The editable install is for a checkout you can build on. Remove `-e` for a regular
installation from the checkout. Install only the optional features you need.
For the development checks, taxonomy APIs, RSS and Reddit support:

```bash
uv pip install -e '.[dev,taxonomy,feeds,reddit]'
```

The same extras syntax works with `python -m pip install` in the pip environment.

| Extra | Enables |
| --- | --- |
| `dev` | pytest, pytest-xdist and pinned Ruff checks |
| `taxonomy` | RDF/SKOS loading, validation and materialization with rdflib and skosify |
| `feeds` | RSS/Atom adapter parsing with feedparser |
| `reddit` | Subreddit API adapter with PRAW; saved-items cache reading needs no extra |
| `embeddings` | SentenceTransformer semantic embeddings; heavy model dependencies and first-use model download |
| `hmc-fit` | Optional cleanlab quality checks for hierarchical classifier fitting |
| `local-llm` | Empty discoverability extra; install and configure the external Ollama runtime separately |
| `youtube-transcripts` | Opt-in unofficial caption enrichment; official playlist discovery needs no extra |

Weak-supervision bootstrap is deferred from this release because Snorkel has
unresolved security advisories with no fixed release. It is neither declared nor
loaded, even if installed separately. Its former API entry points raise a clear
unavailable-feature error. Use curated labels or the independent
`rumen.taxonomy.bootstrap.llm_teacher.run_teacher_pass` API instead. Restoration
requires a fixed dependency and a fresh security review.

Taxonomy imports accept local files, inline RDF, and JSON-LD with embedded
contexts. URL sources and remote JSON-LD contexts (including `@import`) are
refused before network access. Download documents separately and embed their
contexts before importing.

## Quick start: SQLite

After either installation method, run this from the repository root with its
virtual environment active. It creates an explicit local database and runs with
no sources. CLI startup applies migrations. No credentials or LLM are needed.

```bash
export RUMEN_DB_DSN='sqlite:///rumen.db'
cp config.example.json config.local.json
rumen run --config config.local.json
rumen feed status
rumen feed items --source rss_atom:example --limit 20
rumen digest --config config.local.json
rumen feed digest
```

An empty source list produces no ingestion rows or digest content. Add a source
using the [tested template merge](docs/configuration.md), replace
its placeholders and run again. Rumen never chooses SQLite implicitly. Set
`RUMEN_DB_DSN` or `database.dsn`; the environment takes precedence.

## Configuration, databases, and adapters

[Configuration](docs/configuration.md) lists every loader setting, default,
validation rule and template-merge procedure. [Environment](docs/environment.md)
covers credentials, database connection controls, logging and `AI_HOST` routing.
Templates are JSON fragments with source entries, not files to copy verbatim over
your whole config. Credentials are supplied through named environment variables.

For Postgres, set `RUMEN_DB_DSN` to your connection URL, optionally set
`RUMEN_DB_SCHEMA` for schema isolation, then use the same CLI commands. Only the
SQLAlchemy storage backend is supported. An unreachable database raises an error;
it does not fall back to SQLite.

## Adapters

Every source is opt-in. Cache readers require existing compatible upstream data;
they do not synchronize those services themselves. The two disabled templates
need additional settings before enabling. See [source settings](docs/configuration.md).

| Source type | Input | Template | Extra |
| --- | --- | --- | --- |
| `bookmarks` | Local bookmark cache | [bookmarks](config/templates/bookmarks.json) | Base |
| `chrome_bookmarks` | Local browser bookmarks JSON | [chrome_bookmarks](config/templates/chrome_bookmarks.json) | Base |
| `chrome_reading_list` | Local connector snapshot; disabled template | [chrome_reading_list](config/templates/chrome_reading_list.json) | Base; browser connector not bundled |
| `github_releases` | Repository release Atom feed | [github_releases](config/templates/github_releases.json) | Base |
| `github_repository` | Repository metadata API | [github_repository](config/templates/github_repository.json) | Base |
| `github_stars` | Authenticated user's stars | [github_stars](config/templates/github_stars.json) | Base |
| `github_stars_list` | Named star list | [github_stars_list](config/templates/github_stars_list.json) | Base |
| `hacker_news` | Search queries | [hacker_news](config/templates/hacker_news.json) | Base |
| `hf_daily_papers` | Daily paper listings | [hf_daily_papers](config/templates/hf_daily_papers.json) | Base |
| `npm_registry` | Package registry releases | [npm_registry](config/templates/npm_registry.json) | Base |
| `paperpile` | Existing local library cache | [paperpile](config/templates/paperpile.json) | Base |
| `reddit_saved` | Existing saved-items cache | [reddit_saved](config/templates/reddit_saved.json) | Base |
| `reddit_subreddits` | Subreddit API; disabled template | [reddit_subreddits](config/templates/reddit_subreddits.json) | `reddit` |
| `rss_atom` | Public HTTPS RSS/Atom feed | [rss_atom](config/templates/rss_atom.json) | `feeds` |
| `spotify_podcasts` | Existing local podcast cache | [spotify_podcasts](config/templates/spotify_podcasts.json) | Base |
| `youtube_playlist` | Official playlist API | [youtube_playlist](config/templates/youtube_playlist.json) | Base; `youtube-transcripts` for optional captions |

## CLI

`rumen --help` lists commands; each command accepts `--help`. The main public
operations are `run` (ingest through ranking), `digest` (generate), and `feed`
(`items`, `status`, `digest`, `rate`). `classify review` records human facet
assignments and `classify metrics` shows telemetry. `fit-classifier` fits bundles;
`taxonomy-audit` runs taxonomy maintenance. `reddit purge` is an explicit,
confirmed deletion operation. Optional `backlog` operations require application
integration for external task-store access and projection.

`run --digest` generates a digest after processing; `--digest-period-days` selects
the window. `digest --period-start` and `--period-end` accept explicit boundaries;
`--force` permits regeneration. The default digest is extractive. `--abstractive`
requires a configured LLM command; `--local-polish` uses a matching local model.
See [LLM configuration](docs/configuration.md) and the
[extension contracts](docs/extending.md).

## Project layout

| Path | Contents |
| --- | --- |
| [src/rumen/cli.py](src/rumen/cli.py) | CLI entry point |
| [src/rumen/config.py](src/rumen/config.py) | JSON loader and adapter builder |
| [src/rumen/adapters/base.py](src/rumen/adapters/base.py) | Normalized items and adapter protocols |
| [src/rumen/jobs.py](src/rumen/jobs.py) | Pipeline orchestration |
| [src/rumen/models.py](src/rumen/models.py) | SQLAlchemy tables; see [data model](docs/data-model.md) |
| [src/rumen/db/migrate.py](src/rumen/db/migrate.py) | Alembic migration runner; revisions live alongside it |
| [src/rumen/embeddings.py](src/rumen/embeddings.py) | Embedding provider seam |
| [src/rumen/llm.py](src/rumen/llm.py) | Subprocess LLM seam |
| [config.example.json](config.example.json) | Minimal configuration; adapter templates are listed above |
| [tests/feed/test_ingest.py](tests/feed/test_ingest.py) | Ingestion tests; other tests cover feed, taxonomy and backlog behavior |
| [pyproject.toml](pyproject.toml) | Dependencies, extras and tool settings |

## Tests

Install the development extras shown above, then run from the repository root:

```bash
pytest tests/feed/test_ingest.py tests/test_config_templates.py
ruff check src/ tests/ && ruff format --check src/ tests/ && pytest
```

The second line is the hard gate. Postgres tests skip without
`RUMEN_TEST_PG_DSN`. Real-model embedding tests are opt-in; heavier extras have
separate optional tests. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution
expectations and [environment](docs/environment.md) for test controls.

## Limitations

- No source is enabled by default; you must configure and authorize each source.
- Optional adapters require their corresponding extras, such as `.[feeds]` for
  RSS/Atom and `.[reddit]` for Reddit.
- The default digest is extractive. Abstractive digest generation requires an
  explicitly configured local LLM command.
- SQLite is suitable for local use; use Postgres for shared or production
  deployments.

## Security

Please do not report vulnerabilities in public issues. Follow the private
reporting process in [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for setup, tests, pull-request expectations, and the public/private boundary.

## Provenance

The public repository history begins at a clean root commit. The release policy is
that private pre-launch history will not be published.

## License

rumen is licensed under the [MIT License](LICENSE).
