# Environment reference

These are all environment reads in the public runtime (`src/rumen`): eight fixed
names and four configurable credential-name fields. Values are inherited by LLM
subprocesses. Do not log credentials or commit a populated environment file.

| Variable | Default | Purpose and commands |
| --- | --- | --- |
| `RUMEN_DB_DSN` | absent | Explicit SQLite/Postgres URL for every storage command; overrides `database.dsn`. Required if config supplies no DSN. `--help` does not open storage. |
| `RUMEN_DB_SCHEMA` | absent | Postgres schema isolation for every storage command; simple SQL identifier only. Created if absent, with search_path pinned. Ignored for SQLite. |
| `RUMEN_DB_KEEPALIVE_IDLE` | `30` | Integer seconds before Postgres TCP keepalive probes; every Postgres storage command. |
| `RUMEN_DB_STATEMENT_TIMEOUT_MS` | `300000` | Postgres statement timeout in milliseconds; `0` or empty disables it. Every Postgres storage command. |
| `RUMEN_CONFIG` | absent | Default config path for `run`, `digest`, `classify review`, `classify metrics`, `fit-classifier`, `taxonomy-audit`, `reddit purge`, `backlog mine`, `backlog review`, and storage initialization for feed/backlog views. Explicit `--config` takes precedence where supported. |
| `RUMEN_LOG_LEVEL` | `INFO` | Python logging level for every CLI invocation. Use `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `AI_HOST` | absent | Operator-chosen routing label when loading config. Filters source `host` and selects `local_llm_by_host`. Unset means only host-independent sources and no local tier. Used by config-loading commands, including `run` and `digest`. |
| `RUMEN_CC_SESSION` | empty | Fallback invoking-session identifier for optional `backlog review` projection. `--cc-session` overrides it. Projection requires an application-supplied integration. |
| Name in `token_env` | no default name or value | GitHub repository, stars and star-list credentials during `run`. Templates use `EXAMPLE_GITHUB_TOKEN`. Missing value raises an adapter error. |
| Name in `api_key_env` | no default name or value | Official YouTube playlist API key during `run`. Template uses `EXAMPLE_YOUTUBE_API_KEY`. Missing value raises an adapter error. |
| Name in `client_id_env` | no default name or value | Reddit API client ID during `run` for enabled subreddit sources. Example name: `EXAMPLE_REDDIT_CLIENT_ID`. |
| Name in `client_secret_env` | no default name or value | Reddit API client secret during `run` for enabled subreddit sources. Example name: `EXAMPLE_REDDIT_CLIENT_SECRET`. |

The validated local Ollama argv sets `OLLAMA_HOST=127.0.0.1:11434` for that child
process. Rumen does not read this as a parent-process environment variable.
`PATH` is used by the operating system to find configured subprocess binaries.
The network safety helper opens direct connections without consulting proxy variables.

The public tests additionally read `RUMEN_TEST_PG_DSN` (absent: skip Postgres
tests) and `RUMEN_TEST_MINILM` (absent: skip the real-model embedding test).
The latter also needs the `embeddings` extra and model availability.

Retired environment spellings are rejected rather than treated as aliases. Use
only the current names above. Private operational scripts and benchmark scripts
are outside the public package and are not required to run the CLI.

See [configuration](configuration.md) for JSON settings and credential-name validation.
