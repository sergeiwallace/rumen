# Contributing to rumen

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/sergeiwallace/rumen.git
cd rumen
uv sync --extra dev
source .venv/bin/activate
```

`uv sync` makes the environment match the extras named, so name all wanted extras
in one command.

Set an explicit SQLite DSN before exercising commands that open storage:

```bash
export RUMEN_DB_DSN='sqlite:///rumen.db'
```

## Tests and checks

Run focused tests while developing:

```bash
pytest tests/test_config_templates.py
pytest tests/feed/test_cli_feed.py
pytest tests/feed/test_cli_feed.py::test_cli_items_view_filters
```

Before opening a pull request, run the hard gate exactly as follows:

```bash
ruff check src/ tests/ && ruff format --check src/ tests/ && pytest
```

Ruff also supports deliberate local fixes:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

## Pull requests

Fork the repository, branch from `main`, make a focused
change with regression coverage, and open a pull request against protected
`main`. Describe the behavior changed, how it was tested, and any configuration
or migration impact. Maintainers merge only after required checks pass.

The `main` branch is protected on the public repository.

## Public/private boundary

Contribute generic engine behavior upstream: ingestion and adapter contracts,
classification, tagging, grouping, ranking, digests, SQLAlchemy models and
migrations, sanitized examples, and public documentation.

Never contribute credentials, real source lists, personal paths, curated data,
exports, caches, databases, transcripts, deployment settings, private
integration wrappers, or private task-system conventions. Use placeholder
environment-variable names and synthetic fixtures in examples and tests.

## Reporting issues

Use the issue forms for reproducible bugs and feature proposals. For a security
vulnerability, follow [SECURITY.md](SECURITY.md) instead of opening a public
issue.
