# Extending rumen

## Source adapter

The contracts live in [base.py](../src/rumen/adapters/base.py). An append-only
adapter implements `source_name() -> str` and
`fetch_new(since: datetime | None) -> list[RawItem]`. The source name must be stable
and distinguish independently configured sources. Use stable source IDs;
ingestion deduplicates `(source, source_id)`. Return candidates at or after the
inclusive cursor, or all candidates for `None`. Use timezone-aware UTC timestamps.

`RawItem` requires `source`, `source_id`, `url`, `title`, `published_at`. Optional
fields are `content`, `author`, `category`, `categories`, `domains`, and a
`metadata` dictionary. Emit source-native information only; classification,
embedding, identity grouping and ranking happen later. Do not include credentials
in metadata or exceptions, since ingestion can persist both.

This minimal offline adapter and regression test exercise the real ingestion entry
point. Save the block as `tests/test_example_adapter.py` in your fork and run
`pytest tests/test_example_adapter.py` after installing the `dev` extra.

```python
from datetime import UTC, datetime

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from rumen.adapters.base import RawItem
from rumen.db.migrate import upgrade_to_head
from rumen.ingest import run_ingestion
from rumen.models import FeedItem


class ExampleAdapter:
    def source_name(self):
        return "example:offline"

    def fetch_new(self, since):
        item = RawItem(
            source=self.source_name(),
            source_id="one",
            url="https://example.com/one",
            title="Example item",
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return [item] if since is None or item.published_at >= since else []


def test_replayed_source_when_ingested_twice_then_one_item_remains():
    engine = create_engine("sqlite://")
    upgrade_to_head(engine)
    with Session(engine) as session:
        adapter = ExampleAdapter()
        assert run_ingestion(session, adapter).status == "completed"
        assert run_ingestion(session, adapter).status == "completed"
        assert session.scalar(select(func.count()).select_from(FeedItem)) == 1
    engine.dispose()
```

To enable JSON configuration, move the class into your adapter module, import it
in [config.py](../src/rumen/config.py), and add a branch to `_build_adapter`:

```python
if kind == "example":
    _validate_source_keys(spec, "example", frozenset({"type", "host"}))
    return ExampleAdapter()
```

Then a config with `"sources": [{"type": "example"}]` selects it. Keep the
required alarm threshold and database settings from the example config. Add a
config-loader test for accepted settings, unknown fields and invalid values, plus
ingestion tests for empty results, cursor boundaries, failures and replay. Supply
a sanitized template if you contribute the adapter upstream. No dynamic plugin
registry is installed; registration is an explicit builder change.

For mutable sources, implement `ChangeSourceAdapter.fetch_changes(since)` returning
`SourceChange` records: `upsert` requires a `RawItem`; `tombstone` requires no item.
Both need `source_id`, `occurred_at`, and a nonempty `revision_digest`.
For paginated/replay sources, implement `StatefulSourceAdapter` and return a
`FetchResult(items, next_state, diagnostics)` from `fetch_new_with_state`.
State and diagnostics must be bounded JSON objects; see
[ingest.py](../src/rumen/ingest.py) for the limits. Only completed runs advance
saved cursors/state; a failed adapter records a failed run and retries later.

### Network fetching

Every adapter fetching publisher-controlled URLs must use
[net_safety.py](../src/rumen/net_safety.py), specifically `open_public_https`.
Its context manager yields `(response, final_url)`. Supply request headers and a
finite timeout. It validates HTTPS syntax, resolves and pins public IP addresses,
keeps hostname certificate verification, and revalidates redirects. Do not replace
it with a separate preflight check followed by an ordinary HTTP client: that
reopens DNS and redirect races. Use `url_policy` when a source needs additional
host/path restrictions. Bound response bytes, pagination and total work in the
adapter; the helper does not impose a response-size limit. Test blocked local
addresses, mixed DNS answers, redirects and oversized payloads without real network
access, following [the safety tests](../tests/feed/test_public_https_fetches.py).

## Taxonomy

Install the `taxonomy` extra. Author SKOS/RDF with stable concept URIs,
`skos:ConceptScheme`, `skos:Concept`, `skos:inScheme` and `skos:prefLabel`.
Each scheme needs a `https://rumen.dev/ns#facet` value; optional
`https://rumen.dev/ns#schemeVersion` and a Dublin Core identifier retain provenance.
Link concepts with SKOS broader/narrower/related relations. Validation rejects
cycles, duplicate labels, orphan concepts and disconnected components.

Use [load_skos](../src/rumen/taxonomy/skos_store.py),
[ensure_valid](../src/rumen/taxonomy/validate.py) and
[materialize_scheme](../src/rumen/taxonomy/materialize.py) in an application-owned
loader. There is no taxonomy-import CLI command. A local Turtle file can be
materialized as follows after setting an explicit database DSN:

```python
from pathlib import Path
from sqlalchemy.orm import Session
from rumen.db.engine import get_engine
from rumen.db.migrate import upgrade_to_head
from rumen.taxonomy.skos_store import load_skos
from rumen.taxonomy.validate import ensure_valid
from rumen.taxonomy.materialize import materialize_scheme

engine = get_engine()
upgrade_to_head(engine)
graph = load_skos(Path("taxonomy.ttl"), format="turtle")
ensure_valid(graph)
with Session(engine) as session:
    result = materialize_scheme(session, graph)
    session.commit()
print(result)
```

Materialization adds missing IDs and relations idempotently; it does not replace
existing rows on label edits. Use reviewed audit/revision APIs for changes.
Donor grafting additionally requires verified reuse terms and crosswalk provenance.
The engine's MIT license does not relicense imported taxonomy data. See
[materialization tests](../tests/taxonomy/test_materialize.py) for minimal graphs.

## LLM command

Implement a program that reads the prompt from stdin and returns model output on
stdout. Keep diagnostics on stderr, exit nonzero on failure and honor the caller's
timeout. The argv list is executed without a shell; metacharacters and environment
expansion are not interpreted. Install and authenticate your provider separately.

For an offline transport smoke test, save this as `model_bridge.py`:

```python
import json
import sys

prompt = sys.stdin.read()
print(json.dumps({"received_characters": len(prompt)}))
```

Set `"llm_command": ["python", "model_bridge.py"]` in your config. Test transport
with `SubprocessLlmCommand(["python", "model_bridge.py"]).run("hello")`; the result
is `{"received_characters": 5}` plus a newline. This smoke bridge is not a
classifier: a real command must return the JSON schema requested in each prompt.
[LLM parsing and fallback code](../src/rumen/llm.py) tolerates JSON fences and
tries fallback commands after subprocess failures. Invalid successful output is
handled by the calling classifier/digest, not by the transport fallback ladder.
Local Ollama commands belong in `local_llm_by_host`, as described in
[configuration](configuration.md#llm-commands).
