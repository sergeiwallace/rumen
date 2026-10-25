# Data model

Rumen owns its SQLAlchemy models and database. SQLite and Postgres share the
schema, with dialect-specific JSON and timestamp handling. Runtime definitions
are in [models.py](../src/rumen/models.py); the
[migration runner](../src/rumen/db/migrate.py) applies Alembic revisions at CLI
startup. Feed item identity is the unique pair `(source, source_id)`.

```text
feed_runs -- source name --> feed_items -- many:1 --> topic_entities
                                   |                     |
                        item_concepts                    +-- feed_feedback
                                   |                     +-- merge_log
                             skos_concepts               +-- merge_candidates (two entities)
                                   |                     |
                             skos_relations       digest_entities -- digests

feed_items/topic_entities <-- logical item reference -- classification_corrections
                                      |
                           classifier_metric_snapshots (pass totals)

topic_entities/digests -- backlog_candidates -- evidence / matches / dispositions
                                                   / task links / promotions
```

Arrows describe stored references, not ORM cascading relationships. Source strings
and polymorphic item references are logical joins without foreign keys.

## Tables

| Table | Purpose and relationships |
| --- | --- |
| `feed_items` | Source pointers, URLs, text, timestamps, classification, tags, opaque embedding and mutable-source revision/deletion markers. Nullable foreign key to `topic_entities`. |
| `topic_entities` | Canonical grouped item, stable identity, display title, relevance and rank. Many source pointers can refer to one entity. |
| `feed_runs` | Per-source ingestion status, high-water mark, state, diagnostics and counts. Source name joins logically to feed items; only completed runs supply cursors. |
| `source_maintenance_runs` | Per-source retention/lifecycle status, checked/purged counts and content-free reasons. Logical source join. |
| `youtube_video_states` | Unique `(playlist_id, video_id)` lifecycle state; unique foreign key to its feed item and optional unique foreign key to a transcript feed item. |
| `skos_concepts` | Materialized controlled vocabulary: facet, preferred label, scheme/version, source URI and status. IDs derived from stable concept URIs. |
| `skos_relations` | Directed edges with two concept foreign keys, relation type and crosswalk provenance. |
| `item_concepts` | Facet assignments, confidence and method. Foreign key to concept; logical `(item_table, item_id)` points to a feed item or entity. Unique item/concept triple. |
| `classification_corrections` | Append-only human review evidence for logical item/facet references; supersedes can reference an earlier correction. |
| `classifier_metric_snapshots` | One unique classification pass ID with cursor, aggregate counters, effective configuration and completion status. |
| `taxonomy_revisions` | Audit mutations with before/after JSON and reversible self-references `reverts` and `reverted_by`. Concept IDs are held in JSON. |
| `digests` | Period boundaries, generated JSON and Markdown, model marker and generation time. |
| `digest_entities` | Many-to-many membership between digests and topic entities, with rank. Unique digest/entity pair. |
| `feed_keywords` | Unique keyword plus include/suppress type. |
| `feed_views` | Unique view name with filter/sort JSON and default flag. |
| `feed_feedback` | Rating (-1/0/1) and foreign key to topic entity. |
| `interest_profiles` | Persisted global embedding centroid, learning alpha and update time. |
| `merge_log` | Auditable grouping action, evidence, confidence, and foreign keys to entity and feed item. |
| `stable_id_alias` | Stable identity redirects; unique `(kind, from_value)`. No entity foreign key. |
| `merge_candidates` | Pending/resolved review queue with two entity foreign keys; unique entity pair. |
| `dropped_items` | Rejected-pointer audit evidence and rescue/review flags; stores source identity without a feed-item foreign key. |
| `backlog_mining_runs` | Optional mining status, config snapshot, counts and error payload. |
| `backlog_mapping_reviews` | Routing evidence linked to a mining run and entity, including confidence, target and notification outcome. |
| `backlog_candidates` | Optional draft evidence linked to entity/digest. Fingerprint, snapshots and state; self-references for supersession and merge. Partial unique fingerprint index for active states. |
| `backlog_candidate_evidence` | Evidence linked to candidate and optionally feed item. |
| `backlog_task_matches` | External task snapshots linked to candidate and optionally mining run; external keys are data, not local database foreign keys. |
| `backlog_candidate_recommendations` | Candidate foreign key with session-projection recommendation and rationale. |
| `backlog_candidate_dispositions` | Review history with unique frame ID, fingerprint, sequence and optional candidate foreign key. Stack snapshots support undo/redo. |
| `backlog_candidate_task_links` | External task links with origin/last-reconciled candidate foreign keys. Unique fingerprint/store/external-key triple. |
| `backlog_candidate_promotions` | Candidate foreign key with operation plan, lease, state and step receipts; operation ID is unique. |
| `alembic_version` | Alembic-managed applied schema revision. Not a pipeline model. |

Embeddings are 384-dimensional float32 arrays serialized as binary blobs and
compared in process using NumPy. There is no database vector extension requirement.
JSON stores structured configuration, evidence and diagnostics, not credentials.

## Migration policy

CLI commands apply `upgrade_to_head` before opening their session. Repeated startup
is safe; revisions are versioned in
[the initial migration](../src/rumen/db/migrations/versions/0001_curation_schema.py)
and later files in that directory. Do not edit an applied revision to change a
released schema. Add a revision and update the runtime models together, with
focused SQLite and Postgres migration tests. Back up a populated database before
upgrading. Downgrades are not a routine recovery procedure and may discard data.

SQLite foreign-key enforcement is enabled by the engine. Postgres can isolate
tables through `RUMEN_DB_SCHEMA`. Migrations own transaction boundaries so
batch backfills can retain progress. An optional single target identity can
complete older backlog rows idempotently; multi-target configurations do not
invent missing provenance. Migration compatibility is defined by tested revisions;
there is no automatic conversion from arbitrary upstream cache schemas.
