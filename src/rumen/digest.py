"""Digest synthesis — four-section digest from ranked entities (design Stage 6 / T-08).

Consumes the D-2 ranking order, asks the config-driven LLM to
synthesize the four sections, and stores JSON + rendered markdown in ``digests`` plus
``digest_entities`` membership rows. Every entry needs a concrete ``action``; entries
without one are routed to ``horizon_developments`` (design rule). Unparseable output
writes **no** partial digest — the prior digest stays latest.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    decode_embedding,
    get_default_provider,
)
from rumen.grouping.merge import _authority
from rumen.llm import LlmCommand, parse_json_response
from rumen.models import (
    Digest,
    DigestEntity,
    FeedItem,
    ItemConcept,
    SkosConcept,
    SkosRelation,
    TopicEntity,
)

_ACTION_SECTIONS = ("active_tool_updates", "immediate_techniques", "horizon_developments")

# D-17 deterministic category→section routing (config-overridable via ``section_map``).
# T-19 keys the configured ``digest_section_map`` by a top-level topic concept's
# lower-cased ``pref_label``; legacy category keys use that same normalization.
# Unmapped values fall to horizon developments.
_DEFAULT_SECTION_MAP = {
    "tool": "active_tool_updates",
    "launch": "active_tool_updates",
    "release": "active_tool_updates",
    "technique": "immediate_techniques",
}
_DEFAULT_SECTION = "horizon_developments"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DISPLAY_TITLE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DISPLAY_TITLE_MENTION = re.compile(r"(?<!\w)@\w+")
_DISPLAY_TITLE_MAX_LENGTH = 80
_DISPLAY_TITLE_MIN_LENGTH = 10

_PROMPT_TEMPLATE = """You are a rumen digest writer. Given these ranked topic \
entities, produce a JSON object with keys active_tool_updates, immediate_techniques, \
horizon_developments. Each is a list of entries: \
{{"title","source","summary","action","sources"}}. Every entry MUST carry a concrete \
"action"; if none applies, place the entry in horizon_developments.

Entities:
{entities}
"""

_POLISH_PROMPT_TEMPLATE = """Rewrite only the prose in this already-persisted extractive digest.
Return ONLY a JSON object with active_tool_updates, immediate_techniques, and
horizon_developments. Keep every entry in its current section and preserve each entry's exact
source and sources values. Each entry must include title, summary, action, source, and sources.
Do not add, remove, reorder, or change provenance references.

Extractive digest:
{sections}
"""


def build_prompt(entities: list[TopicEntity]) -> str:
    lines = [
        f"- [{e.id}] ({e.category or 'uncategorized'}) {e.display_title or e.canonical_title} "
        f"<{e.canonical_url}> sources={e.source_count}"
        for e in entities
    ]
    return _PROMPT_TEMPLATE.format(entities="\n".join(lines))


def _route_sections(parsed: dict) -> dict:
    """Keep only actioned entries in their section; route action-less entries to horizon."""
    sections: dict[str, list] = {s: [] for s in _ACTION_SECTIONS}
    orphans: list = []
    for section in _ACTION_SECTIONS:
        for entry in parsed.get(section, []) or []:
            if isinstance(entry, dict) and entry.get("action"):
                sections[section].append(entry)
            else:
                orphans.append(entry)
    sections["horizon_developments"].extend(orphans)
    return sections


def _render_markdown(sections: dict, period_start: datetime, period_end: datetime) -> str:
    titles = {
        "active_tool_updates": "Active Tool Updates",
        "immediate_techniques": "Immediate Techniques",
        "horizon_developments": "Horizon Developments",
    }
    out = [f"# Digest {period_start.date()} → {period_end.date()}", ""]
    for key in _ACTION_SECTIONS:
        out.append(f"## {titles[key]}")
        entries = sections.get(key, [])
        if not entries:
            out.append("_(none)_")
        for entry in entries:
            out.append(f"- **{entry.get('title', '')}** — {entry.get('summary', '')}")
            if entry.get("action"):
                out.append(f"  - Action: {entry['action']}")
        out.append("")
    noise = sections.get("noise_suppressed", {})
    out.append(f"## Noise Suppressed ({noise.get('count', 0)})")
    return "\n".join(out)


def append_backlog_candidate_review_references(markdown: str, candidates: list[dict]) -> str:
    """Add read-only candidate review commands to a stored digest Markdown body."""
    if not candidates:
        return markdown
    lines = [markdown.rstrip(), "", "## Backlog candidates"]
    for candidate in candidates:
        fingerprint = candidate["fingerprint"]
        lines.append(f"- {fingerprint}: `rumen backlog show {fingerprint}`")
    return "\n".join(lines) + "\n"


def build_polish_prompt(sections: dict) -> str:
    """Prompt contract for the local-only prose enrichment of an extractive digest."""
    import json

    preserved = {section: sections.get(section, []) for section in _ACTION_SECTIONS}
    return _POLISH_PROMPT_TEMPLATE.format(sections=json.dumps(preserved, sort_keys=True))


def _validated_polished_sections(original: dict, raw: str) -> dict:
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        raise TypeError("digest polish output was not a JSON object")
    polished = deepcopy(original)
    for section in _ACTION_SECTIONS:
        before = original.get(section, [])
        after = parsed.get(section)
        if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
            raise ValueError(f"digest polish changed entry count for {section}")
        for index, (original_entry, polished_entry) in enumerate(zip(before, after, strict=True)):
            if not isinstance(original_entry, dict) or not isinstance(polished_entry, dict):
                raise TypeError("digest polish entries must be JSON objects")
            if (
                polished_entry.get("title") != original_entry.get("title")
                or polished_entry.get("source") != original_entry.get("source")
                or polished_entry.get("sources") != original_entry.get("sources")
            ):
                raise ValueError("digest polish changed a required title or provenance reference")
            if not all(
                isinstance(polished_entry.get(field), str)
                for field in ("title", "summary", "action")
            ):
                raise ValueError("digest polish must provide title, summary, and action prose")
            for field in ("title", "summary", "action"):
                polished[section][index][field] = polished_entry[field]
    return polished


def polish_extractive_digest(
    session: Session,
    *,
    digest: Digest,
    llm_command: LlmCommand,
    model_used: str,
) -> Digest:
    """Atomically enrich extractive prose while retaining all membership and provenance.

    The extractive digest has already committed.  Any command, schema, or transaction failure is
    logged and leaves that durable result untouched; an explicitly configured command ladder may
    still perform its own normal fallback internally.
    """
    try:
        sections = _validated_polished_sections(
            digest.content_json, llm_command.run(build_polish_prompt(digest.content_json))
        )
        digest.content_json = sections
        digest.content_markdown = _render_markdown(sections, digest.period_start, digest.period_end)
        digest.model_used = model_used
        session.commit()
    except Exception as exc:  # noqa: BLE001 - a polish failure must never erase the committed extractive digest
        session.rollback()
        import logging

        logging.getLogger(__name__).warning(
            "local digest polish failed; retaining extractive digest: %s", exc
        )
        return session.get(Digest, digest.id) or digest
    return digest


def _split_sentences(text: str | None) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _is_informative_title(title: str | None) -> bool:
    """Return whether a constituent title is safe evidence of a real headline."""
    candidate = (title or "").strip()
    return (
        len(candidate) >= 3
        and len(candidate) <= _DISPLAY_TITLE_MAX_LENGTH * 3
        and _DISPLAY_TITLE_URL.fullmatch(candidate) is None
    )


def _heuristic_display_title(constituents: list[FeedItem]) -> str:
    """Derive a compact zero-LLM headline from social-style constituent text."""
    for constituent in constituents:
        for text in (constituent.title, constituent.content):
            cleaned = _DISPLAY_TITLE_MENTION.sub("", text or "")
            cleaned = _DISPLAY_TITLE_URL.sub("", cleaned)
            sentences = _split_sentences(" ".join(cleaned.split()))
            if sentences:
                return sentences[0][:_DISPLAY_TITLE_MAX_LENGTH].rstrip()
    return ""


def ensure_display_title(
    session: Session,
    entity: TopicEntity,
    constituents: list[FeedItem],
    llm_command: LlmCommand | None,
) -> str:
    """Resolve the digest title for one selected entity without committing the transaction."""
    if any(
        _authority(constituent.source) >= 2 and _is_informative_title(constituent.title)
        for constituent in constituents
    ):
        entity.display_title = entity.canonical_title
        return entity.display_title

    if entity.display_title is not None and entity.display_title.strip():
        entity.display_title = entity.display_title
        return entity.display_title

    if not constituents:
        entity.display_title = entity.canonical_title
        return entity.display_title

    display_title = _heuristic_display_title(constituents)
    if len(display_title) < _DISPLAY_TITLE_MIN_LENGTH and llm_command is not None:
        constituent_text = "\n".join(
            text
            for constituent in constituents
            for text in (constituent.title, constituent.content)
            if text
        )
        prompt = (
            "Write one concise digest headline (maximum 80 characters) from this source text. "
            "Return only the headline.\n\n"
            f"{constituent_text}"
        )
        try:
            generated = llm_command.run(prompt).strip()
        except Exception:  # noqa: BLE001 - an unavailable fallback ladder must not fail a digest
            generated = ""
        if generated:
            display_title = generated

    entity.display_title = display_title
    return display_title


def _extractive_summary(
    entity: TopicEntity,
    constituents: list[FeedItem],
    provider: EmbeddingProvider,
    sentences_per_entity: int,
) -> str:
    """Top-``k`` sentences of the entity's constituents by cosine to its centroid (D-17).

    Reuses the already-computed embedding space — zero LLM. Falls back to the canonical
    title when there is no text, and to leading sentences when there is no centroid."""
    candidates: list[str] = []
    seen: set[str] = set()
    for item in constituents:
        for source_text in (item.title, item.content):
            for sentence in _split_sentences(source_text):
                if sentence not in seen:
                    seen.add(sentence)
                    candidates.append(sentence)
    if not candidates:
        return entity.canonical_title
    centroid = decode_embedding(entity.embedding)
    if centroid is None or centroid.size == 0:
        return " ".join(candidates[:sentences_per_entity])
    vectors = provider.embed(candidates)
    scored = sorted(
        range(len(candidates)),
        key=lambda i: cosine_similarity(vectors[i], centroid),
        reverse=True,
    )[:sentences_per_entity]
    scored.sort()  # keep the selected sentences in their original reading order
    return " ".join(candidates[i] for i in scored)


def _top_level_topic_label(session: Session, entity_id: int) -> str | None:
    """Return the deterministic root label for an entity's topic-facet assignment.

    A valid taxonomy has one topic path.  If an in-progress taxonomy has multiple
    broader parents, select the lowest-id root deterministically rather than making
    routing depend on database row order.  A cycle or a missing concept is treated as
    unresolved so the legacy category migration fallback remains available.
    """
    assignment = session.scalar(
        select(ItemConcept)
        .where(
            ItemConcept.item_table == "topic_entities",
            ItemConcept.item_id == entity_id,
            ItemConcept.facet == "topic",
        )
        .order_by(ItemConcept.id)
    )
    if assignment is None:
        return None

    concept = session.get(SkosConcept, assignment.concept_id)
    if concept is None:
        return None
    current_ids = {concept.id}
    visited: set[int] = set()
    root_ids: set[int] = set()
    while current_ids:
        if current_ids & visited:
            return None
        visited.update(current_ids)
        relations = session.execute(
            select(SkosRelation.source_concept_id, SkosRelation.target_concept_id).where(
                SkosRelation.source_concept_id.in_(current_ids),
                SkosRelation.relation_type == "broader",
            )
        ).all()
        children_with_parents = {source_id for source_id, _ in relations}
        root_ids.update(current_ids - children_with_parents)
        parent_ids = {target_id for _, target_id in relations}
        current_ids = parent_ids
    if not root_ids:
        return None
    root = session.get(SkosConcept, min(root_ids))
    return root.pref_label if root is not None else None


def _section_for_entity(session: Session, entity: TopicEntity, routing: dict[str, str]) -> str:
    """Route topic facet first, then legacy category, then the unchanged default."""
    topic_label = _top_level_topic_label(session, entity.id)
    if topic_label is not None:
        section = routing.get(topic_label.lower())
        if section is not None:
            return section
    return routing.get((entity.category or "").lower(), _DEFAULT_SECTION)


def synthesize_digest_extractive(
    session: Session,
    *,
    entity_ids: list[int],
    period_start: datetime,
    period_end: datetime,
    provider: EmbeddingProvider | None = None,
    llm_command: LlmCommand | None = None,
    section_map: dict[str, str] | None = None,
    sentences_per_entity: int = 2,
    suppressed_count: int = 0,
    suppressed_examples: list[str] | None = None,
    model_used: str = "extractive",
) -> Digest:
    """Zero-LLM digest (design D-17): deterministic facet/category→section routing + extractive
    MiniLM-cosine sentence summaries. Always produces a digest — the failure mode fixed is
    the hosted-LLM hard dependency (the digest no longer breaks when the LLM backend is down).
    Optional local-LLM abstractive polish (D-16) is a later enrichment, not required here."""
    provider = provider or get_default_provider()
    routing = {**_DEFAULT_SECTION_MAP, **{k.lower(): v for k, v in (section_map or {}).items()}}

    sections: dict[str, list] = {s: [] for s in _ACTION_SECTIONS}
    present_ids: list[int] = []
    for eid in entity_ids:
        entity = session.get(TopicEntity, eid)
        if entity is None:
            continue
        present_ids.append(eid)
        constituents = list(
            session.scalars(select(FeedItem).where(FeedItem.topic_entity_id == eid))
        )
        display_title = ensure_display_title(session, entity, constituents, llm_command)
        section = _section_for_entity(session, entity, routing)
        sources = [c.url for c in constituents] or [entity.canonical_url]
        sections[section].append(
            {
                "title": display_title,
                "source": entity.canonical_url,
                "summary": _extractive_summary(
                    entity, constituents, provider, sentences_per_entity
                ),
                "action": "",  # no LLM to synthesize an action in the extractive path
                "sources": sources,
            }
        )

    sections["noise_suppressed"] = {
        "count": suppressed_count,
        "examples": suppressed_examples or [],
    }
    markdown = _render_markdown(sections, period_start, period_end)
    digest = Digest(
        period_start=period_start,
        period_end=period_end,
        content_json=sections,
        content_markdown=markdown,
        model_used=model_used,
    )
    session.add(digest)
    session.flush()
    for rank_idx, eid in enumerate(present_ids):
        session.add(DigestEntity(digest_id=digest.id, topic_entity_id=eid, rank=rank_idx))
    session.commit()
    return digest


def _lock_abstractive_titles(
    sections: dict,
    display_titles_by_url: dict[str, str],
) -> None:
    """Replace LLM titles with selected entities' resolved display titles.

    The prompt's provenance contract identifies an entity by ``source`` (or a member of
    ``sources``). A one-entity digest has an unambiguous fallback for older model responses
    that omit provenance fields.
    """
    only_title = (
        next(iter(display_titles_by_url.values())) if len(display_titles_by_url) == 1 else None
    )
    for section in _ACTION_SECTIONS:
        for entry in sections.get(section, []):
            if not isinstance(entry, dict):
                continue
            references = [entry.get("source")]
            sources = entry.get("sources")
            if isinstance(sources, list):
                references.extend(sources)
            title = next(
                (
                    display_titles_by_url[reference]
                    for reference in references
                    if reference in display_titles_by_url
                ),
                only_title,
            )
            if title is not None:
                entry["title"] = title


def synthesize_digest(
    session: Session,
    *,
    llm_command: LlmCommand,
    entity_ids: list[int],
    period_start: datetime,
    period_end: datetime,
    suppressed_count: int = 0,
    suppressed_examples: list[str] | None = None,
    model_used: str = "unspecified",
) -> Digest:
    """Synthesize + persist one digest. Raises ``ValueError`` or ``TypeError`` for invalid LLM
    output before any row is written, so no partial digest is created."""
    entities = [session.get(TopicEntity, eid) for eid in entity_ids]
    entities = [e for e in entities if e is not None]
    display_titles_by_url = {}
    for entity in entities:
        constituents = list(
            session.scalars(select(FeedItem).where(FeedItem.topic_entity_id == entity.id))
        )
        display_titles_by_url[entity.canonical_url] = ensure_display_title(
            session, entity, constituents, llm_command
        )

    raw = llm_command.run(build_prompt(entities))
    parsed = parse_json_response(raw)  # raises ValueError → no partial write
    if not isinstance(parsed, dict):
        raise TypeError("digest LLM output was not a JSON object")

    # The abstractive contract gives the LLM its own section placement; unlike the
    # extractive path, it has no category-keyed deterministic routing to replace.
    sections = _route_sections(parsed)
    _lock_abstractive_titles(sections, display_titles_by_url)
    sections["noise_suppressed"] = {
        "count": suppressed_count,
        "examples": suppressed_examples or [],
    }
    markdown = _render_markdown(sections, period_start, period_end)

    digest = Digest(
        period_start=period_start,
        period_end=period_end,
        content_json=sections,
        content_markdown=markdown,
        model_used=model_used,
    )
    session.add(digest)
    session.flush()
    for rank_idx, eid in enumerate(entity_ids):
        session.add(DigestEntity(digest_id=digest.id, topic_entity_id=eid, rank=rank_idx))
    session.commit()
    return digest
