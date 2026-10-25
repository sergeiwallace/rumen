"""T-19: digest section routing prefers topic-facet roots over legacy categories."""

from __future__ import annotations

from datetime import UTC, datetime

from rumen.digest import synthesize_digest_extractive
from rumen.models import FeedItem, ItemConcept, SkosConcept, SkosRelation, TopicEntity

_START = datetime(2026, 7, 9, tzinfo=UTC)
_END = datetime(2026, 7, 16, tzinfo=UTC)


def _entity(session, title: str, category: str | None) -> TopicEntity:
    entity = TopicEntity(
        canonical_title=title,
        canonical_url=f"https://example.test/{title}",
        category=category,
        earliest_seen=_START,
        source_count=1,
    )
    session.add(entity)
    session.flush()
    return entity


def _section_titles(digest, section: str) -> set[str]:
    return {entry["title"] for entry in digest.content_json[section]}


def test_topic_facet_root_routes_before_conflicting_legacy_category(session) -> None:
    entity = _entity(session, "Facet wins", "tool")
    root = SkosConcept(facet="topic", pref_label="Research", scheme="fixture")
    child = SkosConcept(facet="topic", pref_label="Model evaluations", scheme="fixture")
    session.add_all([root, child])
    session.flush()
    session.add_all(
        [
            SkosRelation(
                source_concept_id=child.id,
                target_concept_id=root.id,
                relation_type="broader",
            ),
            ItemConcept(
                item_table="topic_entities",
                item_id=entity.id,
                concept_id=child.id,
                facet="topic",
                confidence=1.0,
                method="fixture",
            ),
        ]
    )
    session.commit()

    digest = synthesize_digest_extractive(
        session,
        entity_ids=[entity.id],
        period_start=_START,
        period_end=_END,
        section_map={"research": "immediate_techniques"},
    )

    assert "Facet wins" in _section_titles(digest, "immediate_techniques")
    assert "Facet wins" not in _section_titles(digest, "active_tool_updates")


def test_no_topic_facet_assignment_falls_back_to_legacy_category(session) -> None:
    entity = _entity(session, "Legacy fallback", "tool")
    session.commit()

    digest = synthesize_digest_extractive(
        session,
        entity_ids=[entity.id],
        period_start=_START,
        period_end=_END,
    )

    assert "Legacy fallback" in _section_titles(digest, "active_tool_updates")


def test_unmapped_topic_facet_and_no_category_fall_back_to_default_section(session) -> None:
    entity = _entity(session, "Default fallback", None)
    root = SkosConcept(facet="topic", pref_label="Unmapped root", scheme="fixture")
    session.add(root)
    session.flush()
    session.add(
        ItemConcept(
            item_table="topic_entities",
            item_id=entity.id,
            concept_id=root.id,
            facet="topic",
            confidence=1.0,
            method="fixture",
        )
    )
    session.commit()

    digest = synthesize_digest_extractive(
        session,
        entity_ids=[entity.id],
        period_start=_START,
        period_end=_END,
        section_map={"other root": "immediate_techniques"},
    )

    assert "Default fallback" in _section_titles(digest, "horizon_developments")


def test_given_recomputed_entity_topic_projection_when_digest_runs_then_topic_routing_is_used(
    session,
) -> None:
    root = SkosConcept(facet="topic", pref_label="Research", scheme="fixture")
    session.add(root)
    session.flush()
    item = FeedItem(
        source="reddit",
        source_id="projected",
        url="https://example/projected",
        title="Projected topic",
        published_at=_START,
        category="tool",
        classification_confidence=0.9,
    )
    session.add(item)
    session.flush()
    entity = _entity(session, "Projected topic", "tool")
    item.topic_entity_id = entity.id
    session.add(
        ItemConcept(
            item_table="feed_items",
            item_id=item.id,
            concept_id=root.id,
            facet="topic",
            confidence=0.9,
            method="hmc",
        )
    )
    from rumen.grouping.merge import recompute_entity

    recompute_entity(session, entity, items=[item])
    session.commit()

    digest = synthesize_digest_extractive(
        session,
        entity_ids=[entity.id],
        period_start=_START,
        period_end=_END,
        section_map={"research": "immediate_techniques"},
    )
    assert "Projected topic" in _section_titles(digest, "immediate_techniques")
