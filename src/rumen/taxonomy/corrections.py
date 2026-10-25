"""Transactional human classification-review capture (T-24)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rumen.classifier.cascade import _closure, _parents, _topic_root
from rumen.dropped_items import log_drop
from rumen.grouping.merge import recompute_entity
from rumen.models import (
    ClassificationCorrection,
    FeedItem,
    ItemConcept,
    SkosConcept,
    TopicEntity,
)

_TARGETS = {"feed_items": FeedItem, "topic_entities": TopicEntity}


def record_classification_review(
    session: Session,
    *,
    item_table: str,
    item_id: int,
    facet: str,
    corrected_concept_ids: Iterable[int],
    actor: str,
    rationale: str,
    source: str,
    classifier_context: dict | None = None,
    out_of_scope: frozenset[str] = frozenset(),
    route: str | None = None,
) -> ClassificationCorrection:
    """Accept one complete human facet review and replace its current projection.

    Validation happens before any mutation.  The event, prior-event supersession, and
    complete ``human:<source>`` projection are contained in one savepoint so callers
    can safely commit the review as a single transaction.
    """
    corrected_ids = _validate_review(
        session,
        item_table=item_table,
        item_id=item_id,
        facet=facet,
        corrected_concept_ids=corrected_concept_ids,
        actor=actor,
        rationale=rationale,
        source=source,
        classifier_context=classifier_context,
    )
    selected_ids = corrected_ids
    target = session.get(_TARGETS[item_table], item_id)
    topic_concepts: dict[int, SkosConcept] | None = None
    topic_parents: dict[int, set[int]] | None = None
    derived_category: str | None = None
    if route not in {None, "route", "keep"}:
        raise ValueError("route must be either 'route' or 'keep'")
    if facet == "topic":
        corrected_ids, topic_concepts, topic_parents = _topic_closure(
            session, corrected_ids, facet=facet
        )
        root = _topic_root({concept_id: 1.0 for concept_id in corrected_ids}, topic_parents)
        if root is None:
            raise ValueError("corrected topic concepts do not resolve to a root")
        derived_category = topic_concepts[root].pref_label
        if item_table == "feed_items" and derived_category in out_of_scope and route is None:
            raise ValueError(
                f"derived category {derived_category!r} is out of scope; choose route or keep"
            )
    prior_assignments = session.scalars(
        select(ItemConcept)
        .where(
            ItemConcept.item_table == item_table,
            ItemConcept.item_id == item_id,
            ItemConcept.facet == facet,
        )
        .order_by(ItemConcept.concept_id)
    ).all()
    prior_ids = [assignment.concept_id for assignment in prior_assignments]
    context = (
        classifier_context
        if classifier_context is not None
        else {
            "assignments": [
                {
                    "concept_id": assignment.concept_id,
                    "method": assignment.method,
                    "confidence": assignment.confidence,
                }
                for assignment in prior_assignments
            ]
        }
    )
    if facet == "topic":
        assert topic_concepts is not None
        context = {
            **context,
            "topic_closure": {
                "selected_concept_ids": selected_ids,
                "inferred_concept_ids": sorted(set(corrected_ids) - set(selected_ids)),
            },
        }
    now = datetime.now(UTC)

    with session.begin_nested():
        prior_events = session.scalars(
            select(ClassificationCorrection)
            .where(
                ClassificationCorrection.item_table == item_table,
                ClassificationCorrection.item_id == item_id,
                ClassificationCorrection.facet == facet,
                ClassificationCorrection.status == "accepted",
            )
            .order_by(
                ClassificationCorrection.decided_at.desc(), ClassificationCorrection.id.desc()
            )
        ).all()
        correction = ClassificationCorrection(
            item_table=item_table,
            item_id=item_id,
            facet=facet,
            action="approval" if set(prior_ids) == set(corrected_ids) else "correction",
            prior_concept_ids=prior_ids,
            corrected_concept_ids=corrected_ids,
            classifier_context_json=context,
            actor=actor.strip(),
            source=source.strip(),
            rationale=rationale.strip(),
            status="accepted",
            decision_actor=actor.strip(),
            decision_rationale=rationale.strip(),
            decided_at=now,
            supersedes=prior_events[0].id if prior_events else None,
        )
        session.add(correction)
        session.flush()

        session.execute(
            delete(ItemConcept).where(
                ItemConcept.item_table == item_table,
                ItemConcept.item_id == item_id,
                ItemConcept.facet == facet,
            )
        )
        session.add_all(
            ItemConcept(
                item_table=item_table,
                item_id=item_id,
                concept_id=concept_id,
                facet=facet,
                confidence=1.0,
                method=f"human:{source.strip()}",
            )
            for concept_id in corrected_ids
        )
        for prior_event in prior_events:
            prior_event.status = "superseded"
        session.flush()

        if facet == "topic":
            assert derived_category is not None
            if item_table == "feed_items":
                assert isinstance(target, FeedItem)
                target.category = derived_category
                target.classification_confidence = 1.0
                target.classification_method = f"human:{source.strip()}"
                if derived_category in out_of_scope and route == "route":
                    log_drop(
                        session,
                        source=target.source,
                        source_id=target.source_id,
                        url=target.url,
                        title=target.title,
                        reason="classified_out_of_scope",
                        drop_confidence=1.0,
                    )
                    target.status = "dropped"
                if target.topic_entity_id is not None:
                    entity = session.get(TopicEntity, target.topic_entity_id)
                    if entity is not None:
                        recompute_entity(session, entity)
            else:
                assert isinstance(target, TopicEntity)
                target.category = derived_category
        session.flush()
    return correction


def _validate_review(
    session: Session,
    *,
    item_table: str,
    item_id: int,
    facet: str,
    corrected_concept_ids: Iterable[int],
    actor: str,
    rationale: str,
    source: str,
    classifier_context: dict | None,
) -> list[int]:
    """Validate every request invariant before the transaction mutates state."""
    target_type = _TARGETS.get(item_table)
    if target_type is None or session.get(target_type, item_id) is None:
        raise ValueError(f"no such {item_table} target: {item_id}")
    if not isinstance(facet, str) or not facet.strip():
        raise ValueError("facet must not be blank")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("actor must not be blank")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must not be blank")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must not be blank")
    if classifier_context is not None and not isinstance(classifier_context, dict):
        raise ValueError("classifier_context must be an object")

    corrected_ids = list(corrected_concept_ids)
    if not corrected_ids:
        raise ValueError("corrected concept set must not be empty")
    if any(
        not isinstance(concept_id, int) or isinstance(concept_id, bool)
        for concept_id in corrected_ids
    ):
        raise ValueError("corrected concept ids must be integers")
    if len(set(corrected_ids)) != len(corrected_ids):
        raise ValueError("corrected concept set must not contain duplicates")
    concepts = {
        concept.id: concept
        for concept in session.scalars(select(SkosConcept).where(SkosConcept.id.in_(corrected_ids)))
    }
    missing_ids = set(corrected_ids) - set(concepts)
    if missing_ids:
        raise ValueError(f"corrected concept does not exist: {min(missing_ids)}")
    if any(concepts[concept_id].facet != facet for concept_id in corrected_ids):
        raise ValueError("corrected concept does not belong to the requested facet")
    return sorted(corrected_ids)


def _topic_closure(
    session: Session, selected_ids: list[int], *, facet: str
) -> tuple[list[int], dict[int, SkosConcept], dict[int, set[int]]]:
    """Resolve selected topic concepts to the cascade's complete active-parent closure."""
    active_concepts = list(
        session.scalars(
            select(SkosConcept).where(SkosConcept.facet == facet, SkosConcept.status == "active")
        )
    )
    concepts = {concept.id: concept for concept in active_concepts}
    parents = _parents(session, facet, active_concepts)
    closure = _closure(set(selected_ids), parents, set(concepts))
    if closure is None:
        raise ValueError(
            "corrected topic concepts do not resolve to a complete active parent closure"
        )
    return sorted(closure), concepts, parents
