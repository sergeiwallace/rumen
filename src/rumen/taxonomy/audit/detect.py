"""Conservative structural issue detection for D-23 taxonomy audits."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rumen.models import ItemConcept, SkosConcept, SkosRelation
from rumen.taxonomy.audit.cluster import SimilarConceptPair


@dataclass(frozen=True)
class AuditIssue:
    """A detected maintenance candidate, to be independently proposed by the LLM."""

    op_type: str
    concept_ids: tuple[int, ...]
    rationale: str
    details: dict[str, object] = field(default_factory=dict)


def detect_issues(
    session: Session,
    *,
    similar_pairs: list[SimilarConceptPair],
    crowded_item_threshold: int = 100,
    sparse_item_threshold: int = 0,
) -> list[AuditIssue]:
    """Find crowded leaves, dead leaves, and unaliased duplicate pairs.

    Only active concepts are candidates.  This is important for self-healing: an
    earlier ``prune`` that deprecated a leaf and an earlier ``consolidate`` that
    added an alias do not get proposed again on every scheduled run.
    """
    active = session.scalars(
        select(SkosConcept).where(SkosConcept.status == "active").order_by(SkosConcept.id)
    ).all()
    active_ids = {concept.id for concept in active}
    counts = dict(
        session.execute(
            select(ItemConcept.concept_id, func.count(ItemConcept.id)).group_by(
                ItemConcept.concept_id
            )
        ).all()
    )
    relations = session.scalars(select(SkosRelation)).all()
    children: set[int] = set()
    aliases: set[frozenset[int]] = set()
    for relation in relations:
        if relation.relation_type == "broader":
            children.add(relation.target_concept_id)
        elif relation.relation_type == "narrower":
            children.add(relation.source_concept_id)
        elif relation.relation_type == "useFor":
            aliases.add(frozenset((relation.source_concept_id, relation.target_concept_id)))

    issues: list[AuditIssue] = []
    for concept in active:
        if concept.id in children:
            continue
        item_count = int(counts.get(concept.id, 0))
        if item_count > crowded_item_threshold:
            issues.append(
                AuditIssue(
                    "deepen",
                    (concept.id,),
                    f"{concept.pref_label!r} is a crowded leaf with {item_count} assignments",
                    {"item_count": item_count, "facet": concept.facet},
                )
            )
        elif item_count <= sparse_item_threshold:
            issues.append(
                AuditIssue(
                    "prune",
                    (concept.id,),
                    f"{concept.pref_label!r} is a sparse leaf with {item_count} assignments",
                    {"item_count": item_count, "facet": concept.facet},
                )
            )
    for pair in similar_pairs:
        pair_ids = frozenset((pair.first_id, pair.second_id))
        if (
            pair.first_id not in active_ids
            or pair.second_id not in active_ids
            or pair_ids in aliases
        ):
            continue
        issues.append(
            AuditIssue(
                "consolidate",
                tuple(sorted(pair_ids)),
                f"near-duplicate labels have embedding similarity {pair.similarity:.3f}",
                {"similarity": pair.similarity, "facet": pair.facet},
            )
        )
    return issues
