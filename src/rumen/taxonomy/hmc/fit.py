"""Fit cleanlab-gated, per-facet HMC classifiers from T-16 bootstrap assignments."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from rumen.classifier.tier1_embedding import FacetHMCClassifier
from rumen.embeddings import EmbeddingProvider, get_default_provider
from rumen.models import FeedItem, ItemConcept, SkosConcept, SkosRelation, TopicEntity
from rumen.taxonomy.hmc.cleanlab_gate import CleanlabGateResult, gate_label_issues
from rumen.taxonomy.hmc.gold_eval import (
    GoldEvaluation,
    GoldExample,
    evaluate_gold_set,
    load_human_gold_examples,
)

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class BootstrapExample:
    """A T-16 provenance-tagged label paired with the source text needed for HMC fitting."""

    key: str
    text: str
    concept_id: int
    confidence: float


@dataclass(frozen=True)
class FacetFitReport:
    """Fit/gate/evaluation result for one facet, including Tier-1 eligibility."""

    facet: str
    classifier: FacetHMCClassifier | None
    n_bootstrap: int
    n_after_gate: int
    n_excluded: int
    gold_evaluation: GoldEvaluation
    tier1_eligible: bool
    skipped: bool

    def assign_for_tier1(self, text: str, *, path_consistency_floor: float) -> dict[int, float]:
        """Return only auto-acceptable assignments; ineligible facets fall through to Tier-2."""
        if not self.tier1_eligible or self.classifier is None:
            return {}
        return self.classifier.assign(text, path_consistency_floor=path_consistency_floor)


def _validate_bundle_name(name: str, *, kind: str) -> None:
    """Require a single filename component on both POSIX and Windows."""
    if (
        not isinstance(name, str)
        or not name
        or name == "."
        or ".." in name
        or Path(name).name != name
        or any(character in name for character in ("/", "\\", ":", "\0"))
    ):
        raise ValueError(f"invalid HMC bundle {kind}: expected a single safe filename component")


def save_report_bundle(reports: dict[str, FacetFitReport], path: str | Path) -> None:
    """Persist one loadable HMC member per facet plus auditable fit/evaluation metadata."""
    for facet in reports:
        _validate_bundle_name(facet, kind="facet")
    bundle = Path(path)
    bundle.mkdir(parents=True, exist_ok=True)
    facets: dict[str, dict] = {}
    for facet, report in reports.items():
        member = None
        if report.classifier is not None:
            member = f"{facet}.npz"
            report.classifier.save(bundle / member)
        evaluation = report.gold_evaluation
        facets[facet] = {
            "member": member,
            "n_bootstrap": report.n_bootstrap,
            "n_after_gate": report.n_after_gate,
            "n_excluded": report.n_excluded,
            "tier1_eligible": report.tier1_eligible,
            "skipped": report.skipped,
            "gold_evaluation": {
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "true_positives": evaluation.true_positives,
                "false_positives": evaluation.false_positives,
                "false_negatives": evaluation.false_negatives,
                "evaluated_count": evaluation.evaluated_count,
                "evaluated": evaluation.evaluated,
            },
        }
    (bundle / _MANIFEST).write_text(
        json.dumps({"version": 1, "facets": facets}, indent=2, sort_keys=True)
    )


def load_report_bundle(
    path: str | Path, *, provider: EmbeddingProvider | None = None
) -> dict[str, FacetFitReport] | None:
    """Load a bundle or return ``None`` when absent; malformed bundles raise ``ValueError``."""
    bundle = Path(path)
    manifest_path = bundle / _MANIFEST
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        facet_data = manifest["facets"]
        if manifest.get("version") != 1 or not isinstance(facet_data, dict):
            raise ValueError("unsupported HMC bundle manifest")
        reports: dict[str, FacetFitReport] = {}
        for facet, data in facet_data.items():
            evaluation = GoldEvaluation(facet, **data["gold_evaluation"])
            member = data.get("member")
            classifier = None
            if member is not None:
                _validate_bundle_name(member, kind="member")
                classifier = FacetHMCClassifier.load(bundle / member, provider=provider)
                if classifier.facet != facet:
                    raise ValueError(f"HMC bundle member facet mismatch for {facet}")
            reports[facet] = FacetFitReport(
                facet,
                classifier,
                int(data["n_bootstrap"]),
                int(data["n_after_gate"]),
                int(data["n_excluded"]),
                evaluation,
                bool(data["tier1_eligible"]),
                bool(data["skipped"]),
            )
        return reports
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"malformed HMC bundle {bundle}: {exc}") from exc


Gate = Callable[[list[int], np.ndarray], CleanlabGateResult]


def fit_facet(
    facet: str,
    bootstrap_examples: list[BootstrapExample],
    *,
    parents: dict[int, set[int]] | None = None,
    gold_examples: list[GoldExample] | None = None,
    provider: EmbeddingProvider | None = None,
    noise_fraction: float = 0.2,
    gold_precision_floor: float = 0.90,
    path_consistency_floor: float = 0.75,
    gate: Gate | None = None,
) -> FacetFitReport:
    """Fit one facet after the real cleanlab gate and evaluate on held-out human labels.

    The cleanlab probabilities come from deterministic leave-one-out centroid scores.  This
    avoids asking cleanlab to judge a row using a centroid that includes that same row.
    """
    gold_examples = gold_examples or []
    empty_evaluation = GoldEvaluation(facet, None, None, 0, 0, 0, 0, False)
    if not bootstrap_examples:
        logger.error("Skipping HMC fit for facet %s: no bootstrap examples", facet)
        return FacetFitReport(facet, None, 0, 0, 0, empty_evaluation, False, True)

    active_provider = provider or get_default_provider()
    concept_ids = sorted({example.concept_id for example in bootstrap_examples})
    label_indices = {concept_id: index for index, concept_id in enumerate(concept_ids)}
    labels = [label_indices[example.concept_id] for example in bootstrap_examples]
    probabilities = _leave_one_out_probabilities(bootstrap_examples, active_provider)
    gate_result = (
        gate(labels, probabilities)
        if gate is not None
        else gate_label_issues(labels, probabilities, noise_fraction=noise_fraction)
    )
    kept = [bootstrap_examples[index] for index in gate_result.kept_indices]
    if not kept:
        logger.error(
            "Skipping HMC fit for facet %s: cleanlab flagged all %d bootstrap examples as noise",
            facet,
            len(bootstrap_examples),
        )
        return FacetFitReport(
            facet,
            None,
            len(bootstrap_examples),
            0,
            len(gate_result.excluded_indices),
            empty_evaluation,
            False,
            True,
        )

    classifier = FacetHMCClassifier(facet, active_provider, parents=parents).fit(
        [(example.text, example.concept_id) for example in kept]
    )
    evaluation = evaluate_gold_set(
        classifier, gold_examples, path_consistency_floor=path_consistency_floor
    )
    tier1_eligible = bool(
        evaluation.evaluated
        and evaluation.precision is not None
        and evaluation.precision >= gold_precision_floor
    )
    return FacetFitReport(
        facet,
        classifier,
        len(bootstrap_examples),
        len(kept),
        len(gate_result.excluded_indices),
        evaluation,
        tier1_eligible,
        False,
    )


def fit_facet_from_session(
    session: Session,
    *,
    facet: str,
    provider: EmbeddingProvider | None = None,
    noise_fraction: float = 0.2,
    gold_precision_floor: float = 0.90,
    path_consistency_floor: float = 0.75,
) -> FacetFitReport:
    """Assemble a facet from ``item_concepts`` bootstrap rows and its human gold rows."""
    return fit_facet(
        facet,
        load_bootstrap_examples(session, facet=facet),
        parents=load_facet_parents(session, facet=facet),
        gold_examples=load_human_gold_examples(session, facet=facet),
        provider=provider,
        noise_fraction=noise_fraction,
        gold_precision_floor=gold_precision_floor,
        path_consistency_floor=path_consistency_floor,
    )


def fit_all_facets_from_session(
    session: Session,
    *,
    facets: Iterable[str] | None = None,
    provider: EmbeddingProvider | None = None,
    noise_fraction: float = 0.2,
    gold_precision_floor: float = 0.90,
    path_consistency_floor: float = 0.75,
) -> dict[str, FacetFitReport]:
    """Fit one classifier per requested/bootstrap-present facet."""
    if facets is None:
        facets = session.scalars(
            select(ItemConcept.facet).where(ItemConcept.method.startswith("bootstrap:")).distinct()
        ).all()
    return {
        facet: fit_facet_from_session(
            session,
            facet=facet,
            provider=provider,
            noise_fraction=noise_fraction,
            gold_precision_floor=gold_precision_floor,
            path_consistency_floor=path_consistency_floor,
        )
        for facet in sorted(set(facets))
    }


def load_bootstrap_examples(session: Session, *, facet: str) -> list[BootstrapExample]:
    """Read only T-16 ``bootstrap:`` item-concept assignments for one facet."""
    labels = session.scalars(
        select(ItemConcept).where(
            ItemConcept.facet == facet,
            ItemConcept.method.startswith("bootstrap:"),
        )
    ).all()
    examples: list[BootstrapExample] = []
    for label in labels:
        item = _load_item(session, label.item_table, label.item_id)
        if item is not None and (text := _item_text(item)):
            examples.append(
                BootstrapExample(
                    f"{label.item_table}:{label.item_id}:{label.concept_id}",
                    text,
                    label.concept_id,
                    label.confidence,
                )
            )
    return examples


def load_facet_parents(session: Session, *, facet: str) -> dict[int, set[int]]:
    """Walk materialized SKOS ``broader``/``narrower`` edges to build parent paths."""
    concept_ids = set(
        session.scalars(select(SkosConcept.id).where(SkosConcept.facet == facet)).all()
    )
    parents: dict[int, set[int]] = {}
    if not concept_ids:
        return parents
    relations = session.scalars(
        select(SkosRelation).where(SkosRelation.relation_type.in_(("broader", "narrower")))
    ).all()
    for relation in relations:
        if {relation.source_concept_id, relation.target_concept_id} - concept_ids:
            continue
        child, parent = (
            (relation.source_concept_id, relation.target_concept_id)
            if relation.relation_type == "broader"
            else (relation.target_concept_id, relation.source_concept_id)
        )
        parents.setdefault(child, set()).add(parent)
    return parents


def _leave_one_out_probabilities(
    examples: list[BootstrapExample], provider: EmbeddingProvider
) -> np.ndarray:
    """Generate cleanlab's class probabilities without training a row on itself."""
    labels = sorted({example.concept_id for example in examples})
    vectors = provider.embed([example.text for example in examples])
    probabilities = np.zeros((len(examples), len(labels)), dtype=np.float64)
    for index, vector in enumerate(vectors):
        centroids: list[np.ndarray] = []
        for concept_id in labels:
            peer_indices = [
                peer_index
                for peer_index, example in enumerate(examples)
                if example.concept_id == concept_id and peer_index != index
            ]
            if not peer_indices:
                peer_indices = [
                    peer_index
                    for peer_index, example in enumerate(examples)
                    if example.concept_id == concept_id
                ]
            centroid = np.mean(vectors[peer_indices], axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids.append(centroid / norm if norm else centroid)
        similarities = np.asarray(centroids) @ vector
        logits = similarities - np.max(similarities)
        weights = np.exp(logits)
        probabilities[index] = weights / np.sum(weights)
    return probabilities


def _load_item(session: Session, item_table: str, item_id: int) -> FeedItem | TopicEntity | None:
    if item_table == "feed_items":
        return session.get(FeedItem, item_id)
    if item_table == "topic_entities":
        return session.get(TopicEntity, item_id)
    return None


def _item_text(item: FeedItem | TopicEntity) -> str:
    if isinstance(item, FeedItem):
        return f"{item.title or ''}\n{item.content or ''}".strip()
    return item.canonical_title.strip()
