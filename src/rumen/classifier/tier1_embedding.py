"""Tier-1 classifier artifacts.

``EmbeddingClassifier`` is retained solely for historical artifact/test compatibility; T-25
retires it from the live classification and fit/load paths. ``FacetHMCClassifier`` is the sole
live Tier-1 model.

A nearest-centroid classifier over embeddings (a label-centroid method): fit per-category
mean centroids from labeled examples, predict by cosine to each centroid, and emit a
calibrated confidence. Categories with no labeled examples fall back to a
**description centroid** (zero-shot).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from rumen.embeddings import EmbeddingProvider, get_default_provider

# Default gate threshold before empirical calibration (design D-12; heuristic 0.75).
DEFAULT_TAU_CLASS = 0.75


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / temperature
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class EmbeddingClassifier:
    """Retired flat nearest-centroid classifier; never invoke from the live cascade."""

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        *,
        temperature: float = 0.1,
    ) -> None:
        self.provider = provider or get_default_provider()
        self.temperature = temperature
        self._labels: list[str] = []
        self._centroids: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def fit(
        self,
        examples: list[tuple[str, str]],
        *,
        descriptions: dict[str, str] | None = None,
    ) -> EmbeddingClassifier:
        """Fit centroids from ``(text, category)`` examples.

        ``descriptions`` supplies zero-shot centroids for categories that have a
        description but no (or few) labeled examples — the novel-category fallback.
        """
        by_category: dict[str, list[np.ndarray]] = {}
        if examples:
            texts = [t for t, _ in examples]
            vectors = self.provider.embed(texts)
            for (_, category), vec in zip(examples, vectors, strict=True):
                by_category.setdefault(category, []).append(vec)

        if descriptions:
            desc_labels = list(descriptions)
            desc_vecs = self.provider.embed([descriptions[c] for c in desc_labels])
            for category, vec in zip(desc_labels, desc_vecs, strict=True):
                by_category.setdefault(category, []).append(vec)

        self._labels = sorted(by_category)
        centroids = []
        for label in self._labels:
            mean = np.mean(np.vstack(by_category[label]), axis=0)
            norm = float(np.linalg.norm(mean))
            centroids.append(mean / norm if norm > 0 else mean)
        self._centroids = np.vstack(centroids) if centroids else np.zeros((0, 0), dtype=np.float32)
        return self

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def predict(self, text: str) -> tuple[str | None, float]:
        """Return ``(category, confidence)``; ``(None, 0.0)`` if unfit/empty embedding."""
        if not self._labels:
            return None, 0.0
        vec = self.provider.embed([text])[0]
        if float(np.linalg.norm(vec)) == 0.0:
            return None, 0.0
        sims = self._centroids @ vec
        probs = _softmax(sims, self.temperature)
        idx = int(np.argmax(sims))
        return self._labels[idx], float(probs[idx])

    def score(self, labeled: list[tuple[str, str]]) -> float:
        """Fraction of ``(text, category)`` pairs predicted correctly (held-out accuracy)."""
        if not labeled:
            return 0.0
        correct = sum(1 for text, cat in labeled if self.predict(text)[0] == cat)
        return correct / len(labeled)

    def save(self, path: str | Path) -> None:
        """Persist labels + centroids + temperature to a ``.npz`` artifact.

        The embedding *provider* is intentionally not persisted — centroids live in the
        provider's vector space, so a load must supply the same provider used to fit
        (``get_default_provider`` in production). ``load`` guards the dimension to catch a
        mismatched provider (e.g. fit under MiniLM, loaded under the hashing fallback)."""
        with open(path, "wb") as fh:
            np.savez(
                fh,
                labels=np.array(self._labels),
                centroids=self._centroids,
                temperature=np.array(self.temperature, dtype=np.float64),
            )

    @classmethod
    def load(
        cls, path: str | Path, provider: EmbeddingProvider | None = None
    ) -> EmbeddingClassifier:
        """Load a classifier saved by :meth:`save`. Raises ``ValueError`` if the supplied
        provider's embedding dimension does not match the persisted centroids (the
        silent-garbage guard)."""
        data = np.load(path)
        clf = cls(provider, temperature=float(data["temperature"]))
        clf._labels = [str(label) for label in data["labels"]]
        clf._centroids = data["centroids"]
        if clf._centroids.size:
            probe_dim = len(clf.provider.embed(["dimension probe"])[0])
            if probe_dim != clf._centroids.shape[1]:
                raise ValueError(
                    f"embedding provider dim {probe_dim} != persisted classifier dim "
                    f"{clf._centroids.shape[1]}; the classifier must be loaded with the same "
                    "embedding provider it was fit with (re-run `rumen fit-classifier`)."
                )
        return clf


def calibrate_tau(
    classifier: EmbeddingClassifier,
    labeled: list[tuple[str, str]],
    *,
    target_precision: float = 0.90,
    default: float = DEFAULT_TAU_CLASS,
) -> float:
    """Pick the ``tau_class`` whose accepted (confidence ≥ tau) slice hits ``target_precision``.

    Standard operating-point calibration (design D-12): predictions are sorted by
    confidence; the lowest threshold whose accepted set is at least ``target_precision``
    correct is returned. Falls back to ``default`` if no threshold reaches the target.
    """
    scored = []
    for text, gold in labeled:
        pred, conf = classifier.predict(text)
        scored.append((conf, pred == gold))
    scored.sort(reverse=True)  # highest confidence first

    best_tau = None
    correct = 0
    for accepted, (conf, is_correct) in enumerate(scored, start=1):
        correct += int(is_correct)
        if correct / accepted >= target_precision:
            best_tau = conf  # threshold that still holds precision
        else:
            break
    return best_tau if best_tau is not None else default


class FacetHMCClassifier:
    """Per-facet centroid HMC classifier with top-down path consistency.

    Each concept receives its own cosine-derived score, so a facet can receive multiple
    assignments.  A non-root concept is considered only after all of its recorded SKOS
    parents have cleared ``path_consistency_floor``.  Parent mappings are derived by the
    HMC fit layer from ``skos_relations``; this classifier intentionally stores no parallel
    hierarchy representation.
    """

    def __init__(
        self,
        facet: str,
        provider: EmbeddingProvider | None = None,
        *,
        parents: Mapping[int, set[int] | frozenset[int]] | None = None,
    ) -> None:
        self.facet = facet
        self.provider = provider or get_default_provider()
        self.parents = {
            concept: frozenset(parent_ids) for concept, parent_ids in (parents or {}).items()
        }
        self._labels: list[int] = []
        self._centroids: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @property
    def labels(self) -> list[int]:
        return list(self._labels)

    def fit(self, examples: list[tuple[str, int]]) -> FacetHMCClassifier:
        """Fit one centroid per concept from this facet's cleaned bootstrap examples."""
        by_concept: dict[int, list[np.ndarray]] = {}
        if examples:
            vectors = self.provider.embed([text for text, _ in examples])
            for (_, concept_id), vector in zip(examples, vectors, strict=True):
                by_concept.setdefault(concept_id, []).append(vector)
        self._labels = sorted(by_concept)
        centroids: list[np.ndarray] = []
        for concept_id in self._labels:
            mean = np.mean(np.vstack(by_concept[concept_id]), axis=0)
            norm = float(np.linalg.norm(mean))
            centroids.append(mean / norm if norm else mean)
        self._centroids = np.vstack(centroids) if centroids else np.zeros((0, 0), dtype=np.float32)
        return self

    def raw_scores(self, text: str) -> dict[int, float]:
        """Return independent per-concept cosine scores mapped from ``[-1, 1]`` to ``[0, 1]``."""
        if not self._labels:
            return {}
        vector = self.provider.embed([text])[0]
        if float(np.linalg.norm(vector)) == 0.0:
            return {concept_id: 0.0 for concept_id in self._labels}
        similarities = self._centroids @ vector
        return {
            concept_id: float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
            for concept_id, similarity in zip(self._labels, similarities, strict=True)
        }

    def assign(self, text: str, *, path_consistency_floor: float) -> dict[int, float]:
        """Assign concepts top-down, rejecting descendants whose parent path is below floor."""
        return self.assign_scores(
            self.raw_scores(text), path_consistency_floor=path_consistency_floor
        )

    def assign_scores(
        self, scores: Mapping[int, float], *, path_consistency_floor: float
    ) -> dict[int, float]:
        """Apply the hierarchy gate to supplied raw scores (also useful for deterministic tests)."""
        assignments: dict[int, float] = {}
        remaining = {concept_id for concept_id in self._labels if concept_id in scores}
        while remaining:
            progressed = False
            for concept_id in sorted(remaining):
                parent_ids = self.parents.get(concept_id, frozenset())
                if parent_ids & remaining:
                    continue
                # Every known parent must already have been accepted above the same floor.
                if parent_ids and not parent_ids.issubset(assignments):
                    remaining.remove(concept_id)
                    progressed = True
                    break
                if scores[concept_id] >= path_consistency_floor:
                    assignments[concept_id] = float(scores[concept_id])
                remaining.remove(concept_id)
                progressed = True
                break
            if not progressed:
                # A malformed cyclic SKOS path cannot establish a valid top-down assignment.
                break
        return assignments

    def save(self, path: str | Path) -> None:
        """Persist the per-facet centroids and SKOS parent ids using the Tier-1 ``.npz`` pattern."""
        parent_pairs = np.array(
            [(child, parent) for child, parents in self.parents.items() for parent in parents],
            dtype=np.int64,
        ).reshape((-1, 2))
        with open(path, "wb") as fh:
            np.savez(
                fh,
                facet=np.array(self.facet),
                labels=np.array(self._labels, dtype=np.int64),
                centroids=self._centroids,
                parent_pairs=parent_pairs,
            )

    @classmethod
    def load(
        cls, path: str | Path, provider: EmbeddingProvider | None = None
    ) -> FacetHMCClassifier:
        """Load a per-facet HMC artifact and validate its embedding dimension."""
        data = np.load(path)
        parent_map: dict[int, set[int]] = {}
        for child, parent in data["parent_pairs"]:
            parent_map.setdefault(int(child), set()).add(int(parent))
        classifier = cls(str(data["facet"]), provider, parents=parent_map)
        classifier._labels = [int(label) for label in data["labels"]]
        classifier._centroids = data["centroids"]
        if classifier._centroids.size:
            probe_dim = len(classifier.provider.embed(["dimension probe"])[0])
            if probe_dim != classifier._centroids.shape[1]:
                raise ValueError(
                    f"embedding provider dim {probe_dim} != persisted classifier dim "
                    f"{classifier._centroids.shape[1]}"
                )
        return classifier
