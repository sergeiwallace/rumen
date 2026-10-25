"""Skosify/qSKOS-style validation for the D-24 RDF taxonomy source of truth."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, SKOS


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding, including the RDF concept URI(s) it affects."""

    code: str
    concept_ids: tuple[str, ...]
    message: str


class TaxonomyValidationError(ValueError):
    """Raised when a scheme cannot safely be materialized."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        super().__init__(details)


_RELATION_PREDICATES = (
    SKOS.broader,
    SKOS.narrower,
    SKOS.related,
    SKOS.broadMatch,
    SKOS.closeMatch,
    SKOS.exactMatch,
)


def validate_scheme(graph: Graph) -> list[ValidationIssue]:
    """Run Skosify plus the qSKOS checks required by D-24.

    Skosify is invoked on a copy, preserving the authoring graph exactly as loaded.  Its
    transformation/QA pass is supplemented with deterministic checks for orphan concepts,
    hierarchy cycles, duplicate preferred labels, and disconnected components; those are
    reported with RDF identifiers so authors can repair the source directly.
    """
    issues = _skosify_issues(graph)
    schemes = _scheme_concepts(graph)
    for concepts in schemes.values():
        issues.extend(_orphan_issues(graph, concepts))
        issues.extend(_cycle_issues(graph, concepts))
        issues.extend(_duplicate_label_issues(graph, concepts))
        issues.extend(_disconnected_component_issues(graph, concepts))
    return issues


def ensure_valid(graph: Graph) -> None:
    """Raise a single fail-closed error if any Skosify/qSKOS check fails."""
    issues = validate_scheme(graph)
    if issues:
        raise TaxonomyValidationError(issues)


def _skosify_issues(graph: Graph) -> list[ValidationIssue]:
    """Run Skosify's QA/normalization pass without mutating the input graph."""
    try:
        import skosify
    except ImportError as exc:  # pragma: no cover - only possible without the taxonomy extra
        raise RuntimeError(
            "SKOS validation requires the taxonomy extra; install rumen[taxonomy]"
        ) from exc

    copied = Graph()
    for prefix, namespace in graph.namespaces():
        copied.bind(prefix, namespace)
    for triple in graph:
        copied.add(triple)
    try:
        skosify.skosify(copied)
    except Exception as exc:  # noqa: BLE001 - Skosify uses multiple exception types for malformed RDF/SKOS.
        return [ValidationIssue("skosify_validation", (), str(exc))]
    return []


def _scheme_concepts(graph: Graph) -> dict[URIRef, set[URIRef]]:
    schemes = {scheme for scheme in graph.subjects(RDF.type, SKOS.ConceptScheme)}
    schemes.update(graph.objects(None, SKOS.inScheme))
    result: dict[URIRef, set[URIRef]] = {}
    for scheme in schemes:
        concepts = {
            concept
            for concept in graph.subjects(SKOS.inScheme, scheme)
            if (concept, RDF.type, SKOS.Concept) in graph
        }
        result[scheme] = concepts
    return result


def _orphan_issues(graph: Graph, concepts: set[URIRef]) -> list[ValidationIssue]:
    if len(concepts) == 1:
        return []
    issues = []
    for concept in sorted(concepts, key=str):
        connected = any(
            (concept, predicate, other) in graph or (other, predicate, concept) in graph
            for predicate in _RELATION_PREDICATES
            for other in concepts
            if other != concept
        )
        if not connected:
            issues.append(
                ValidationIssue("orphan_concept", (str(concept),), "concept has no SKOS relation")
            )
    return issues


def _cycle_issues(graph: Graph, concepts: set[URIRef]) -> list[ValidationIssue]:
    parents: dict[URIRef, set[URIRef]] = {concept: set() for concept in concepts}
    for child in concepts:
        parents[child].update(
            parent for parent in graph.objects(child, SKOS.broader) if parent in concepts
        )
        for parent in graph.objects(child, SKOS.narrower):
            if parent in concepts:
                parents[parent].add(child)

    visited: set[URIRef] = set()
    active: list[URIRef] = []
    active_set: set[URIRef] = set()
    cycles: set[tuple[str, ...]] = set()

    def visit(concept: URIRef) -> None:
        visited.add(concept)
        active.append(concept)
        active_set.add(concept)
        for parent in parents[concept]:
            if parent in active_set:
                start = active.index(parent)
                cycles.add(tuple(sorted(str(item) for item in active[start:])))
            elif parent not in visited:
                visit(parent)
        active.pop()
        active_set.remove(concept)

    for concept in sorted(concepts, key=str):
        if concept not in visited:
            visit(concept)
    return [
        ValidationIssue("hierarchy_cycle", cycle, "broader/narrower hierarchy contains a cycle")
        for cycle in sorted(cycles)
    ]


def _duplicate_label_issues(graph: Graph, concepts: set[URIRef]) -> list[ValidationIssue]:
    labels: dict[str, list[URIRef]] = defaultdict(list)
    for concept in concepts:
        for label in graph.objects(concept, SKOS.prefLabel):
            labels[str(label).strip().casefold()].append(concept)
    return [
        ValidationIssue(
            "duplicate_pref_label",
            tuple(sorted(str(concept) for concept in duplicates)),
            f"ambiguous prefLabel {label!r} within one scheme",
        )
        for label, duplicates in sorted(labels.items())
        if len(set(duplicates)) > 1
    ]


def _disconnected_component_issues(graph: Graph, concepts: set[URIRef]) -> list[ValidationIssue]:
    if len(concepts) < 2:
        return []
    neighbours: dict[URIRef, set[URIRef]] = {concept: set() for concept in concepts}
    for predicate in _RELATION_PREDICATES:
        for source, target in graph.subject_objects(predicate):
            if source in concepts and target in concepts:
                neighbours[source].add(target)
                neighbours[target].add(source)

    remaining = set(concepts)
    components = []
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            current = frontier.pop()
            discovered = neighbours[current] & remaining
            remaining -= discovered
            component.update(discovered)
            frontier.extend(discovered)
        components.append(component)
    if len(components) < 2:
        return []
    return [
        ValidationIssue(
            "disconnected_components",
            tuple(sorted(str(concept) for component in components for concept in component)),
            "scheme has disconnected concept components",
        )
    ]
