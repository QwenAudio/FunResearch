from __future__ import annotations

from typing import Any

from .models import AtomicTuple, AudioSceneGraph


TEMPORAL_ATTRIBUTES = {"occurrence_count", "count", "repetition_pattern", "repetition_rate", "repetition_interval", "duration", "absolute_start", "absolute_end", "temporal_position", "temporal_coverage"}
ACOUSTIC_ATTRIBUTES = {"loudness", "relative_loudness", "distance", "left_right", "reverberation", "pitch", "timbre", "foreground_background", "fade_in", "fade_out", "increase", "decrease", "continuity"}
TEMPORAL_RELATIONS = {"before", "after", "overlap"}


def _domain_for_node(graph: AudioSceneGraph, node_id: str) -> list[str]:
    event = next((event for event in graph.events if event.id == node_id), None)
    if event:
        return [event.type]
    return ["scene"]


def build_tuples(graph: AudioSceneGraph) -> list[AtomicTuple]:
    rows: list[AtomicTuple] = []

    def add(**kwargs: Any) -> None:
        rows.append(AtomicTuple(tuple_id=f"T{len(rows)+1:03d}", **kwargs))

    for event in graph.events:
        add(predicate_type="presence", predicate="exists", subject_ids=[event.id],
            object_value=event.label, basis=event.basis, axis="content_fidelity",
            scope="local", domain=[event.type])

    for attribute in graph.attributes:
        kind = attribute.type.lower()
        if attribute.family in {"quantity", "temporal"} or kind in TEMPORAL_ATTRIBUTES:
            axis = "temporal_structural_fidelity"
        elif attribute.family in {"acoustic", "dynamic", "spatial", "scene"} or kind in ACOUSTIC_ATTRIBUTES:
            axis = "acoustic_presentation_fidelity"
        else:
            axis = "content_fidelity"
        add(predicate_type="attribute", predicate=attribute.type,
            subject_ids=[attribute.owner_id], object_value=attribute.value,
            object_unit=attribute.unit,
            basis=attribute.basis, axis=axis,
            scope="global" if attribute.owner_id == "SCENE" else "local",
            domain=_domain_for_node(graph, attribute.owner_id))

    for relation in graph.relations:
        axis = "temporal_structural_fidelity" if relation.type in TEMPORAL_RELATIONS else (
            "acoustic_presentation_fidelity" if relation.type in {"louder_than", "masks", "foreground_of", "background_of", "near", "far", "left_of", "right_of"}
            else "content_fidelity"
        )
        domains = sorted(set(_domain_for_node(graph, relation.source_id) + _domain_for_node(graph, relation.target_id)))
        add(predicate_type="relation", predicate=relation.type,
            subject_ids=[relation.source_id, relation.target_id], object_value=None,
            basis=relation.basis, axis=axis, scope="relational", domain=domains)
    return rows
