#!/usr/bin/env python3
"""Build canonical atomic rubrics from deterministic Route 2 construction truth.

Route 2 already has an authoritative scene graph and mix timeline, so this
adapter does not ask an LLM to rediscover them from prose.  It does, however,
use the same public AudioSceneGraph -> AtomicTuple -> DependencyRecord ->
RubricQuestion contract as the normal rubric-decomposition pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubric_decomposition.models import (  # noqa: E402
    Attribute,
    AudioSceneGraph,
    Event,
    GeneratedQuestion,
    Relation,
)
from rubric_decomposition.prompts import (  # noqa: E402
    deterministic_question,
    event_label_map,
    question_tasks,
)
from rubric_decomposition.tuples import build_tuples  # noqa: E402
from rubric_decomposition.validation import (  # noqa: E402
    build_dependencies,
    canonicalize_scene_graph,
    canonicalize_questions,
    validate_dependencies,
    validate_questions,
    validate_scene_graph,
)


ROLE_TO_EVENT_TYPE = {
    "speech": "speech",
    "sound_effect": "sound",
    "ambience": "ambience",
    "music": "music",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def event_name(event: dict[str, Any], language: str) -> str:
    return str(event["caption_label_zh"] if language == "zh" else event["caption_label_en"])


def event_identity(event: dict[str, Any], language: str) -> str:
    """Return only source/action identity; speech properties stay Attributes."""
    if event["role"] != "speech":
        return event_name(event, language)
    speaker = str(event.get("speaker") or "").strip().casefold()
    if language == "zh":
        if speaker in {"女性", "女", "woman", "female"}:
            return "一名女性说话"
        if speaker in {"男性", "男", "man", "male"}:
            return "一名男性说话"
        return "一名说话者说话"
    if speaker in {"女性", "女", "woman", "female"}:
        return "a woman speaking"
    if speaker in {"男性", "男", "man", "male"}:
        return "a man speaking"
    return "a speaker speaking"


def attribute_phrase(attribute: dict[str, Any], language: str) -> str:
    if attribute.get("caption_text"):
        return str(attribute["caption_text"])
    value = str(attribute.get("value") or "")
    if language != "zh":
        return value
    mapping = {
        "speed": {"慢": "语速缓慢", "中": "语速适中", "快": "语速较快"},
        "volume": {"低": "音量较低", "中": "音量适中", "高": "音量较高"},
        "pitch": {"低": "音调较低", "中": "音调适中", "高": "音调较高"},
    }
    return mapping.get(str(attribute.get("name")), {}).get(value, value)


def compact_relations(row: dict[str, Any]) -> list[dict[str, str]]:
    """Mirror the relation policy used by the positive caption generator."""
    events = sorted(
        row["events"], key=lambda value: (value["start"], value["end"], value["event_id"])
    )
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for left, right in zip(events, events[1:]):
        relation = "overlap" if right["start"] < left["end"] - 0.08 else "before"
        key = (left["event_id"], relation, right["event_id"])
        if key not in seen:
            result.append({
                "relation_id": f"R{len(result) + 1}",
                "source": key[0], "relation": key[1], "target": key[2],
            })
            seen.add(key)
    foreground = [event for event in events if event["role"] != "ambience"]
    for index, left in enumerate(foreground):
        for right in foreground[index + 1:]:
            if min(left["end"], right["end"]) - max(left["start"], right["start"]) <= 0.08:
                continue
            key = (left["event_id"], "overlap", right["event_id"])
            reverse = (right["event_id"], "overlap", left["event_id"])
            if key in seen or reverse in seen:
                continue
            result.append({
                "relation_id": f"R{len(result) + 1}",
                "source": key[0], "relation": key[1], "target": key[2],
            })
            seen.add(key)
    return result


def relation_evidence(row: dict[str, Any], language: str) -> dict[str, str]:
    audit = (row.get("caption_rewrite_audit_by_language") or {}).get(language) or {}
    return {
        str(item.get("relation_id")): str(item.get("quote") or "")
        for item in audit.get("relation_evidence") or []
    }


def verbatim_span(caption: str, value: str) -> str:
    """Return the source-caption glyphs for a case-insensitive locked span."""
    if value in caption:
        return value
    start = caption.casefold().find(value.casefold())
    if start < 0:
        return value
    return caption[start:start + len(value)]


def build_row(row: dict[str, Any]) -> dict[str, Any]:
    language = row["target_caption_language"]
    caption = row["caption_natural_by_language"][language]
    graph_events: list[Event] = []
    graph_attributes: list[Attribute] = []
    graph_relations: list[Relation] = []
    evidence_rows: list[dict[str, Any]] = []

    # build_tuples emits all presence tuples first, then attributes, then relations.
    for event in row["events"]:
        graph_events.append(Event(
            id=event["event_id"],
            type=ROLE_TO_EVENT_TYPE[event["role"]],
            label=event_identity(event, language),
            basis=verbatim_span(caption, event_name(event, language)),
        ))
        evidence_rows.append({
            "language": language,
            "event_id": event["event_id"],
            "event_identity": event_identity(event, language),
            "locked_caption_label": event_name(event, language),
            "interval": [event["start"], event["end"]],
        })

    attribute_evidence: list[dict[str, Any]] = []
    for event in row["events"]:
        if event["role"] != "speech":
            continue
        graph_attributes.append(Attribute(
            id=f"A{len(graph_attributes) + 1:03d}",
            owner_id=event["event_id"],
            family="content",
            type="speech_content",
            value=event["text"],
            basis=verbatim_span(caption, str(event["text"])),
        ))
        attribute_evidence.append({
            "language": language, "event_id": event["event_id"],
            "transcript": event["text"],
        })
        for attribute in event.get("selected_caption_attrs") or []:
            graph_attributes.append(Attribute(
                id=f"A{len(graph_attributes) + 1:03d}",
                owner_id=event["event_id"],
                family="acoustic",
                type="speech_attribute",
                value=attribute["value"],
                basis=verbatim_span(caption, attribute_phrase(attribute, language)),
            ))
            attribute_evidence.append({
                "language": language, "event_id": event["event_id"],
                "attribute": attribute,
            })
    evidence_rows.extend(attribute_evidence)

    by_id = {event["event_id"]: event for event in row["events"]}
    relation_quotes = relation_evidence(row, language)
    for relation in compact_relations(row):
        left, right = by_id[relation["source"]], by_id[relation["target"]]
        basis = relation_quotes.get(relation["relation_id"]) or caption
        graph_relations.append(Relation(
            id=relation["relation_id"], type=relation["relation"],
            source_id=relation["source"], target_id=relation["target"], basis=basis,
        ))
        evidence_rows.append({
            "language": language,
            "relation_id": relation["relation_id"],
            "source_event_id": relation["source"],
            "target_event_id": relation["target"],
            "source_interval": [left["start"], left["end"]],
            "target_interval": [right["start"], right["end"]],
            "caption_evidence": basis,
        })

    graph = canonicalize_scene_graph(caption, AudioSceneGraph(
        events=graph_events, attributes=graph_attributes, relations=graph_relations
    ))
    graph_errors = validate_scene_graph(caption, graph)
    if graph_errors:
        raise ValueError(f"{row['id']}: invalid scene graph: {graph_errors}")
    tuples = build_tuples(graph)
    dependencies = build_dependencies(tuples)
    dependency_errors = validate_dependencies(tuples, dependencies)
    if dependency_errors:
        raise ValueError(f"{row['id']}: invalid dependency graph: {dependency_errors}")

    tuple_dicts = [item.model_dump() for item in tuples]
    tasks = question_tasks(
        tuple_dicts,
        event_label_map(tuple_dicts),
        "CHINESE (中文)" if language == "zh" else "ENGLISH",
    )
    generated = [
        GeneratedQuestion(
            tuple_id=task["tuple_id"],
            question=deterministic_question(
                task, "CHINESE (中文)" if language == "zh" else "ENGLISH"
            ),
        )
        for task in tasks
    ]
    canonical = canonicalize_questions(
        tuples, dependencies, generated, prompt=caption, language_override=language
    )
    question_errors = validate_questions(
        caption, tuples, dependencies, canonical, language_override=language
    )
    if question_errors:
        raise ValueError(f"{row['id']}: invalid canonical questions: {question_errors}")
    if len(canonical) != len(evidence_rows):
        raise ValueError(f"{row['id']}: evidence/question count mismatch")

    questions = []
    for item, evidence in zip(canonical, evidence_rows):
        payload = item.model_dump()
        payload.update({
            "type": "audio",
            "construction_answer": "yes",
            "label_source": "deterministic_scene_graph_and_mix_timeline",
            "construction_evidence": evidence,
        })
        questions.append(payload)

    return {
        "sample_id": row["id"],
        "route": "path2_real_clip_composition",
        "audio_path": row["audio_path"],
        "language": language,
        "caption": caption,
        "scene_graph": graph.model_dump(),
        "atomic_tuples": tuple_dicts,
        "dependencies": [item.model_dump() for item in dependencies],
        "questions": questions,
        "rubric_count": len(questions),
        "construction_label_method": "canonical_atomic_rubric_graph_from_deterministic_route2_truth",
        "rubric_decomposition_contract": "rubric_decomposition",
        "scene_graph_origin": "deterministic_route2_truth_adapter",
        "validation": {
            "valid": True,
            "scene_graph_errors": [],
            "dependency_errors": [],
            "question_errors": [],
            "question_count": len(questions),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = [build_row(row) for row in read_jsonl(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    counts = [row["rubric_count"] for row in output]
    print(json.dumps({
        "samples": len(output), "rubrics": sum(counts),
        "min_per_sample": min(counts), "max_per_sample": max(counts),
        "schema": "scene_graph_to_atomic_tuples_to_dependencies_to_questions",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
