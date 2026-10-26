#!/usr/bin/env python3
"""Rebuild Route 2 P' construction labels from recorded atomic edits.

This is the fail-safe alternative to an unavailable remote text-model service.
It transforms the already validated positive scene/label graph with the exact
atomic edits recorded by the prompt-negative materializer. No label fact is
guessed from free-form text.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubric_decomposition.models import (  # noqa: E402
    Attribute, AudioSceneGraph, Event, GeneratedQuestion, Relation,
)
from rubric_decomposition.prompts import (  # noqa: E402
    deterministic_question, event_label_map, question_tasks,
)
from rubric_decomposition.tuples import build_tuples  # noqa: E402
from rubric_decomposition.validation import (  # noqa: E402
    build_dependencies, canonicalize_questions, canonicalize_scene_graph,
    validate_dependencies, validate_questions, validate_scene_graph,
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label(event: dict[str, Any], language: str) -> str:
    return str(event["caption_label_zh"] if language == "zh" else event["caption_label_en"])


def attr_phrase(attribute: dict[str, Any], language: str) -> str:
    if attribute.get("caption_text"):
        return str(attribute["caption_text"])
    if language == "zh":
        mapping = {
            "speed": {"慢": "语速缓慢", "中": "语速适中", "快": "语速较快"},
            "volume": {"低": "音量较低", "中": "音量适中", "高": "音量较高"},
            "pitch": {"低": "音调较低", "中": "音调适中", "高": "音调较高"},
        }
        return mapping.get(str(attribute.get("name")), {}).get(
            str(attribute.get("value")), str(attribute.get("value") or "")
        )
    return str(attribute.get("value") or "")


def event_identity(event: dict[str, Any], language: str) -> str:
    if event["role"] != "speech":
        return label(event, language)
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


def verbatim_span(caption: str, value: str) -> str:
    if value in caption:
        return value
    start = caption.casefold().find(value.casefold())
    return caption[start:start + len(value)] if start >= 0 else value


def canonical_graph(
    questions: list[dict[str, Any]], events: dict[str, dict[str, Any]],
    caption: str, language: str,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Canonicalize the edited graph with the authoritative decomposition code."""
    graph_events: list[Event] = []
    graph_attributes: list[Attribute] = []
    graph_relations: list[Relation] = []
    evidence_by_object: dict[int, dict[str, Any]] = {}
    role_type = {"speech": "speech", "sound_effect": "sound", "ambience": "ambience", "music": "music"}
    for question in questions:
        subtype = question["subtype"]
        evidence = question["construction_evidence"]
        if subtype == "exists":
            event = events[evidence["event_id"]]
            record = Event(
                id=event["event_id"], type=role_type[event["role"]],
                label=event_identity(event, language), basis=question["basis"],
            )
            graph_events.append(record)
        elif subtype in {"speech_content", "speech_attribute"}:
            record = Attribute(
                id=f"AT{len(graph_attributes) + 1}", owner_id=evidence["event_id"],
                family="content" if subtype == "speech_content" else "acoustic",
                type=subtype,
                value=(
                    evidence["transcript"] if subtype == "speech_content"
                    else evidence["attribute"]["value"]
                ),
                unit=None, basis=question["basis"],
            )
            graph_attributes.append(record)
        else:
            record = Relation(
                id=f"REL{len(graph_relations) + 1}", type=subtype,
                source_id=evidence["source_event_id"],
                target_id=evidence["target_event_id"], basis=question["basis"],
            )
            graph_relations.append(record)
        evidence_by_object[id(record)] = deepcopy(question)

    graph = canonicalize_scene_graph(caption, AudioSceneGraph(
        events=graph_events, attributes=graph_attributes, relations=graph_relations,
    ))
    if errors := validate_scene_graph(caption, graph):
        raise ValueError(f"canonical P-prime scene graph failed validation: {errors}")
    ordered_records = [*graph.events, *graph.attributes, *graph.relations]
    if any(id(record) not in evidence_by_object for record in ordered_records):
        raise ValueError("scene graph canonicalization introduced an unmapped record")

    tuples = build_tuples(graph)
    dependencies = build_dependencies(tuples)
    if errors := validate_dependencies(tuples, dependencies):
        raise ValueError(f"canonical P-prime dependencies failed validation: {errors}")
    tuple_dicts = [item.model_dump() for item in tuples]
    target_language = "CHINESE (中文)" if language == "zh" else "ENGLISH"
    tasks = question_tasks(tuple_dicts, event_label_map(tuple_dicts), target_language)
    generated = [
        GeneratedQuestion(
            tuple_id=task["tuple_id"],
            question=deterministic_question(task, target_language),
        )
        for task in tasks
    ]
    canonical_questions = canonicalize_questions(
        tuples, dependencies, generated, caption, language
    )
    if errors := validate_questions(
        caption, tuples, dependencies, canonical_questions, language
    ):
        raise ValueError(f"canonical P-prime questions failed validation: {errors}")

    enriched_questions = []
    for canonical, record in zip(canonical_questions, ordered_records):
        source = evidence_by_object[id(record)]
        payload = canonical.model_dump()
        payload.update({
            "type": source.get("type", "audio"),
            "construction_answer": "yes",
            "label_source": "deterministic_pprime_atomic_edit_graph",
            "construction_evidence": deepcopy(source["construction_evidence"]),
            "source_positive_rubric_id": source.get(
                "source_positive_rubric_id", source["rubric_id"]
            ),
        })
        enriched_questions.append(payload)
    return (
        graph.model_dump(), tuple_dicts,
        [item.model_dump() for item in dependencies], enriched_questions,
    )


def make_question(
    subtype: str, evidence: dict[str, Any], events: dict[str, dict[str, Any]], language: str
) -> str:
    if subtype == "exists":
        event_label = event_identity(events[evidence["event_id"]], language)
        return (
            f"音频中是否能听到{event_label}？"
            if language == "zh" else f"Can {event_label} be heard?"
        )
    if subtype == "speech_content":
        transcript = str(evidence["transcript"])
        return (
            f"说话者是否说了“{transcript}”？"
            if language == "zh" else f'Does the speaker say, "{transcript}"?'
        )
    if subtype == "speech_attribute":
        value = str(evidence["attribute"]["value"])
        return (
            f"说话是否具有属性“{value}”？"
            if language == "zh" else f'Does the speech have the attribute "{value}"?'
        )
    source = event_identity(events[evidence["source_event_id"]], language)
    target = event_identity(events[evidence["target_event_id"]], language)
    if subtype == "before":
        return (
            f"{source}是否先于{target}出现？"
            if language == "zh" else f"Does {source} occur before {target}?"
        )
    if subtype == "overlap":
        return (
            f"{source}是否与{target}在时间上重叠？"
            if language == "zh" else f"Do {source} and {target} overlap in time?"
        )
    if subtype == "non_overlap":
        return (
            f"{source}是否不与{target}在时间上重叠？"
            if language == "zh" else f"Do {source} and {target} not overlap in time?"
        )
    raise ValueError(f"unsupported subtype: {subtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--positive-pairs", type=Path, required=True)
    parser.add_argument("--positive-construction-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jobs = {row["sample_id"]: row for row in read_jsonl(args.run / "prepared/prompt_jobs.jsonl")}
    prompt_pairs = {row["sample_id"]: row for row in read_jsonl(args.run / "prompt_side/pairs.jsonl")}
    audits = {row["sample_id"]: row for row in read_jsonl(args.run / "prompt_side/semantic_repair_audit.jsonl")}
    positive_pairs = {row["id"]: row for row in read_jsonl(args.positive_pairs)}
    positive_rubrics = {
        row["sample_id"]: row for row in read_jsonl(args.positive_construction_labels)
    }
    speech_pool = {
        row["uid"]: row for row in read_jsonl(args.run / "assets/speech_pool.jsonl")
    }
    if not jobs or not (len(jobs) == len(prompt_pairs) == len(audits)):
        raise ValueError(
            "jobs, prompt pairs, and semantic audits must be non-empty and have equal counts"
        )

    output_rows = []
    invalid = []
    for sample_id, job in jobs.items():
        language = job["language"]
        pair = positive_pairs[job["parent_positive_id"]]
        original = positive_rubrics[job["parent_positive_id"]]
        events = {event["event_id"]: deepcopy(event) for event in pair["events"]}
        audit = audits[sample_id]
        prompt_prime = prompt_pairs[sample_id]["caption"]
        repair_by_rubric = {item["rubric_id"]: item for item in audit.get("repairs") or []}

        # Apply recorded identity/content/attribute edits to a copy of the scene.
        for question in job["selected_rubrics"]:
            repair = repair_by_rubric.get(question["rubric_id"])
            if repair is None or repair.get("kind") in {"relation_negation", "relation_reversal"}:
                continue
            evidence = question["construction_evidence"]
            event = events[evidence["event_id"]]
            if repair["kind"] == "event_identity":
                event["caption_label_zh" if language == "zh" else "caption_label_en"] = repair["replacement_label"]
            elif repair["kind"] == "speech_event_identity":
                source = speech_pool[repair["replacement_source_id"]]
                event["speaker_description"] = source["speaker_description"]
                event["caption_label_zh"] = source["speaker_description"]
                event["caption_label_en"] = source["speaker_description"]
                event["text"] = source["text"]
                event["selected_caption_attrs"] = deepcopy(source.get("selected_caption_attrs") or [])
            elif repair["kind"] == "speech_content":
                source = speech_pool[repair["replacement_source_id"]]
                event["text"] = source["text"]
            elif repair["kind"] == "speech_attribute":
                old_attribute = evidence["attribute"]
                candidate = deepcopy(repair.get("new_attribute"))
                if not candidate:
                    source = speech_pool[repair["replacement_source_id"]]
                    candidate = next(
                        item for item in source.get("selected_caption_attrs") or []
                        if str(item.get("name")) == str(old_attribute.get("name"))
                        and str(item.get("value")) == str(repair["new_value"])
                    )
                old_phrase = attr_phrase(old_attribute, language)
                new_phrase = attr_phrase(candidate, language)
                event["speaker_description"] = str(event["speaker_description"]).replace(old_phrase, new_phrase)
                event["caption_label_zh"] = event["speaker_description"]
                event["caption_label_en"] = event["speaker_description"]
                attrs = deepcopy(event.get("selected_caption_attrs") or [])
                for index, attribute in enumerate(attrs):
                    if str(attribute.get("name")) == str(old_attribute.get("name")) \
                            and str(attribute.get("value")) == str(old_attribute.get("value")):
                        attrs[index] = deepcopy(candidate)
                        break
                event["selected_caption_attrs"] = attrs

        old_to_new_id: dict[str, str] = {}
        questions = []
        original_questions = original["questions"]
        for index, question in enumerate(original_questions, start=1):
            old_to_new_id[question["rubric_id"]] = f"R{index:03d}"

        selected_relation_ids = {
            question["rubric_id"] for question in job["selected_rubrics"]
            if question["subtype"] in {"before", "overlap"}
        }
        event_starts = {event_id: float(event["start"]) for event_id, event in events.items()}
        presence_rubric_by_event = {
            item["construction_evidence"]["event_id"]: old_to_new_id[item["rubric_id"]]
            for item in original_questions if item["subtype"] == "exists"
        }
        local_errors = []
        for original_question in original_questions:
            question = deepcopy(original_question)
            old_id = question["rubric_id"]
            subtype = question["subtype"]
            evidence = deepcopy(question["construction_evidence"])
            evidence["language"] = language

            if subtype == "speech_content":
                evidence["transcript"] = events[evidence["event_id"]]["text"]
            elif subtype == "speech_attribute":
                old_attribute = evidence["attribute"]
                matching = [
                    attribute for attribute in events[evidence["event_id"]].get("selected_caption_attrs") or []
                    if str(attribute.get("name")) == str(old_attribute.get("name"))
                ]
                if matching:
                    evidence["attribute"] = deepcopy(matching[0])
                elif label(events[evidence["event_id"]], language) != label(
                    {event["event_id"]: event for event in pair["events"]}[evidence["event_id"]], language
                ):
                    # A selected speech identity replaces the whole speaker,
                    # transcript, and its 2-3 caption attributes.  Preserve the
                    # original rubric granularity by mapping attributes in their
                    # caption order.
                    original_attribute_questions = [
                        item for item in original_questions
                        if item["subtype"] == "speech_attribute"
                        and item["construction_evidence"]["event_id"] == evidence["event_id"]
                    ]
                    position = next(
                        i for i, item in enumerate(original_attribute_questions)
                        if item["rubric_id"] == old_id
                    )
                    new_attributes = events[evidence["event_id"]].get("selected_caption_attrs") or []
                    if position >= len(new_attributes):
                        local_errors.append(f"speech_attribute_cardinality:{old_id}")
                    else:
                        evidence["attribute"] = deepcopy(new_attributes[position])
            elif subtype in {"before", "overlap"}:
                source = evidence["source_event_id"]
                target = evidence["target_event_id"]
                if job["path2_perturbation_category"] == "relation":
                    # The relation-only constructor reverses every event into a
                    # strict non-overlapping sequence.
                    if event_starts[source] < event_starts[target]:
                        source, target = target, source
                    evidence["source_event_id"] = source
                    evidence["target_event_id"] = target
                    subtype = "before"
                elif old_id in selected_relation_ids:
                    if subtype == "before":
                        evidence["source_event_id"], evidence["target_event_id"] = target, source
                        subtype = "before"
                    else:
                        source_label = label(events[source], language)
                        target_label = label(events[target], language)
                        if prompt_prime.casefold().find(source_label.casefold()) > prompt_prime.casefold().find(target_label.casefold()):
                            evidence["source_event_id"], evidence["target_event_id"] = target, source
                        subtype = "before"

            if subtype == "exists":
                basis = verbatim_span(prompt_prime, label(events[evidence["event_id"]], language))
                target_ids = [evidence["event_id"]]
            elif subtype == "speech_content":
                basis = verbatim_span(prompt_prime, str(evidence["transcript"]))
                target_ids = [evidence["event_id"]]
            elif subtype == "speech_attribute":
                basis = verbatim_span(prompt_prime, attr_phrase(evidence["attribute"], language))
                target_ids = [evidence["event_id"]]
            else:
                basis = prompt_prime
                target_ids = [evidence["source_event_id"], evidence["target_event_id"]]

            parent_ids = [old_to_new_id[parent] for parent in question.get("parent_ids") or []]
            if subtype in {"before", "overlap"}:
                parent_ids = [presence_rubric_by_event[event_id] for event_id in target_ids]
            question.update({
                "rubric_id": old_to_new_id[old_id],
                "parent_ids": parent_ids,
                "subtype": subtype,
                "basis": basis,
                "target_ids": target_ids,
                "expected_answer": "yes",
                "question": make_question(subtype, evidence, events, language),
                "construction_answer": "yes",
                "label_source": "deterministic_pprime_atomic_edit_graph",
                "construction_evidence": evidence,
                "source_positive_rubric_id": old_id,
            })
            questions.append(question)

            # Direct textual coverage checks for every local fact.
            if subtype == "exists" and events[evidence["event_id"]]["role"] != "speech":
                needle = label(events[evidence["event_id"]], language)
                if needle.casefold() not in prompt_prime.casefold():
                    local_errors.append(f"missing_event:{old_id}:{needle}")
            elif subtype == "speech_content":
                needle = str(evidence["transcript"])
                if needle.casefold() not in prompt_prime.casefold():
                    local_errors.append(f"missing_transcript:{old_id}")
            elif subtype == "speech_attribute":
                needle = attr_phrase(evidence["attribute"], language)
                if needle and needle.casefold() not in prompt_prime.casefold():
                    local_errors.append(f"missing_attribute:{old_id}:{needle}")

        scene_graph, atomic_tuples, dependencies, questions = canonical_graph(
            questions, events, prompt_prime, language
        )
        rubric_ids = {question["rubric_id"] for question in questions}
        validation = {
            "valid": not local_errors,
            "errors": local_errors,
            "scene_graph_errors": [],
            "dependency_errors": [],
            "question_errors": [],
            "question_count": len(questions),
            "all_parent_ids_resolve": all(
                set(question.get("parent_ids") or []) <= rubric_ids
                for question in questions
            ),
            "method": "deterministic_pprime_atomic_edit_graph",
        }
        validation["valid"] = validation["valid"] and validation["all_parent_ids_resolve"]
        if not validation["valid"]:
            invalid.append({"sample_id": sample_id, "errors": local_errors[:10]})
        output_rows.append({
            "sample_id": sample_id,
            "route": "path2_prompt_side_negative",
            "audio_path": prompt_pairs[sample_id]["audio_path"],
            "language": language,
            "caption": prompt_prime,
            "scene_graph": scene_graph,
            "atomic_tuples": atomic_tuples,
            "dependencies": dependencies,
            "questions": questions,
            "rubric_count": len(questions),
            "construction_label_method": "deterministic_transform_of_validated_scene_and_label_graph",
            "rubric_decomposition_contract": "rubric_decomposition",
            "scene_graph_origin": "deterministic_pprime_atomic_edit_adapter",
            "source_positive_id": job["parent_positive_id"],
            "source_semantic_audit_sha256": sha256(args.run / "prompt_side/semantic_repair_audit.jsonl"),
            "validation": validation,
        })

    if invalid:
        raise ValueError(f"invalid deterministic P' rubrics: {len(invalid)}; first={invalid[:3]}")
    output_rows.sort(key=lambda row: row["sample_id"])
    write_jsonl(args.output, output_rows)
    summary = {
        "method": "deterministic_pprime_atomic_edit_graph",
        "rows": len(output_rows),
        "questions": sum(len(row["questions"]) for row in output_rows),
        "valid": sum(row["validation"]["valid"] for row in output_rows),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
