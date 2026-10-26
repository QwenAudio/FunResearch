#!/usr/bin/env python3
"""Freeze Route 2 A' pairs, original captions, and construction labels."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


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


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def relation_truth(subtype: str, left: dict[str, Any], right: dict[str, Any]) -> bool:
    if subtype == "before":
        return float(left["end"]) <= float(right["start"]) + 0.08
    if subtype == "overlap":
        return min(float(left["end"]), float(right["end"])) - max(
            float(left["start"]), float(right["start"])
        ) > 0.08
    raise ValueError(subtype)


def attribute_present(event: dict[str, Any], attribute: dict[str, Any]) -> bool:
    return any(
        str(item.get("name")) == str(attribute.get("name"))
        and str(item.get("value")) == str(attribute.get("value"))
        for item in event.get("selected_caption_attrs") or []
    )


def observed_evidence(question: dict[str, Any], events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = question["construction_evidence"]
    subtype = question["subtype"]
    if subtype == "exists":
        event = events[evidence["event_id"]]
        return {
            "event_id": event["event_id"], "interval": [event["start"], event["end"]],
            "source_path": event.get("source_path"), "caption_label_zh": event.get("caption_label_zh"),
            "caption_label_en": event.get("caption_label_en"),
        }
    if subtype == "speech_content":
        event = events[evidence["event_id"]]
        return {"event_id": event["event_id"], "transcript": event.get("text")}
    if subtype == "speech_attribute":
        event = events[evidence["event_id"]]
        return {"event_id": event["event_id"], "selected_caption_attrs": event.get("selected_caption_attrs") or []}
    left = events[evidence["source_event_id"]]
    right = events[evidence["target_event_id"]]
    return {
        "source_event_id": left["event_id"], "target_event_id": right["event_id"],
        "source_interval": [left["start"], left["end"]],
        "target_interval": [right["start"], right["end"]],
        "observed_relation_truth": relation_truth(subtype, left, right),
    }


def direct_target_realized(
    question: dict[str, Any], original_events: dict[str, dict[str, Any]],
    current_events: dict[str, dict[str, Any]],
) -> bool:
    evidence = question["construction_evidence"]
    subtype = question["subtype"]
    if subtype == "exists":
        event_id = evidence["event_id"]
        old, new = original_events[event_id], current_events[event_id]
        return (
            old.get("source_path") != new.get("source_path")
            and (
                old.get("caption_label_zh") != new.get("caption_label_zh")
                or old.get("caption_label_en") != new.get("caption_label_en")
            )
        )
    if subtype == "speech_content":
        return current_events[evidence["event_id"]].get("text") != evidence["transcript"]
    if subtype == "speech_attribute":
        return not attribute_present(current_events[evidence["event_id"]], evidence["attribute"])
    return not relation_truth(
        subtype,
        current_events[evidence["source_event_id"]],
        current_events[evidence["target_event_id"]],
    )


def retained_fact_true(question: dict[str, Any], events: dict[str, dict[str, Any]]) -> bool:
    evidence = question["construction_evidence"]
    subtype = question["subtype"]
    if subtype == "exists":
        return evidence["event_id"] in events
    if subtype == "speech_content":
        return events[evidence["event_id"]].get("text") == evidence["transcript"]
    if subtype == "speech_attribute":
        return attribute_present(events[evidence["event_id"]], evidence["attribute"])
    return relation_truth(
        subtype,
        events[evidence["source_event_id"]], events[evidence["target_event_id"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--composed", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    recipes = read_jsonl(args.recipes)
    composed = {row["id"]: row for row in read_jsonl(args.composed)}
    validation = {row["id"]: row for row in read_jsonl(args.validation)}
    missing = [row["negative_id"] for row in recipes if safe_id(row["negative_id"]) not in composed]
    invalid = [row_id for row_id, row in validation.items() if not row.get("valid")]
    if (missing or invalid) and not args.allow_partial:
        raise SystemExit(f"missing={len(missing)} invalid={len(invalid)} examples={(missing + invalid)[:8]}")

    pairs = []
    rubrics = []
    semantic_failures = []
    for recipe in recipes:
        row = composed.get(safe_id(recipe["negative_id"]))
        if row is None or not validation.get(row["id"], {}).get("valid"):
            continue
        audio_path = Path(row["audio_path"])
        original_events = {event["event_id"]: event for event in recipe["parent_scene"]["events"]}
        current_events = {event["event_id"]: event for event in row["events"]}
        direct_ids = set(recipe["direct_target_rubric_ids"])
        false_ids = set(recipe["hierarchy_affected_rubric_ids"])
        questions = []
        for source in recipe["original_questions"]:
            question = deepcopy(source)
            rubric_id = question["rubric_id"]
            if rubric_id in direct_ids:
                realized = direct_target_realized(question, original_events, current_events)
                if not realized:
                    semantic_failures.append({
                        "negative_id": recipe["negative_id"], "rubric_id": rubric_id,
                        "kind": "direct_target_not_realized", "subtype": question["subtype"],
                    })
                label_computation = "direct_target_realized_then_no"
            elif rubric_id in false_ids:
                label_computation = "hierarchy_parent_false_then_no"
            else:
                realized = retained_fact_true(question, current_events)
                if not realized:
                    semantic_failures.append({
                        "negative_id": recipe["negative_id"], "rubric_id": rubric_id,
                        "kind": "retained_fact_changed", "subtype": question["subtype"],
                    })
                label_computation = "retained_fact_verified_then_yes"
            question["construction_answer"] = "no" if question["rubric_id"] in false_ids else "yes"
            question["label_source"] = "deterministic_path2_controlled_audio_edit"
            question["observed_audio_evidence"] = observed_evidence(question, current_events)
            question["label_computation"] = label_computation
            questions.append(question)
        yes_count = sum(question["construction_answer"] == "yes" for question in questions)
        yes_ratio = yes_count / len(questions)
        low, high = recipe["construction_target_yes_range"]
        final_level = "low" if yes_ratio <= 0.33 else ("mid" if yes_ratio <= 0.66 else "high")
        pair = {
            **row,
            "schema": "route2_audio_side_negative_pair",
            "sample_id": recipe["negative_id"],
            "negative_id": recipe["negative_id"],
            "parent_positive_id": recipe["parent_positive_id"],
            "route": "path2_audio_side_negative",
            "positive_pair": False,
            "id": recipe["negative_id"],
            "caption": recipe["original_caption"],
            "scene_graph": recipe.get("original_scene_graph"),
            "atomic_tuples": recipe.get("original_atomic_tuples"),
            "dependencies": recipe.get("original_dependencies"),
            "original_caption": recipe["original_caption"],
            "counterfactual_truth_caption": row["audio_caption"],
            "audio_sha256": sha256(audio_path),
            "parent_audio_path": recipe["original_audio_path"],
            "parent_audio_sha256": recipe["original_audio_sha256"],
            "perturbation_category": recipe["perturbation_category"],
            "intended_consistency_level": recipe["intended_consistency_level"],
            "final_deterministic_consistency_level": final_level,
            "deterministic_yes_ratio": round(yes_ratio, 6),
            "construction_target_yes_range": recipe["construction_target_yes_range"],
            "construction_target_band_pass": low <= yes_ratio <= high,
            "direct_target_rubric_ids": recipe["direct_target_rubric_ids"],
            "hierarchy_affected_rubric_ids": recipe["hierarchy_affected_rubric_ids"],
            "audio_relation_materialization": recipe.get("audio_relation_materialization"),
        }
        pairs.append(pair)
        rubrics.append({
            "sample_id": recipe["negative_id"],
            "parent_positive_id": recipe["parent_positive_id"],
            "route": "path2_audio_side_negative",
            "audio_path": str(audio_path),
            "language": recipe["language"],
            "caption": recipe["original_caption"],
            "scene_graph": recipe.get("original_scene_graph"),
            "atomic_tuples": recipe.get("original_atomic_tuples"),
            "dependencies": recipe.get("original_dependencies"),
            "questions": questions,
            "rubric_count": len(questions),
            "yes_count": yes_count,
            "yes_ratio": round(yes_ratio, 6),
            "construction_label_method": "parent_labels_plus_deterministic_controlled_edit",
        })

    if semantic_failures and not args.allow_partial:
        failure_path = args.output_dir / "semantic_failures.jsonl"
        write_jsonl(failure_path, semantic_failures)
        raise SystemExit(
            f"semantic realization failures={len(semantic_failures)}; "
            f"samples={len({row['negative_id'] for row in semantic_failures})}; "
            f"details={failure_path}"
        )

    pairs.sort(key=lambda row: row["negative_id"])
    rubrics.sort(key=lambda row: row["sample_id"])
    pair_path = args.output_dir / "pairs.jsonl"
    rubric_path = args.output_dir / "construction_labels.jsonl"
    write_jsonl(pair_path, pairs)
    write_jsonl(rubric_path, rubrics)
    summary = {
        "schema": "route2_audio_side_negative_summary",
        "recipes": len(recipes),
        "materialized": len(pairs),
        "missing": len(missing),
        "invalid": len(invalid),
        "categories": dict(Counter(row["perturbation_category"] for row in pairs)),
        "intended_levels": dict(Counter(row["intended_consistency_level"] for row in pairs)),
        "final_levels": dict(Counter(row["final_deterministic_consistency_level"] for row in pairs)),
        "target_band_pass": sum(row["construction_target_band_pass"] for row in pairs),
        "target_band_fail": sum(not row["construction_target_band_pass"] for row in pairs),
        "unique_audio_sha256": len({row["audio_sha256"] for row in pairs}),
        "semantic_realization_failures": len(semantic_failures),
        "pairs_sha256": sha256(pair_path),
        "rubrics_sha256": sha256(rubric_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
