#!/usr/bin/env python3
"""Join Route 2 positives, construction labels, and negative assignments.

The output is split into two restartable inputs:

* prompt_jobs.jsonl: exact-span P -> P' edit jobs for a user-selected text model.
* audio_recipes.jsonl: deterministic scene-graph edits for A -> A'.

This stage does not call a model and does not render audio.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
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


def event_label(event: dict[str, Any], language: str) -> str:
    return str(event["caption_label_zh"] if language == "zh" else event["caption_label_en"])


def semantic_ids(question: dict[str, Any]) -> list[str]:
    evidence = question.get("construction_evidence") or {}
    values = []
    for key in ("event_id", "source_event_id", "target_event_id"):
        if evidence.get(key):
            values.append(str(evidence[key]))
    return values


def replacement_candidates(
    pair: dict[str, Any], assets: list[dict[str, Any]], rng: random.Random, limit: int = 18
) -> list[dict[str, Any]]:
    language = pair["target_caption_language"]
    occupied_slugs = {event.get("asset_slug") for event in pair["events"]}
    occupied_families = {event.get("acoustic_confusion_family") for event in pair["events"]}
    candidates = [
        row for row in assets
        if row.get("slug") not in occupied_slugs
        and row.get("slug") not in occupied_families
    ]
    rng.shuffle(candidates)
    chosen = []
    seen_labels = set()
    for row in candidates:
        label = str(row.get("label") if language == "zh" else row.get("event_name") or "").strip()
        if not label or label.casefold() in seen_labels:
            continue
        chosen.append({
            "source_dataset": row.get("source_dataset", "AudioSet-Strong"),
            "source_id": row.get("video_id"),
            "role": row.get("role"),
            "slug": row.get("slug"),
            "label": label,
        })
        seen_labels.add(label.casefold())
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        raise ValueError(f"insufficient dynamic labels for {pair['id']}: {len(chosen)}")
    return chosen


def speech_replacement_candidates(
    pair: dict[str, Any], speech_pool: list[dict[str, Any]], rng: random.Random, limit: int = 6
) -> list[dict[str, Any]]:
    occupied_text = {
        str(event.get("text") or "").strip().casefold()
        for event in pair["events"] if event.get("role") == "speech"
    }
    rows = [
        row for row in speech_pool
        if row.get("language") == pair["target_caption_language"]
        and str(row.get("text") or "").strip().casefold() not in occupied_text
    ]
    requested = min(limit, len(rows))
    if requested == 0:
        raise ValueError(f"no speech text candidates for {pair['id']}")
    rng.shuffle(rows)
    chosen = []
    seen = set()
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text or text.casefold() in seen:
            continue
        chosen.append({
            "candidate_type": "speech_content",
            "source_dataset": row["source_dataset"],
            "source_id": row["uid"],
            "language": row["language"],
            "speaker_description": row["speaker_description"],
            "text": text,
            "label": text,
        })
        seen.add(text.casefold())
        if len(chosen) >= requested:
            break
    if len(chosen) < requested:
        raise ValueError(f"insufficient speech text candidates for {pair['id']}: {len(chosen)}")
    return chosen


def prompt_job(
    assignment: dict[str, Any], pair: dict[str, Any], rubric: dict[str, Any],
    assets: list[dict[str, Any]], speech_pool: list[dict[str, Any]], rng: random.Random,
) -> dict[str, Any]:
    by_rubric = {question["rubric_id"]: question for question in rubric["questions"]}
    selected = [by_rubric[rid] for rid in assignment["direct_target_rubric_ids"]]
    event_ids = sorted({value for question in selected for value in semantic_ids(question)})
    category = assignment["perturbation_category"]
    preserve_anchors = []
    if category in {"attribute", "relation"}:
        # Attribute edits are allowed to modify the selected speech phrase, so
        # only non-target event anchors are immutable. Relation edits retain
        # every event identity and therefore preserve every event anchor.
        preserve_anchors = [
            event_label(event, pair["target_caption_language"])
            for event in pair["events"]
            if category == "relation" or event["event_id"] not in event_ids
        ]
    candidates = replacement_candidates(pair, assets, rng) if category in {"event", "all_three"} else []
    if any(question["subtype"] == "speech_content" for question in selected):
        candidates.extend(speech_replacement_candidates(pair, speech_pool, rng))
    return {
        "schema": "route2_prompt_negative_job",
        "sample_id": assignment["negative_id"],
        "negative_id": assignment["negative_id"],
        "parent_positive_id": pair["id"],
        "audio_path": pair["audio_path"],
        "prompt": pair["caption"],
        "language": pair["target_caption_language"],
        "disturbance_category": "mixed" if category == "all_three" else category,
        "path2_perturbation_category": category,
        "intended_consistency_level": assignment["intended_consistency_level"],
        "construction_target_yes_range": assignment["construction_target_yes_range"],
        "selected_rubric_ids": assignment["direct_target_rubric_ids"],
        "hierarchy_affected_rubric_ids": assignment["hierarchy_affected_rubric_ids"],
        "selected_rubrics": selected,
        "selected_semantic_target_ids": event_ids,
        "must_preserve_event_anchors": preserve_anchors,
        "replacement_candidates": candidates,
        "parent_audio_sha256": sha256(Path(pair["audio_path"])),
        "sampling_seed": assignment["sampling_seed"],
    }


def audio_recipe(
    assignment: dict[str, Any], pair: dict[str, Any], rubric: dict[str, Any]
) -> dict[str, Any]:
    by_rubric = {question["rubric_id"]: question for question in rubric["questions"]}
    primitives = []
    for rubric_id in assignment["direct_target_rubric_ids"]:
        question = by_rubric[rubric_id]
        evidence = question["construction_evidence"]
        subtype = question["subtype"]
        if subtype == "exists":
            primitive = {"type": "event", "event_id": evidence["event_id"]}
        elif subtype == "speech_content":
            primitive = {"type": "attribute", "subtype": "speech_content", "event_id": evidence["event_id"]}
        elif subtype == "speech_attribute":
            primitive = {
                "type": "attribute", "subtype": "speech_attribute",
                "event_id": evidence["event_id"], "attribute": evidence["attribute"],
            }
        elif subtype in {"before", "overlap"}:
            primitive = {
                "type": "relation", "subtype": subtype,
                "source_event_id": evidence["source_event_id"],
                "target_event_id": evidence["target_event_id"],
            }
        else:
            raise ValueError(f"unsupported target subtype {subtype}: {rubric_id}")
        primitives.append({"rubric_id": rubric_id, **primitive})
    return {
        "schema": "route2_audio_negative_recipe",
        "sample_id": assignment["negative_id"],
        "negative_id": assignment["negative_id"],
        "parent_positive_id": pair["id"],
        "language": pair["target_caption_language"],
        "original_caption": pair["caption"],
        "original_audio_path": pair["audio_path"],
        "original_audio_sha256": sha256(Path(pair["audio_path"])),
        "perturbation_category": assignment["perturbation_category"],
        "intended_consistency_level": assignment["intended_consistency_level"],
        "construction_target_yes_range": assignment["construction_target_yes_range"],
        "direct_target_rubric_ids": assignment["direct_target_rubric_ids"],
        "hierarchy_affected_rubric_ids": assignment["hierarchy_affected_rubric_ids"],
        "retained_rubric_ids": assignment["retained_rubric_ids"],
        "original_questions": rubric["questions"],
        "original_scene_graph": rubric.get("scene_graph"),
        "original_atomic_tuples": rubric.get("atomic_tuples"),
        "original_dependencies": rubric.get("dependencies"),
        "primitives": primitives,
        "parent_scene": pair,
        "sampling_seed": assignment["sampling_seed"],
        "audio_relation_materialization": assignment.get("audio_relation_materialization"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--construction-labels", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--speech-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    assignments = read_jsonl(args.plan)
    pairs = {row["id"]: row for row in read_jsonl(args.pairs)}
    rubrics = {row["sample_id"]: row for row in read_jsonl(args.construction_labels)}
    assets = read_jsonl(args.assets)
    speech_pool = read_jsonl(args.speech_pool)
    if not assignments or not pairs or not rubrics:
        raise ValueError("plan, pairs, and construction labels must be non-empty")
    if not (len(assignments) == len(pairs) == len(rubrics)):
        raise ValueError(
            "plan/pair/rubric row counts differ: "
            f"{len(assignments)}/{len(pairs)}/{len(rubrics)}"
        )
    if {row["parent_positive_id"] for row in assignments} != set(pairs) or set(pairs) != set(rubrics):
        raise ValueError("plan, pairs, and rubrics do not have the same positive ids")

    prompt_rows = []
    audio_rows = []
    for assignment in assignments:
        sample_id = assignment["parent_positive_id"]
        rng = random.Random(f"{assignment['sampling_seed']}:{assignment['negative_id']}")
        if assignment["branch"] == "prompt_side":
            prompt_rows.append(prompt_job(
                assignment, pairs[sample_id], rubrics[sample_id], assets, speech_pool, rng
            ))
        elif assignment["branch"] == "audio_side":
            audio_rows.append(audio_recipe(assignment, pairs[sample_id], rubrics[sample_id]))
        else:
            raise ValueError(f"invalid branch: {assignment['branch']}")

    prompt_rows.sort(key=lambda row: row["negative_id"])
    audio_rows.sort(key=lambda row: row["negative_id"])
    expected_prompt = sum(row["branch"] == "prompt_side" for row in assignments)
    expected_audio = sum(row["branch"] == "audio_side" for row in assignments)
    if len(prompt_rows) != expected_prompt or len(audio_rows) != expected_audio:
        raise ValueError(
            "materialization branch counts differ from plan: "
            f"{len(prompt_rows)}/{len(audio_rows)} vs {expected_prompt}/{expected_audio}"
        )

    prompt_path = args.output_dir / "prompt_jobs.jsonl"
    audio_path = args.output_dir / "audio_recipes.jsonl"
    write_jsonl(prompt_path, prompt_rows)
    write_jsonl(audio_path, audio_rows)
    summary = {
        "schema": "route2_negative_materialization_inputs",
        "inputs": {
            "plan": {"path": str(args.plan), "sha256": sha256(args.plan)},
            "pairs": {"path": str(args.pairs), "sha256": sha256(args.pairs)},
            "construction_labels": {"path": str(args.construction_labels), "sha256": sha256(args.construction_labels)},
            "assets": {"path": str(args.assets), "sha256": sha256(args.assets)},
            "speech_pool": {"path": str(args.speech_pool), "sha256": sha256(args.speech_pool)},
        },
        "prompt_jobs": len(prompt_rows),
        "audio_recipes": len(audio_rows),
        "prompt_categories": dict(Counter(row["path2_perturbation_category"] for row in prompt_rows)),
        "audio_categories": dict(Counter(row["perturbation_category"] for row in audio_rows)),
        "prompt_output": {"path": str(prompt_path), "sha256": sha256(prompt_path)},
        "audio_output": {"path": str(audio_path), "sha256": sha256(audio_path)},
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
