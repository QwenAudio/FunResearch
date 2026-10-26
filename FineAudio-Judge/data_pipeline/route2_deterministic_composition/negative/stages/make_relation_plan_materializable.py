#!/usr/bin/env python3
"""Make Route 2 relation assignments audio-materializable before generation.

For every audio-side relation edit, enumerate complete foreground time-slot
permutations. The chosen permutation determines the relation rubrics that will
actually flip; all of them are recorded as direct targets before rendering.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import permutations
import json
from pathlib import Path
from build_audio_negative_scenes import truth
from plan_negatives import hierarchy_closure


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_layout(
    pair: dict, rubric: dict, base_direct: set[str], target_yes: float,
    target_range: list[float],
):
    questions = rubric["questions"]
    relation_questions = [q for q in questions if q["subtype"] in {"before", "overlap"}]
    original = {e["event_id"]: (float(e["start"]), float(e["end"])) for e in pair["events"]}
    foreground = sorted(
        [e for e in pair["events"] if e["role"] != "ambience"],
        key=lambda e: (e["start"], e["end"], e["event_id"]),
    )
    ids = [e["event_id"] for e in foreground]
    slots = [(float(e["start"]), float(e["end"])) for e in foreground]
    by_event = {e["event_id"]: e for e in foreground}
    best = None
    for slot_order in permutations(range(len(slots))):
        if all(index == value for index, value in enumerate(slot_order)):
            continue
        intervals = dict(original)
        feasible = True
        for event_id, slot_index in zip(ids, slot_order):
            slot_duration = slots[slot_index][1] - slots[slot_index][0]
            event = by_event[event_id]
            if event["role"] == "speech" and slot_duration > float(event.get("source_duration", event["duration"])) + 1e-3:
                feasible = False
                break
            intervals[event_id] = slots[slot_index]
        if not feasible:
            continue
        changed = set()
        for question in relation_questions:
            evidence = question["construction_evidence"]
            if not truth(
                question["subtype"],
                intervals[evidence["source_event_id"]],
                intervals[evidence["target_event_id"]],
            ):
                changed.add(question["rubric_id"])
        if not changed:
            continue
        direct = set(base_direct) | changed
        affected = hierarchy_closure(direct, questions)
        yes_ratio = (len(questions) - len(affected)) / len(questions)
        moved = sum(index != value for index, value in enumerate(slot_order))
        displacement = sum(abs(slots[value][0] - slots[index][0]) for index, value in enumerate(slot_order))
        in_band = target_range[0] <= yes_ratio <= target_range[1]
        score = (not in_band, abs(yes_ratio - target_yes), moved, displacement, slot_order)
        if best is None or score < best[0]:
            best = (score, direct, affected, changed, yes_ratio, slot_order)
    if best is None:
        raise RuntimeError(f"no nontrivial relation layout: {pair['id']}")
    return {
        "direct": sorted(best[1]),
        "affected": sorted(best[2]),
        "changed_relations": sorted(best[3]),
        "yes_ratio": round(best[4], 4),
        "slot_assignment": {event_id: slot_index for event_id, slot_index in zip(ids, best[5])},
        "foreground_event_order": ids,
        "score": [best[0][0], round(best[0][1], 6), best[0][2], round(best[0][3], 4)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--positive-pairs", type=Path, required=True)
    parser.add_argument("--construction-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plans = read_jsonl(args.plan)
    pairs = {row["id"]: row for row in read_jsonl(args.positive_pairs)}
    rubrics = {row["sample_id"]: row for row in read_jsonl(args.construction_labels)}
    output = []
    for index, source in enumerate(plans, 1):
        row = dict(source)
        row.pop("schema_version", None)
        row["schema"] = "route2_materializable_negative_assignment"
        row["source_plan"] = str(args.plan)
        row["materialization_status"] = "planned_not_rendered"
        if row["branch"] == "audio_side" and row["perturbation_category"] in {"attribute", "all_three"}:
            sample_id = row["parent_positive_id"]
            questions = rubrics[sample_id]["questions"]
            by_id = {question["rubric_id"]: question for question in questions}
            direct = set(row["direct_target_rubric_ids"])
            # Replacing a speech waveform changes its transcript and complete
            # selected-caption attribute bundle together.  Expand the plan
            # before rendering so no physically changed fact is mislabeled as
            # retained merely to hit a numerical target.
            speech_event_ids = {
                by_id[rubric_id]["construction_evidence"]["event_id"]
                for rubric_id in direct
                if by_id[rubric_id]["subtype"] in {"speech_content", "speech_attribute"}
            }
            direct.update(
                question["rubric_id"] for question in questions
                if question["subtype"] in {"speech_content", "speech_attribute"}
                and question["construction_evidence"]["event_id"] in speech_event_ids
            )
            affected = hierarchy_closure(direct, questions)
            all_ids = {question["rubric_id"] for question in questions}
            row["direct_target_rubric_ids"] = sorted(direct)
            row["hierarchy_affected_rubric_ids"] = sorted(affected)
            row["retained_rubric_ids"] = sorted(all_ids - affected)
            row["direct_target_count"] = len(direct)
            row["hierarchy_affected_count"] = len(affected)
            row["planned_construction_yes_ratio"] = round(
                (len(questions) - len(affected)) / len(questions), 4
            )
            row["audio_attribute_materialization"] = {
                "method": "whole_speech_fact_bundle_replacement",
                "expanded_speech_event_ids": sorted(speech_event_ids),
            }
        if row["branch"] == "audio_side" and row["perturbation_category"] in {"relation", "all_three"}:
            sample_id = row["parent_positive_id"]
            by_id = {q["rubric_id"]: q for q in rubrics[sample_id]["questions"]}
            base_direct = {
                rubric_id for rubric_id in row["direct_target_rubric_ids"]
                if by_id[rubric_id]["subtype"] not in {"before", "overlap"}
            }
            layout = choose_layout(
                pairs[sample_id], rubrics[sample_id], base_direct,
                float(row["intended_yes_ratio_target"]),
                [float(value) for value in row["construction_target_yes_range"]],
            )
            row["direct_target_rubric_ids"] = layout["direct"]
            row["hierarchy_affected_rubric_ids"] = layout["affected"]
            all_ids = {q["rubric_id"] for q in rubrics[sample_id]["questions"]}
            row["retained_rubric_ids"] = sorted(all_ids - set(layout["affected"]))
            row["direct_target_count"] = len(layout["direct"])
            row["hierarchy_affected_count"] = len(layout["affected"])
            row["planned_construction_yes_ratio"] = layout["yes_ratio"]
            row["audio_relation_materialization"] = {
                "method": "complete_foreground_time_slot_permutation",
                "changed_relation_rubric_ids": layout["changed_relations"],
                "slot_assignment": layout["slot_assignment"],
                "foreground_event_order": layout["foreground_event_order"],
                "selection_score": layout["score"],
            }
        output.append(row)
        if index % 100 == 0:
            print(json.dumps({"planned": index, "total": len(plans)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    summary = {
        "schema": "route2_materializable_negative_assignment_summary",
        "source": str(args.plan),
        "source_sha256": sha256(args.plan),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "count": len(output),
        "branch_counts": dict(Counter(row["branch"] for row in output)),
        "category_counts": dict(Counter(row["perturbation_category"] for row in output)),
        "level_counts": dict(Counter(row["intended_consistency_level"] for row in output)),
        "audio_relation_plans": sum("audio_relation_materialization" in row for row in output),
        "audio_attribute_plans": sum("audio_attribute_materialization" in row for row in output),
        "audio_relation_in_target_band": sum(
            "audio_relation_materialization" in row
            and row["construction_target_yes_range"][0] <= row["planned_construction_yes_ratio"] <= row["construction_target_yes_range"][1]
            for row in output
        ),
        "scope": "assignment plus deterministic audio relation slot plans; no caption/audio changed",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
