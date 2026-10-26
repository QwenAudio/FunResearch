#!/usr/bin/env python3
"""Plan one controlled Route 2 negative for every supplied positive pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORIES = ("event", "attribute", "relation", "all_three")
LEVEL_BY_CATEGORY = {
    "event": "low",
    # Attribute/relation edits intentionally preserve event-presence roots and
    # therefore cannot usually drive the all-rubric yes ratio into the mid
    # band.  Treat them as high-consistency localized negatives.
    "attribute": "high",
    "relation": "high",
    "all_three": "low",
}
# Select enough roots for the hierarchy-closed no set to approach the target
# of the category's declared consistency band.
LEVEL_TARGET_YES_RATIO = {"high": 0.80, "low": 0.26}
LEVEL_TARGET_YES_RANGE = {"high": [0.67, 1.0], "low": [0.0, 0.33]}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def groups(record: dict[str, Any]):
    result = {"event": [], "attribute": [], "relation": []}
    for question in record.get("questions") or []:
        subtype = question.get("subtype")
        if subtype == "exists":
            result["event"].append(question)
        elif subtype in {"speech_content", "speech_attribute"}:
            result["attribute"].append(question)
        elif subtype in {"before", "overlap"}:
            result["relation"].append(question)
    return result


def hierarchy_closure(selected: set[str], questions: list[dict[str, Any]]):
    affected = set(selected)
    changed = True
    while changed:
        changed = False
        for question in questions:
            rubric_id = question["rubric_id"]
            if rubric_id in affected:
                continue
            if any(parent in affected for parent in (question.get("parent_ids") or [])):
                affected.add(rubric_id)
                changed = True
    return affected


def choose_targets(record: dict[str, Any], category: str, level: str, rng: random.Random):
    questions = record["questions"]
    by_group = groups(record)
    required_groups = ("event", "attribute", "relation") if category == "all_three" else (category,)
    selected: set[str] = set()
    for group in required_groups:
        options = list(by_group[group])
        rng.shuffle(options)
        selected.add(options[0]["rubric_id"])

    allowed = [q for group in required_groups for q in by_group[group]]
    rng.shuffle(allowed)
    target = LEVEL_TARGET_YES_RATIO[level]

    def distance(ids: set[str]):
        affected = hierarchy_closure(ids, questions)
        yes_ratio = (len(questions) - len(affected)) / len(questions)
        return abs(yes_ratio - target), yes_ratio, affected

    current_distance, expected_yes, affected = distance(selected)
    # Add rubric roots while doing so improves the intended consistency target.
    # A single event edit may fan out through multiple dependent rubrics.
    while True:
        best = None
        for question in allowed:
            rubric_id = question["rubric_id"]
            if rubric_id in selected:
                continue
            trial = set(selected)
            trial.add(rubric_id)
            trial_distance, trial_yes, trial_affected = distance(trial)
            candidate = (trial_distance, rubric_id, trial_yes, trial_affected)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None or best[0] >= current_distance:
            break
        current_distance, rubric_id, expected_yes, affected = best
        selected.add(rubric_id)
    return sorted(selected), sorted(affected), round(expected_yes, 4)


def balanced_category_counts(total: int) -> dict[str, int]:
    """Scale the four perturbation categories without changing the 2K policy."""
    base, remainder = divmod(total, len(CATEGORIES))
    counts = {category: base for category in CATEGORIES}
    # Give remainders to the two unconstrained categories first.  This keeps a
    # tiny demo feasible when relatively few positives contain speech.
    for category in ("event", "relation", "attribute", "all_three")[:remainder]:
        counts[category] += 1
    return counts


def category_schedule(
    category: str, count: int, extra_prompt: bool, rng: random.Random
):
    prompt_count = count // 2 + int(count % 2 == 1 and extra_prompt)
    branches = ["prompt_side"] * prompt_count + ["audio_side"] * (count - prompt_count)
    rng.shuffle(branches)
    return [(branch, LEVEL_BY_CATEGORY[category]) for branch in branches]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    seed = args.seed
    rng = random.Random(seed)
    records = read_jsonl(args.construction_labels)
    by_id = {row["sample_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("construction labels contain duplicate sample ids")
    if len(by_id) < len(CATEGORIES):
        raise ValueError("at least four positives are required to cover all negative categories")
    category_counts = balanced_category_counts(len(by_id))
    available = set(by_id)
    assignments: dict[str, str] = {}

    # Allocate constrained classes first, without reusing a positive.
    for category in ("all_three", "attribute", "relation", "event"):
        eligible = []
        for sample_id in sorted(available):
            present = groups(by_id[sample_id])
            required = ("event", "attribute", "relation") if category == "all_three" else (category,)
            if all(present[group] for group in required):
                eligible.append(sample_id)
        rng.shuffle(eligible)
        wanted = category_counts[category]
        chosen = eligible[:wanted]
        if len(chosen) != wanted:
            raise ValueError(f"not enough positives for {category}: {len(chosen)}/{wanted}")
        for sample_id in chosen:
            assignments[sample_id] = category
            available.remove(sample_id)
    if available or len(assignments) != len(by_id):
        raise ValueError("category assignment did not consume each positive exactly once")

    plans = []
    odd_category_index = 0
    for category in CATEGORIES:
        sample_ids = [sample_id for sample_id, value in assignments.items() if value == category]
        rng.shuffle(sample_ids)
        is_odd = len(sample_ids) % 2 == 1
        schedule = category_schedule(
            category, len(sample_ids), extra_prompt=odd_category_index % 2 == 0, rng=rng
        )
        if is_odd:
            odd_category_index += 1
        for sample_id, (branch, level) in zip(sample_ids, schedule):
            record = by_id[sample_id]
            direct, affected, expected_yes = choose_targets(record, category, level, rng)
            retained = sorted({q["rubric_id"] for q in record["questions"]} - set(affected))
            plans.append({
                "schema": "route2_negative_assignment",
                "negative_id": f"{sample_id}::neg::{branch}::{category}::{level}",
                "parent_positive_id": sample_id,
                "branch": branch,
                "perturbation_category": category,
                "included_primitive_types": (
                    ["event", "attribute", "relation"] if category == "all_three" else [category]
                ),
                "intended_consistency_level": level,
                "intended_yes_ratio_target": LEVEL_TARGET_YES_RATIO[level],
                "construction_target_yes_range": LEVEL_TARGET_YES_RANGE[level],
                "final_consistency_level": None,
                "final_level_authority": "post_materialization_perceptual_or_human_rubric_yes_ratio",
                "direct_target_rubric_ids": direct,
                "hierarchy_affected_rubric_ids": affected,
                "retained_rubric_ids": retained,
                "direct_target_count": len(direct),
                "hierarchy_affected_count": len(affected),
                "rubric_count": len(record["questions"]),
                "planned_construction_yes_ratio": expected_yes,
                "materialization_status": "planned_not_rendered",
                "required_validation": [
                    "materialize_only_the_declared_primitive_edits",
                    "recompute_hierarchy_closed_rubric_labels",
                    "reject_undeclared_collateral_changes",
                    "verify_audio_signal_and_overlap_qc" if branch == "audio_side" else "verify_audio_hash_unchanged",
                    "verify_training_source_only_no_instructttseval",
                    "assign_final_level_from_observed_rubric_yes_ratio",
                ],
                "sampling_seed": seed,
            })

    plans.sort(key=lambda row: row["parent_positive_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in plans:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    summary = {
        "schema": "route2_negative_assignment_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.construction_labels), "source_sha256": sha256(args.construction_labels),
        "output": str(args.output), "output_sha256": sha256(args.output),
        "count": len(plans), "seed": seed,
        "category_counts": dict(Counter(row["perturbation_category"] for row in plans)),
        "negative_level_counts": dict(Counter(row["intended_consistency_level"] for row in plans)),
        "combined_positive_and_negative_level_counts": {
            "high": len(by_id),
            "mid": sum(row["intended_consistency_level"] == "mid" for row in plans),
            "low": sum(row["intended_consistency_level"] == "low" for row in plans),
        },
        "branch_counts": dict(Counter(row["branch"] for row in plans)),
        "category_level_counts": {
            category: dict(Counter(row["intended_consistency_level"] for row in plans if row["perturbation_category"] == category))
            for category in CATEGORIES
        },
        "planned_mean_yes_ratio_by_level": {
            level: round(
                sum(row["planned_construction_yes_ratio"] for row in plans if row["intended_consistency_level"] == level)
                / sum(row["intended_consistency_level"] == level for row in plans),
                4,
            )
            for level in ("mid", "low")
            if any(row["intended_consistency_level"] == level for row in plans)
        },
        "final_level_authority": "post_materialization_perceptual_or_human_rubric_yes_ratio",
        "scope": "assignment only; no caption/audio was changed",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
