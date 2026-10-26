#!/usr/bin/env python3
"""Shrink near-threshold foreground gaps without changing declared relations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from build_audio_negative_scenes import safe_id, truth


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extend_end(event: dict[str, Any], end: float) -> None:
    event["end"] = round(end, 4)
    event["duration"] = round(float(event["end"]) - float(event["start"]), 4)
    if event.get("replacement_source_role", event["role"]) != "speech":
        event["loop"] = True


def extend_start(event: dict[str, Any], start: float) -> None:
    event["start"] = round(start, 4)
    event["duration"] = round(float(event["end"]) - float(event["start"]), 4)
    if event.get("replacement_source_role", event["role"]) != "speech":
        event["loop"] = True


def repair(scene: dict[str, Any], target_gap: float) -> dict[str, Any]:
    scene = deepcopy(scene)
    foreground = sorted(
        [event for event in scene["events"] if event["role"] != "ambience"],
        key=lambda event: (float(event["start"]), float(event["end"]), event["event_id"]),
    )
    changes = []
    if foreground and float(foreground[0]["start"]) > target_gap:
        old = float(foreground[0]["start"])
        extend_start(foreground[0], target_gap)
        changes.append({"event_id": foreground[0]["event_id"], "field": "start", "old": old, "new": target_gap})
    cursor_event = foreground[0]
    cursor_end = float(cursor_event["end"])
    for event in foreground[1:]:
        start = float(event["start"])
        if start - cursor_end > target_gap:
            desired_end = start - target_gap
            if cursor_event.get("replacement_source_role", cursor_event["role"]) != "speech":
                old = float(cursor_event["end"])
                extend_end(cursor_event, desired_end)
                changes.append({"event_id": cursor_event["event_id"], "field": "end", "old": old, "new": desired_end})
                cursor_end = desired_end
            elif event.get("replacement_source_role", event["role"]) != "speech":
                desired_start = cursor_end + target_gap
                old = float(event["start"])
                extend_start(event, desired_start)
                changes.append({"event_id": event["event_id"], "field": "start", "old": old, "new": desired_start})
                start = desired_start
        if float(event["end"]) > cursor_end:
            cursor_event = event
            cursor_end = float(event["end"])
    duration = float(scene["duration"])
    if foreground and duration - cursor_end > target_gap:
        desired_end = duration - target_gap
        if cursor_event.get("replacement_source_role", cursor_event["role"]) != "speech":
            old = float(cursor_event["end"])
            extend_end(cursor_event, desired_end)
            changes.append({"event_id": cursor_event["event_id"], "field": "end", "old": old, "new": desired_end})
    scene.setdefault("negative_construction", {})["gap_repair"] = {
        "method": "extend_non_speech_boundaries_to_target_gap",
        "target_gap_seconds": target_gap,
        "changes": changes,
    }
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-gap", type=float, default=2.1)
    args = parser.parse_args()
    failed = {row["id"] for row in read_jsonl(args.validation) if not row.get("valid")}
    recipes = {safe_id(row["negative_id"]): row for row in read_jsonl(args.recipes)}
    selected = [row for row in read_jsonl(args.scenes) if row["id"] in failed]
    repaired = []
    for source in selected:
        scene = repair(source, args.target_gap)
        recipe = recipes[scene["id"]]
        intervals = {
            event["event_id"]: (float(event["start"]), float(event["end"]))
            for event in scene["events"]
        }
        realized_false = sorted(
            question["rubric_id"] for question in recipe["original_questions"]
            if question["subtype"] in {"before", "overlap"}
            and not truth(
                question["subtype"],
                intervals[question["construction_evidence"]["source_event_id"]],
                intervals[question["construction_evidence"]["target_event_id"]],
            )
        )
        expected_false = sorted(
            (scene.get("negative_construction") or {}).get("relation_audit", {}).get(
                "changed_relation_rubric_ids", []
            )
        )
        if realized_false != expected_false:
            raise ValueError(
                f"gap repair changed relations for {scene['id']}: "
                f"expected={expected_false} realized={realized_false}"
            )
        repaired.append(scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in repaired:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"failed": len(failed), "repaired_scenes": len(repaired), "target_gap": args.target_gap}))


if __name__ == "__main__":
    main()
