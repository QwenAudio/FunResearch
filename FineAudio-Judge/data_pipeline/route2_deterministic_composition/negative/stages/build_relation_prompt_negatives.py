#!/usr/bin/env python3
"""Write exact deterministic P' seeds for relation-only Route 2 jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_phrase(event: dict[str, Any], language: str) -> str:
    label = str(event["caption_label_zh"] if language == "zh" else event["caption_label_en"])
    if event["role"] == "speech":
        description = str(event.get("speaker_description") or label)
        transcript = str(event.get("text") or "")
        return (
            f"{description}，内容为“{transcript}”"
            if language == "zh"
            else f'{description} says, "{transcript}"'
        )
    return f"{label}响起" if language == "zh" else f"{label} is heard"


def reverse_caption(pair: dict[str, Any], language: str) -> str:
    # Every original relation rubric is either a non-overlapping before edge or
    # an overlap edge.  Strictly reversing the complete event order and rendering
    # it as a non-overlapping sequence makes both kinds false while preserving
    # every event identity and every speech/attribute fact.
    events = sorted(
        pair["events"], key=lambda event: (float(event["start"]), float(event["end"]), event["event_id"]),
        reverse=True,
    )
    phrases = [event_phrase(event, language) for event in events]
    if language == "zh":
        connectors = ["先是", "随后", "接着", "之后", "再然后", "再往后", "临近结束时", "最后"]
        parts = [connectors[min(index, len(connectors) - 1)] + phrase for index, phrase in enumerate(phrases)]
        return "，".join(parts) + "。"
    connectors = ["First, ", "Next, ", "Then, ", "After that, ", "Later, ", "Subsequently, ", "Near the end, ", "Finally, "]
    return " ".join(
        connectors[min(index, len(connectors) - 1)] + phrase + "."
        for index, phrase in enumerate(phrases)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = [row for row in read_jsonl(args.jobs) if row["path2_perturbation_category"] == "relation"]
    pairs = {row["id"]: row for row in read_jsonl(args.pairs)}
    rows = []
    for job in jobs:
        prompt_prime = reverse_caption(pairs[job["parent_positive_id"]], job["language"])
        if prompt_prime == job["prompt"]:
            raise ValueError(f"deterministic reversal did not change prompt: {job['sample_id']}")
        missing = [
            anchor for anchor in job["must_preserve_event_anchors"]
            if anchor.casefold() not in prompt_prime.casefold()
        ]
        if missing:
            raise ValueError(f"lost anchors for {job['sample_id']}: {missing}")
        patch = {
            "original_span": job["prompt"], "replacement_span": prompt_prime,
            "changed_rubric_ids": job["selected_rubric_ids"],
            "start": 0, "end": len(job["prompt"]), "patch_order": 0,
            "new_start": 0, "new_end": len(prompt_prime),
        }
        rows.append({
            "schema": "route2_prompt_negative_generation",
            "sample_id": job["sample_id"], "parent_positive_id": job["parent_positive_id"],
            "status": "ok", "prompt": job["prompt"], "prompt_prime": prompt_prime,
            "patches": [patch], "attempts": 0, "endpoint": None,
            "model": "deterministic_relation_reversal",
            "disturbance_category": "relation", "selected_rubric_ids": job["selected_rubric_ids"],
            "outside_patch_equality": True,
            "note": "All events reversed into an explicitly non-overlapping sequence; all selected original relation facts become false.",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"relation_jobs": len(jobs), "written": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
