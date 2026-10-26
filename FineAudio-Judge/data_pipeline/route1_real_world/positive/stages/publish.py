#!/usr/bin/env python3
"""Publish one final Route 1 manifest with auditable speaker-patch provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset") or "").lower(), str(row["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    publication = config["publication"]

    speaker_enabled = bool((config.get("speaker_alignment") or {}).get("enabled", False))
    route_a_candidates, route_b_candidates = [], []
    for dataset in config["datasets"]:
        route_a_candidates.extend(read_jsonl(args.run_root / dataset / "caption_candidates.jsonl"))
        if speaker_enabled:
            route_b_candidates.extend(read_jsonl(args.run_root / dataset / "speaker_aligned_candidates.jsonl"))
    a_candidates = {key(row): row for row in route_a_candidates}
    b_candidates = {key(row): row for row in route_b_candidates}
    target = int(config["selection"]["target_per_dataset"])
    route_a = []
    for dataset in config["datasets"]:
        manifest_order = [
            (dataset, str(row["id"]))
            for row in read_jsonl(args.run_root / "data" / dataset / "selected_manifest.jsonl")
        ]
        seen = set(manifest_order)
        fallback = sorted(
            (value for value in a_candidates if value[0] == dataset and value not in seen),
            key=lambda value: value[1],
        )
        ordered = [value for value in manifest_order if value in a_candidates] + fallback
        route_a.extend(a_candidates[value] for value in ordered[:target])
    route_a.sort(key=key)
    published_keys = {key(row) for row in route_a}
    if bool(publication.get("publish_speaker_candidates_without_base", True)):
        route_b = [b_candidates[value] for value in sorted(b_candidates)]
    else:
        route_b = [b_candidates[value] for value in sorted(published_keys & set(b_candidates))]
    a_index, b_index = {key(row): row for row in route_a}, {key(row): row for row in route_b}
    final = []
    for value in sorted(a_index):
        base = a_index[value]
        speaker = b_index.get(value)
        applied = bool(speaker and speaker.get("speaker_edits"))
        selected = dict(speaker if applied else base)
        selected["final_caption_source"] = "speaker_aligned_patch" if applied else "base_audio_grounded"
        selected["speaker_alignment_applied"] = applied
        final.append(selected)
    write_jsonl(args.run_root / publication["final"], final)
    summary = {
        "protocol": config["protocol"],
        "target_per_dataset": target,
        "base_candidates": len(route_a_candidates),
        "speaker_candidates": len(route_b_candidates),
        "final": len(final),
        "final_with_speaker_patch": sum(bool(row.get("speaker_alignment_applied")) for row in final),
        "final_using_base_caption": sum(not bool(row.get("speaker_alignment_applied")) for row in final),
        "release_manifest": publication["final"],
        "per_dataset": {
            dataset: {
                "final": sum(str(row.get("dataset") or "").lower() == dataset for row in final),
                "with_speaker_patch": sum(
                    str(row.get("dataset") or "").lower() == dataset
                    and bool(row.get("speaker_alignment_applied"))
                    for row in final
                ),
            } for dataset in config["datasets"]
        },
    }
    summary["shortfalls"] = {
        dataset: max(0, target - summary["per_dataset"][dataset]["final"])
        for dataset in config["datasets"]
    }
    report = args.run_root / "reports" / "publish_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
