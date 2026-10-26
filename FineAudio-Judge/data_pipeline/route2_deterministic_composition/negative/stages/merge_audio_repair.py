#!/usr/bin/env python3
"""Merge validated repair rows into the canonical Route 2 audio mix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


POSITIVE_STAGES = Path(__file__).resolve().parents[2] / "positive" / "stages"
sys.path.insert(0, str(POSITIVE_STAGES))
from compose_scene_graphs import validate_rows  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    args = parser.parse_args()
    canonical = {row["id"]: row for row in read_jsonl(args.canonical / "pairs.jsonl")}
    repair_validation = {
        row["id"]: row for row in read_jsonl(args.repair / "validation_report.jsonl")
    }
    replacements = 0
    for row in read_jsonl(args.repair / "pairs.jsonl"):
        if repair_validation.get(row["id"], {}).get("valid"):
            row["repair_source_dir"] = str(args.repair)
            canonical[row["id"]] = row
            replacements += 1
    rows = sorted(canonical.values(), key=lambda row: row["id"])
    report, summary = validate_rows(rows)
    write_jsonl(args.canonical / "pairs.partial.jsonl", rows)
    write_jsonl(args.canonical / "pairs.jsonl", rows)
    write_jsonl(args.canonical / "validation_report.jsonl", report)
    (args.canonical / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"merged_valid_repairs": replacements, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
