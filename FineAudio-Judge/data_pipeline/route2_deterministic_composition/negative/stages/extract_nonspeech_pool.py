#!/usr/bin/env python3
"""Extract a selected negative asset pool one parquet shard at a time."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

POSITIVE_STAGES = Path(__file__).resolve().parents[2] / "positive" / "stages"
sys.path.insert(0, str(POSITIVE_STAGES))
from build_asset_pool import extract  # noqa: E402


def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def worker(items_path: Path, output_dir: Path, part_path: Path) -> None:
    write(part_path, extract(read(items_path), output_dir))


def parent(selection: Path, output_dir: Path, manifest: Path, parts_dir: Path) -> None:
    groups = defaultdict(list)
    for row in read(selection):
        groups[row["parquet_path"]].append(row)
    parts_dir.mkdir(parents=True, exist_ok=True)
    for index, (parquet_path, rows) in enumerate(sorted(groups.items())):
        part = parts_dir / f"{index:04d}.jsonl"
        if part.exists() and len(read(part)) == len(rows):
            continue
        items = parts_dir / f"{index:04d}.selection.jsonl"
        write(items, rows)
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--worker-items", str(items), "--output-dir", str(output_dir),
            "--part", str(part),
        ], check=True)
        print(json.dumps({
            "part": index + 1, "parts": len(groups), "rows": len(rows),
            "parquet": parquet_path,
        }))
    merged = []
    for part in sorted(parts_dir.glob("*.jsonl")):
        if ".selection." not in part.name:
            merged.extend(read(part))
    write(manifest, merged)
    print(json.dumps({"extracted": len(merged), "manifest": str(manifest)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--parts-dir", type=Path)
    parser.add_argument("--worker-items", type=Path)
    parser.add_argument("--part", type=Path)
    args = parser.parse_args()
    if args.worker_items:
        if args.part is None:
            parser.error("--part is required with --worker-items")
        worker(args.worker_items, args.output_dir, args.part)
    else:
        for name in ("selection", "manifest", "parts_dir"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required in parent mode")
        parent(args.selection, args.output_dir, args.manifest, args.parts_dir)


if __name__ == "__main__":
    main()
