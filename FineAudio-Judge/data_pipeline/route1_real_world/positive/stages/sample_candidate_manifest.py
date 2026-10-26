"""Take a deterministic source-diverse sample from a discovered manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import read_jsonl, write_jsonl


def stable_key(row: dict, seed: int) -> str:
    identity = "|".join(str(row.get(key, "")) for key in (
        "original_video_filename", "clip_id", "parquet", "row_index",
        "start_seconds", "end_seconds",
    ))
    return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    selected = sorted(rows, key=lambda row: stable_key(row, args.seed))[:args.limit]
    write_jsonl(args.output, selected)
    print(json.dumps({
        "input": len(rows), "selected": len(selected), "seed": args.seed,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
