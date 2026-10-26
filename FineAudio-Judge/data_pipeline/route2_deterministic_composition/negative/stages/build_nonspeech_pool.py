#!/usr/bin/env python3
"""Select AudioSet-Strong sources unused by every Route 2 positive."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


POSITIVE_STAGES = Path(__file__).resolve().parents[2] / "positive" / "stages"
sys.path.insert(0, str(POSITIVE_STAGES))
from build_asset_pool import balanced_select, collect_candidates  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-pairs", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effect-count", type=int, default=1400)
    parser.add_argument("--ambience-count", type=int, default=250)
    parser.add_argument("--music-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    positive = read_jsonl(args.positive_pairs)
    used_ids = {
        str(event["source_id"])
        for pair in positive for event in pair["events"]
        if event["role"] != "speech"
    }
    pools = collect_candidates(sorted(args.shards.glob("*.parquet")), args.seed)
    for slug in list(pools):
        pools[slug] = [row for row in pools[slug] if str(row["video_id"]) not in used_ids]
    selected = balanced_select(
        pools, args.effect_count, args.ambience_count, args.music_count
    )
    if used_ids & {str(row["video_id"]) for row in selected}:
        raise RuntimeError("positive AudioSet source leaked into negative selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    summary = {
        "selected": len(selected),
        "roles": dict(Counter(row["role"] for row in selected)),
        "unique_video_ids": len({row["video_id"] for row in selected}),
        "positive_video_id_overlap": 0,
        "source_dataset": "AudioSet-Strong",
        "instructttseval_rows": 0,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
