from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from common import write_jsonl


TIME_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}(?:\.\d+)?)$"
)


def parse_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = TIME_PATTERN.match(str(value).strip())
    if not match:
        return None
    return (
        int(match.group("hours") or 0) * 3600
        + int(match.group("minutes")) * 60
        + float(match.group("seconds"))
    )


def scene_interval(scene: dict[str, Any]) -> tuple[float, float] | None:
    timestamps = scene.get("timestamps") or scene.get("timestamp") or {}
    start = parse_time(timestamps.get("start_timestamp") or timestamps.get("start"))
    end = parse_time(timestamps.get("end_timestamp") or timestamps.get("end"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def discover_in_row(
    parquet_path: Path,
    row_index: int,
    meta: dict[str, Any],
    min_duration: float,
    max_duration: float,
    max_per_video: int,
) -> list[dict[str, Any]]:
    video_duration = float(meta.get("duration_seconds") or 0)
    base = {
        "parquet": str(parquet_path),
        "row_index": row_index,
        "original_video_filename": meta.get("original_video_filename"),
        "youtube_title": meta.get("youtube_title"),
        "content_parent_category": meta.get("content_parent_category"),
        "content_fine_category": meta.get("content_fine_category"),
    }
    if min_duration <= video_duration <= max_duration:
        return [{
            **base,
            "start_seconds": 0.0,
            "end_seconds": video_duration,
            "selection_basis": "native_full_video",
        }]
    scenes = ((meta.get("content_metadata") or {}).get("scenes") or [])
    candidates: list[dict[str, Any]] = []
    for scene in scenes:
        interval = scene_interval(scene)
        if interval is None:
            continue
        start, end = interval
        duration = end - start
        if min_duration <= duration <= max_duration and end <= video_duration + 1:
            candidates.append({
                **base,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "selection_basis": "single_semantic_scene",
                "scene_id": scene.get("sceneId"),
                "scene_title": scene.get("title"),
                "scene_context": scene.get("contextualRelevance"),
            })
    # Prefer longer single scenes because they usually contain more temporal development.
    candidates.sort(
        key=lambda row: (
            float(row["end_seconds"]) - float(row["start_seconds"]),
            -float(row["start_seconds"]),
        ),
        reverse=True,
    )
    return candidates[:max_per_video]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the FineVideo parquet shards.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--max-per-video", type=int, default=1)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int)
    args = parser.parse_args()
    if args.max_per_video < 1:
        raise ValueError("--max-per-video must be >= 1")
    parquet_paths = sorted(args.data_dir.glob("*.parquet"))
    parquet_paths = parquet_paths[args.start_shard : args.end_shard]
    candidates: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        rows = pq.read_table(parquet_path, columns=["json"]).to_pylist()
        for row_index, row in enumerate(rows):
            meta = row["json"]
            candidates.extend(discover_in_row(
                parquet_path,
                row_index,
                meta,
                args.min_duration,
                args.max_duration,
                args.max_per_video,
            ))
            if args.max_candidates and len(candidates) >= args.max_candidates:
                candidates = candidates[: args.max_candidates]
                write_jsonl(args.output, candidates)
                print(json.dumps({"candidates": len(candidates)}, ensure_ascii=False))
                return
    write_jsonl(args.output, candidates)
    print(json.dumps({
        "parquet_shards": len(parquet_paths),
        "candidates": len(candidates),
        "full_native": sum(row["selection_basis"] == "native_full_video" for row in candidates),
        "single_scene_crop": sum(row["selection_basis"] == "single_semantic_scene" for row in candidates),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
