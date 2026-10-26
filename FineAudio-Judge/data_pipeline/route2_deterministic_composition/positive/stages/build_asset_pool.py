#!/usr/bin/env python3
"""Build a large, low-reuse Path2 source pool from local AudioSet-Strong shards.

Only strongly annotated intervals with a sufficiently long target event are
kept.  Intervals overlapping speech are rejected; non-music roles also reject
music overlap.  Each source video is selected at most once so downstream
mixes can use distinct acoustic recordings rather than repeatedly reusing the
small review-demo pool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import random
import gc
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from asset_taxonomy import TARGETS

# event_name: (slug, Chinese label, role)
LEGACY_TARGETS: dict[str, tuple[str, str, str]] = {
    "Alarm": ("alarm", "警报声", "sound_effect"),
    "Applause": ("applause", "掌声", "sound_effect"),
    "Clapping": ("clapping", "拍手声", "sound_effect"),
    "Bell": ("bell", "钟声", "sound_effect"),
    "Church bell": ("church_bell", "教堂钟声", "sound_effect"),
    "Computer keyboard": ("computer_keyboard", "键盘敲击声", "sound_effect"),
    "Dishes, pots, and pans": ("dishes", "碗碟碰撞声", "sound_effect"),
    "Engine": ("engine", "发动机声", "sound_effect"),
    "Medium engine (mid frequency)": ("medium_engine", "中频发动机声", "sound_effect"),
    "Fireworks": ("fireworks", "烟花爆响", "sound_effect"),
    "Frying (food)": ("frying", "食物煎炸声", "sound_effect"),
    "Helicopter": ("helicopter", "直升机声", "sound_effect"),
    "Laughter": ("laughter", "笑声", "sound_effect"),
    "Mechanical fan": ("mechanical_fan", "风扇运转声", "sound_effect"),
    "Motorcycle": ("motorcycle", "摩托车声", "sound_effect"),
    "Pour": ("pour", "倾倒液体声", "sound_effect"),
    "Printer": ("printer", "打印机声", "sound_effect"),
    "Siren": ("siren", "警笛声", "sound_effect"),
    "Splash, splatter": ("splash", "水花飞溅声", "sound_effect"),
    "Train": ("train", "列车声", "sound_effect"),
    "Vacuum cleaner": ("vacuum_cleaner", "吸尘器声", "sound_effect"),
    "Vehicle horn, car horn, honking": ("vehicle_horn", "汽车鸣笛声", "sound_effect"),
    "Walk, footsteps": ("footsteps", "脚步声", "sound_effect"),
    "Water tap, faucet": ("water_tap", "水龙头流水声", "sound_effect"),
    "Bird vocalization, bird call, bird song": ("birdsong", "鸟鸣声", "sound_effect"),
    "Cricket": ("cricket", "蟋蟀鸣叫声", "sound_effect"),
    "Thunder": ("thunder", "雷声", "sound_effect"),
    "Rain on surface": ("rain", "雨水敲击表面声", "ambience"),
    "Traffic noise, roadway noise": ("traffic", "道路交通底噪", "ambience"),
    "Waterfall": ("waterfall", "瀑布声", "ambience"),
    "Waves, surf": ("waves", "连续海浪声", "ambience"),
    "Music": ("music", "背景音乐", "music"),
}

SPEECH_MARKERS = (
    "speech", "speaking", "conversation", "narration", "whisper",
    "shout", "scream", "babbling", "hubbub",
)
MUSIC_MARKERS = ("music", "singing", "choir", "song", "musical")


def excluded_source_ids(paths: list[Path]) -> set[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("source_id", "video_id", "uid", "original_source_id", "replacement_source_id"):
                if value.get(key) is not None:
                    values.add(str(value[key]))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                visit(json.loads(line))
    return values


def overlaps(left: dict[str, Any], right: dict[str, Any]) -> float:
    return min(float(left["end"]), float(right["end"])) - max(
        float(left["start"]), float(right["start"])
    )


def forbidden_overlap(target: dict[str, Any], events: list[dict[str, Any]], role: str) -> bool:
    for event in events:
        if event is target or overlaps(target, event) <= 0.15:
            continue
        name = str(event.get("event_name", "")).lower()
        if any(marker in name for marker in SPEECH_MARKERS):
            return True
        if role != "music" and any(marker in name for marker in MUSIC_MARKERS):
            return True
    return False


def collect_candidates(shards: list[Path], seed: int) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parquet_path in shards:
        parquet = pq.ParquetFile(parquet_path)
        try:
            table = parquet.read(columns=["video_id", "events"])
            for row_index, row in enumerate(table.to_pylist()):
                events = row.get("events") or []
                for event_index, event in enumerate(events):
                    name = str(event.get("event_name", ""))
                    spec = TARGETS.get(name)
                    if not spec:
                        continue
                    slug, label, role = spec
                    duration = float(event["end"]) - float(event["start"])
                    minimum = 7.5 if role in {"ambience", "music"} else 4.2
                    if duration < minimum or forbidden_overlap(event, events, role):
                        continue
                    pools[slug].append({
                        "parquet_path": str(parquet_path),
                        "row_index": row_index,
                        "event_index": event_index,
                        "video_id": row["video_id"],
                        "event_name": name,
                        "slug": slug,
                        "label": label,
                        "role": role,
                        "event_start": round(float(event["start"]), 4),
                        "event_end": round(float(event["end"]), 4),
                        "event_duration": round(duration, 4),
                    })
        finally:
            parquet.close()
        del table, parquet
        gc.collect()
    rng = random.Random(seed)
    for values in pools.values():
        rng.shuffle(values)
    return pools


def balanced_select(
    pools: dict[str, list[dict[str, Any]]], effect_count: int,
    ambience_count: int, music_count: int,
) -> list[dict[str, Any]]:
    quotas = {"sound_effect": effect_count, "ambience": ambience_count, "music": music_count}
    selected: list[dict[str, Any]] = []
    used_video_ids: set[str] = set()
    positions = Counter()
    role_slugs = {
        role: sorted(slug for slug, values in pools.items() if values and values[0]["role"] == role)
        for role in quotas
    }
    for role, quota in quotas.items():
        slugs = role_slugs[role]
        if not slugs:
            raise RuntimeError(f"No candidates for role={role}")
        exhausted: set[str] = set()
        while sum(item["role"] == role for item in selected) < quota:
            progress = False
            for slug in slugs:
                if slug in exhausted:
                    continue
                values = pools[slug]
                while positions[slug] < len(values):
                    item = values[positions[slug]]
                    positions[slug] += 1
                    if item["video_id"] in used_video_ids:
                        continue
                    selected.append(item)
                    used_video_ids.add(item["video_id"])
                    progress = True
                    break
                else:
                    exhausted.add(slug)
                if sum(item["role"] == role for item in selected) >= quota:
                    break
            if not progress:
                raise RuntimeError(
                    f"Insufficient unique sources for role={role}: "
                    f"wanted={quota}, got={sum(item['role'] == role for item in selected)}"
                )
    return selected


def mono_float(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=True, dtype="float32")
    return audio.mean(axis=1), int(sample_rate)


def extract(selected: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_parquet: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    result: list[dict[str, Any]] = []
    for item in selected:
        filename = f"{item['role']}__{item['slug']}__{item['video_id']}.flac"
        path = output_dir / filename
        if path.exists():
            with sf.SoundFile(path) as handle:
                duration = len(handle) / int(handle.samplerate)
                sample_rate = int(handle.samplerate)
            result.append({
                **item,
                "source_path": str(path),
                "duration": round(duration, 4),
                "sample_rate": sample_rate,
                "source_dataset": "AudioSet-Strong",
                "atomic_count_verified": False,
            })
        else:
            by_parquet[item["parquet_path"]][item["row_index"]].append(item)

    for parquet_name, row_map in sorted(by_parquet.items()):
        wanted = set(row_map)
        offset = 0
        parquet = pq.ParquetFile(parquet_name)
        try:
            for batch in parquet.iter_batches(batch_size=16, columns=["audio"]):
                batch_rows = batch.to_pylist()
                for local_index, row in enumerate(batch_rows):
                    row_index = offset + local_index
                    if row_index not in wanted:
                        continue
                    audio, sample_rate = mono_float(row["audio"]["bytes"])
                    for item in row_map[row_index]:
                        filename = (
                            f"{item['role']}__{item['slug']}__{item['video_id']}.flac"
                        )
                        path = output_dir / filename
                        start = max(0, round(item["event_start"] * sample_rate))
                        end = min(len(audio), round(item["event_end"] * sample_rate))
                        segment = audio[start:end]
                        if len(segment) < round(4.0 * sample_rate):
                            continue
                        temporary = path.with_suffix(".part.flac")
                        for attempt in range(1, 7):
                            try:
                                sf.write(
                                    temporary, segment, sample_rate,
                                    format="FLAC", subtype="PCM_16",
                                )
                                temporary.replace(path)
                                break
                            except Exception:
                                if temporary.exists():
                                    temporary.unlink()
                                if attempt == 6:
                                    raise
                                # CPFS occasionally returns a transient metadata
                                # write error when thousands of clips are created.
                                time.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
                        result.append({
                            **item,
                            "source_path": str(path),
                            "duration": round(len(segment) / sample_rate, 4),
                            "sample_rate": sample_rate,
                            "source_dataset": "AudioSet-Strong",
                            "atomic_count_verified": False,
                        })
                offset += len(batch_rows)
                del batch_rows, batch
        finally:
            parquet.close()
        gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        type=Path,
        required=True,
        help="Directory containing AudioSet-Strong parquet shards with embedded audio.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--effect-count", type=int, default=5000)
    parser.add_argument("--ambience-count", type=int, default=800)
    parser.add_argument("--music-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--exclude-source-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    shards = sorted(args.shards.glob("*.parquet"))
    pools = collect_candidates(shards, args.seed)
    excluded = excluded_source_ids(args.exclude_source_manifest)
    excluded_candidates = 0
    for slug, values in pools.items():
        retained = [row for row in values if str(row["video_id"]) not in excluded]
        excluded_candidates += len(values) - len(retained)
        pools[slug] = retained
    selected = balanced_select(
        pools, args.effect_count, args.ambience_count, args.music_count
    )
    selection_manifest = args.selection_manifest or args.manifest.with_name(
        args.manifest.stem + "_selection.jsonl"
    )
    selection_manifest.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    if args.selection_only:
        print(json.dumps({
            "selected": len(selected),
            "roles": Counter(row["role"] for row in selected),
            "excluded_source_ids": len(excluded),
            "excluded_candidates": excluded_candidates,
            "selection_manifest": str(selection_manifest),
        }, ensure_ascii=False, default=dict))
        return
    rows = extract(selected, args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": len(selected),
        "extracted": len(rows),
        "roles": Counter(row["role"] for row in rows),
        "labels": Counter(row["slug"] for row in rows),
        "unique_video_ids": len({row["video_id"] for row in rows}),
        "excluded_source_ids": len(excluded),
        "excluded_candidates": excluded_candidates,
        "manifest": str(args.manifest),
    }, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
