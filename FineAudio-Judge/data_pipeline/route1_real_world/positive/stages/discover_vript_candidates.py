#!/usr/bin/env python3
"""Discover native duration-bounded Vript scenes before audio-model screening.

This stage never concatenates scenes. It reads duration and ASR sidecars from
the downloaded Vript ZIP archives, keeps Chinese/English clips without lexical
profanity, then uses only coarse metadata heuristics to prioritize an
audio-screening pool. The audio model remains the authority for complexity.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common import write_jsonl
from content_safety import contains_profanity


POSITIVE_TERMS = {
    "action", "battle", "fight", "fighting", "war", "chase", "crash",
    "explosion", "gun", "shoot", "horror", "thriller", "trailer", "movie",
    "film", "drama", "dramatic", "crowd", "concert", "dance", "sports",
    "skate", "race", "kitchen", "cook", "machine", "factory", "storm",
    "fire", "rescue", "attack", "argument", "multiple people", "group",
    "scream", "shout", "vehicle", "motorcycle", "train", "aircraft",
    "动画", "电影", "战斗", "打斗", "追逐", "爆炸", "枪", "人群", "演唱会",
    "舞蹈", "比赛", "厨房", "机器", "暴雨", "火灾", "争吵", "尖叫",
}
NEGATIVE_TERMS = {
    "tutorial", "how to", "lecture", "podcast", "interview", "news",
    "review", "unboxing", "talking head", "commentary", "sermon", "speech",
    "教程", "访谈", "播客", "新闻", "演讲", "讲座", "开箱", "评测",
}


def normalize_language(value: Any) -> str | None:
    language = str(value or "").strip().lower().replace("_", "-")
    if language == "en" or language.startswith("en-"):
        return "en"
    if language in {"zh", "cmn", "yue"} or language.startswith(("zh-", "cmn-", "yue-")):
        return "zh"
    return None


def text_from_asr(row: dict[str, Any]) -> str:
    if row.get("full_text"):
        return str(row["full_text"])
    return " ".join(
        str(segment.get("text", ""))
        for segment in row.get("scene_text") or []
        if isinstance(segment, dict)
    )


def speech_statistics(asr: dict[str, Any], duration: float) -> dict[str, float | int]:
    spans: list[tuple[float, float]] = []
    for segment in asr.get("scene_text") or []:
        span = segment.get("span") if isinstance(segment, dict) else None
        if (
            isinstance(span, list)
            and len(span) == 2
            and isinstance(span[0], (int, float))
            and isinstance(span[1], (int, float))
        ):
            spans.append((max(0.0, float(span[0])), min(duration, float(span[1]))))
    speech_seconds = sum(max(0.0, end - start) for start, end in spans)
    return {
        "asr_segment_count": len(spans),
        "asr_speech_ratio_estimate": round(min(1.0, speech_seconds / max(duration, 0.001)), 4),
        "asr_longest_segment_seconds": round(
            max((end - start for start, end in spans), default=0.0), 3
        ),
    }


def sidecar_name(names: set[str], cut_meta_name: str, suffix: str) -> str | None:
    base = cut_meta_name.removesuffix("_cut_meta.json")
    candidate = base + suffix
    return candidate if candidate in names else None


def iter_archive_candidates(
    zip_paths: Iterable[Path],
    caption_jsonl: Path,
    min_duration: float,
    max_duration: float,
) -> Iterable[dict[str, Any]]:
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            for cut_meta_name in sorted(
                name for name in names if name.endswith("_cut_meta.json")
            ):
                cut_meta = json.loads(archive.read(cut_meta_name))
                eligible = {
                    Path(str(item["clip_id"])).stem: item
                    for item in cut_meta.get("clips") or []
                    if min_duration
                    <= float(item.get("duration", 0))
                    <= max_duration
                }
                if not eligible:
                    continue
                asr_name = sidecar_name(names, cut_meta_name, "_asr.jsonl")
                asr_by_clip: dict[str, dict[str, Any]] = {}
                if asr_name:
                    for line in archive.read(asr_name).decode("utf-8").splitlines():
                        if line.strip():
                            value = json.loads(line)
                            asr_by_clip[str(value.get("clip_id"))] = value
                prefix = cut_meta_name.rsplit("/", 1)[0]
                for clip_id, item in eligible.items():
                    member = f"{prefix}/{clip_id}.mp4"
                    if member not in names:
                        continue
                    asr = asr_by_clip.get(clip_id, {})
                    language = normalize_language(asr.get("lang"))
                    if language is None:
                        continue
                    asr_text = text_from_asr(asr)
                    if contains_profanity(asr_text):
                        continue
                    duration = float(item["duration"])
                    yield {
                        "clip_id": clip_id,
                        "zip_path": str(zip_path),
                        "member": member,
                        "caption_jsonl": str(caption_jsonl),
                        "duration": duration,
                        "language_hint": language,
                        "source_asr": asr,
                        "source_span": item.get("span"),
                        **speech_statistics(asr, duration),
                    }


def metadata_score(row: dict[str, Any]) -> float:
    caption = row.get("caption_row") or {}
    visual = caption.get("caption") or {}
    meta = caption.get("meta") or {}
    text = " ".join(
        [
            str(meta.get("video_title", "")),
            str(visual.get("scene_title", "")),
            str(visual.get("content", "")),
        ]
    ).lower()
    positive = sum(term in text for term in POSITIVE_TERMS)
    negative = sum(term in text for term in NEGATIVE_TERMS)
    segment_count = int(row.get("asr_segment_count", 0))
    speech_ratio = float(row.get("asr_speech_ratio_estimate", 0.0))
    # Prefer potential event-driven scenes while retaining no-speech candidates.
    density_bonus = min(segment_count, 8) * 0.15
    speech_penalty = max(0.0, speech_ratio - 0.72) * 8.0
    return round(positive * 1.5 - negative * 2.5 + density_bonus - speech_penalty, 4)


def load_candidate_captions(
    candidates: dict[str, dict[str, Any]],
    caption_path: Path,
) -> None:
    wanted = set(candidates)
    with caption_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            clip_id = str(value.get("clip_id"))
            if clip_id in wanted:
                candidates[clip_id]["caption_row"] = value


def round_robin_languages(
    candidates: list[dict[str, Any]],
    limit: int,
    max_per_source: int,
) -> list[dict[str, Any]]:
    buckets = {
        language: sorted(
            [row for row in candidates if row["language_hint"] == language],
            key=lambda row: (float(row["metadata_score"]), row["clip_id"]),
            reverse=True,
        )
        for language in ("zh", "en")
    }
    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for index in range(max(len(bucket) for bucket in buckets.values())):
        for language in ("zh", "en"):
            if index >= len(buckets[language]) or len(selected) >= limit:
                continue
            row = buckets[language][index]
            source_id = str(
                ((row.get("caption_row") or {}).get("meta") or {}).get("video_id")
                or row["clip_id"].rsplit("-Scene-", 1)[0]
            )
            if source_counts[source_id] >= max_per_source:
                continue
            selected.append(row)
            source_counts[source_id] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vript-root",
        required=True,
        type=Path,
        help=(
            "Vript root containing vript_long_videos_clips/, "
            "vript_short_videos_clips/, and vript_captions/."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--long-zip-limit", type=int)
    parser.add_argument("--short-zip-limit", type=int)
    args = parser.parse_args()
    if args.max_per_source < 1:
        raise ValueError("--max-per-source must be >= 1")

    long_zips = sorted((args.vript_root / "vript_long_videos_clips").glob("*.zip"))
    short_zips = sorted((args.vript_root / "vript_short_videos_clips").glob("*.zip"))
    long_captions = (
        args.vript_root
        / "vript_captions"
        / "vript_long_videos_captions.jsonl"
    )
    short_captions = (
        args.vript_root
        / "vript_captions"
        / "vript_short_videos_captions.jsonl"
    )
    if args.long_zip_limit is not None:
        long_zips = long_zips[: args.long_zip_limit]
    if args.short_zip_limit is not None:
        short_zips = short_zips[: args.short_zip_limit]

    rows = list(
        iter_archive_candidates(
            long_zips, long_captions, args.min_duration, args.max_duration
        )
    )
    rows.extend(
        iter_archive_candidates(
            short_zips, short_captions, args.min_duration, args.max_duration
        )
    )
    by_caption_path: dict[Path, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_caption_path.setdefault(Path(row["caption_jsonl"]), {})[
            row["clip_id"]
        ] = row
    for caption_path, candidates in by_caption_path.items():
        load_candidate_captions(candidates, caption_path)
    for row in rows:
        row["metadata_score"] = metadata_score(row)

    selected = round_robin_languages(rows, args.limit, args.max_per_source)
    selection_rows = [
        {
            "clip_id": row["clip_id"],
            "zip_path": row["zip_path"],
            "member": row["member"],
            "caption_jsonl": row["caption_jsonl"],
            "source_group_id": str(
                ((row.get("caption_row") or {}).get("meta") or {}).get("video_id")
                or row["clip_id"].rsplit("-Scene-", 1)[0]
            ),
            "discovery": {
                key: row.get(key)
                for key in (
                    "duration",
                    "language_hint",
                    "source_span",
                    "asr_segment_count",
                    "asr_speech_ratio_estimate",
                    "asr_longest_segment_seconds",
                    "metadata_score",
                )
            },
        }
        for row in selected
    ]
    write_jsonl(args.output, selection_rows)
    if args.audit_output:
        write_jsonl(args.audit_output, rows)
    counts = Counter(row["language_hint"] for row in rows)
    selected_counts = Counter(row["language_hint"] for row in selected)
    print(json.dumps({
        "archives": {"long": len(long_zips), "short": len(short_zips)},
        "eligible_native_scenes": len(rows),
        "eligible_languages": counts,
        "selected_for_audio_screening": len(selected),
        "selected_unique_sources": len({
            str(
                ((row.get("caption_row") or {}).get("meta") or {}).get("video_id")
                or row["clip_id"].rsplit("-Scene-", 1)[0]
            )
            for row in selected
        }),
        "max_per_source": args.max_per_source,
        "selected_languages": selected_counts,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
