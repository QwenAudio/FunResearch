#!/usr/bin/env python3
"""Recompose only failed mixes with two-pass loudness and rebuild manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from compose_scene_graphs import (
    SAMPLE_RATE, compose, load_audio, loudness_metrics, role_normalize,
    silence_metrics, validate_rows, write_jsonl,
)
from asset_pool import family


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_scene(scene):
    """Close true acoustic gaps while preserving order and overlap classes."""
    foreground = sorted(
        [event for event in scene["events"] if event["role"] != "ambience"],
        key=lambda event: (event["start"], event["event_id"]),
    )
    relation = []
    for previous, current in zip(foreground, foreground[1:]):
        previous_end = float(previous["start"]) + float(previous["duration"])
        if abs(float(current["start"]) - float(previous["start"])) <= 0.02:
            relation.append("co_onset")
        elif float(current["start"]) < previous_end - 0.08:
            relation.append("overlap")
        else:
            relation.append("after")

    def place(gap):
        cursor, previous_start, previous_end = gap, 0.0, 0.0
        for index, event in enumerate(foreground):
            if index:
                if relation[index - 1] == "co_onset":
                    cursor = previous_start
                elif relation[index - 1] == "overlap":
                    overlap = min(1.0, float(foreground[index - 1]["duration"]) * 0.3, float(event["duration"]) * 0.3)
                    cursor = previous_end - overlap
                else:
                    cursor = previous_end + gap
            event["start"] = round(cursor, 2)
            previous_start = cursor
            previous_end = max(previous_end, cursor + float(event["duration"]))
        return previous_end + gap

    # If the compact arrangement would be shorter than 30 s, use more of the
    # already selected strongly annotated source intervals before adding gaps.
    duration_at_one_second_gaps = place(1.0)
    if duration_at_one_second_gaps < 30.0:
        shortage = 30.0 - duration_at_one_second_gaps
        for event in foreground:
            if event["role"] != "sound_effect" or shortage <= 0:
                continue
            available = float(sf.info(event["source_path"]).duration)
            increase = min(shortage, max(0.0, available - float(event["duration"])))
            event["duration"] = round(float(event["duration"]) + increase, 2)
            shortage -= increase

    after_slots = sum(value == "after" for value in relation) + 2
    base_duration = place(0.0)
    gap = min(1.0, max(0.25, (30.0 - base_duration) / max(1, after_slots)))
    compact_duration = place(gap)
    scene["duration"] = round(max(30.0, compact_duration), 2)
    scene["mix_policy"]["max_foreground_gap_seconds"] = max(1.05, round(gap + 0.05, 2))
    scene["mix_policy"]["silence_qc_repair"] = "compact_timeline_preserving_overlap_classes"
    for event in scene["events"]:
        if event["role"] == "ambience":
            event["duration"] = scene["duration"]
    return scene


def max_source_quiet(event, path=None):
    source_path = Path(path or event["source_path"])
    audio = load_audio(source_path)[:round(float(event["duration"]) * SAMPLE_RATE)]
    audio = role_normalize(audio, event["role"], float(event.get("gain_db", 0.0)))
    frame = round(0.02 * SAMPLE_RATE)
    audio = audio[:len(audio) - len(audio) % frame]
    if not len(audio):
        return float("inf")
    quiet = np.sqrt(np.mean(np.square(audio.reshape(-1, frame)), axis=1) + 1e-12) < 10 ** (-42 / 20)
    longest = current = 0
    for value in quiet:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * frame / SAMPLE_RATE


def replace_silent_sources(scene, asset_rows, used_paths):
    for event in scene["events"]:
        if event["role"] == "speech" or max_source_quiet(event) <= 2.0:
            continue
        blocked_families = {
            other.get("acoustic_confusion_family") for other in scene["events"]
            if other is not event and other.get("acoustic_confusion_family")
        }
        candidates = [
            row for row in asset_rows
            if row["role"] == event["role"]
            and row["source_path"] not in used_paths
            and float(row["duration"]) >= min(4.2, float(event["duration"]))
            and family(row["slug"]) not in blocked_families
        ]
        candidates.sort(key=lambda row: (row["slug"] != event["asset_slug"], row["slug"], row["video_id"]))
        chosen = next((row for row in candidates if max_source_quiet(event, row["source_path"]) <= 1.0), None)
        if chosen is None:
            raise RuntimeError(f"no dense unused replacement for {scene['id']}:{event['event_id']}")
        used_paths.remove(event["source_path"])
        used_paths.add(chosen["source_path"])
        event.update({
            "source_dataset": chosen["source_dataset"], "source_id": chosen["video_id"],
            "source_path": chosen["source_path"], "source_event_name": chosen["event_name"],
            "source_event_start": chosen["event_start"], "source_event_end": chosen["event_end"],
            "duration": round(min(float(event["duration"]), float(chosen["duration"])), 2),
            "label": chosen["label"], "asset_slug": chosen["slug"],
            "acoustic_confusion_family": family(chosen["slug"]),
            "caption_phrase": f"{chosen['label']}出现",
        })
        scene["mix_policy"].setdefault("qc_source_replacements", []).append({
            "event_id": event["event_id"], "reason": "source_interval_contains_long_silence",
            "replacement_source_id": chosen["video_id"],
        })


def attenuate_peak(sidecar_path):
    row = json.loads(sidecar_path.read_text(encoding="utf-8"))
    peak = float(row["signal_qc"]["true_peak_dbtp"])
    if peak <= -1.0:
        return
    audio = Path(row["audio_path"])
    temporary = audio.with_suffix(".peakfix.wav")
    delta = -1.2 - peak
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(audio),
        "-af", f"volume={delta:.3f}dB", "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", str(temporary),
    ], check=True)
    temporary.replace(audio)
    row["signal_qc"] = {**loudness_metrics(audio), **silence_metrics(audio)}
    sidecar_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graphs", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    report_path = args.output_dir / "validation_report.jsonl"
    failed = {row["id"]: row["errors"] for row in read_jsonl(report_path) if not row["valid"]}
    all_scenes = read_jsonl(args.scene_graphs)
    asset_rows = read_jsonl(args.asset_manifest)
    used_paths = {event["source_path"] for scene in all_scenes for event in scene["events"]}
    scenes = []
    for scene in all_scenes:
        if scene["id"] not in failed:
            continue
        if any(error in {"large_silence", "large_foreground_gap"} for error in failed[scene["id"]]):
            replace_silent_sources(scene, asset_rows, used_paths)
            compact_scene(scene)
        scenes.append(scene)
    write_jsonl(args.scene_graphs, all_scenes)
    print(json.dumps({"repairing": len(scenes), "mode": "two_pass_loudness"}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compose, scene, args.output_dir, False) for scene in scenes]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 10 == 0:
                print(json.dumps({"recomposed": index, "total": len(scenes)}), flush=True)
    for sample_id in failed:
        attenuate_peak(args.output_dir / "rows" / f"{sample_id}.json")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.output_dir / "rows").glob("*.json"))]
    report, summary = validate_rows(rows)
    write_jsonl(args.output_dir / "pairs.jsonl", rows)
    write_jsonl(args.output_dir / "scene_graphs_resolved.jsonl", rows)
    write_jsonl(args.output_dir / "validation_report.jsonl", report)
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["invalid"]:
        raise SystemExit(f"{summary['invalid']} mixes still fail QC after repair")


if __name__ == "__main__":
    main()
