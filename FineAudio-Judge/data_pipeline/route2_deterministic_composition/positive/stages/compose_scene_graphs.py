#!/usr/bin/env python3
"""Compose auditable audio-caption pairs from arbitrary scene graphs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SAMPLE_RATE = 16_000
ROLE_RMS = {
    "ambience": 0.032,
    "music": 0.050,
    "speech": 0.105,
    "sound_effect": 0.075,
}
PROFANITY = re.compile(
    r"\b(fuck|fucking|shit|bitch|cunt|wanker|wankers|asshole)\b|"
    r"操你|草你|他妈的|妈的|傻逼|煞笔|狗日的",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        value = resample_poly(
            value, SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32)
    return value


def crossfade_loop(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) >= target_samples:
        return audio[:target_samples].copy()
    overlap = min(int(0.08 * SAMPLE_RATE), max(1, len(audio) // 8))
    output = audio.copy()
    while len(output) < target_samples:
        take = audio.copy()
        if len(output) >= overlap and len(take) >= overlap:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = 1.0 - fade_out
            merged = output[-overlap:] * fade_out + take[:overlap] * fade_in
            output = np.concatenate([output[:-overlap], merged, take[overlap:]])
        else:
            output = np.concatenate([output, take])
    return output[:target_samples]


def add_fades(audio: np.ndarray, role: str) -> np.ndarray:
    seconds = 0.015 if role == "speech" else (0.30 if role == "music" else 0.06)
    count = min(len(audio) // 2, int(seconds * SAMPLE_RATE))
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        audio[:count] *= ramp
        audio[-count:] *= ramp[::-1]
    return audio


def role_normalize(audio: np.ndarray, role: str, gain_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
    target = ROLE_RMS[role] * (10.0 ** (gain_db / 20.0))
    return audio * (target / rms)


def derive_relations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ordered = sorted(events, key=lambda item: (item["start"], item["end"]))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            overlap = min(left["end"], right["end"]) - max(
                left["start"], right["start"]
            )
            if overlap > 0.08:
                relation = "overlap"
            elif left["end"] <= right["start"]:
                relation = "before"
            else:
                continue
            result.append({
                "source_event_id": left["event_id"],
                "relation": relation,
                "target_event_id": right["event_id"],
            })
    return result


def event_caption_phrase(event: dict[str, Any]) -> str:
    if event.get("occurrence_caption_phrase"):
        return str(event["occurrence_caption_phrase"]).rstrip("。")
    if event["expected_count"] > 1:
        return (
            f"第{event['occurrence_index']}次{event['label']}出现"
        )
    return str(event["caption_phrase"]).rstrip("。")


def make_timestamp_caption(scene: dict[str, Any], events: list[dict[str, Any]]) -> str:
    sentences = []
    for event in sorted(events, key=lambda item: (item["start"], item["end"])):
        span = f"{event['start']:.1f}至{event['end']:.1f}秒"
        phrase = event_caption_phrase(event)
        if event["role"] == "speech":
            sentences.append(f"{span}，{phrase}：“{event['text']}”。")
        else:
            sentences.append(f"{span}，{phrase}。")
    return "".join(sentences)


def build_construction_controls(
    events: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        groups.setdefault(event["event_group_id"], []).append(event)
    event_groups: list[dict[str, Any]] = []
    for group_id, group in groups.items():
        ordered = sorted(group, key=lambda item: (item["start"], item["end"]))
        event_groups.append({
            "event_group_id": group_id,
            "label": ordered[0]["label"],
            "role": ordered[0]["role"],
            "expected_count": len(ordered),
            "occurrences": [
                {
                    "event_id": item["event_id"],
                    "occurrence_index": item["occurrence_index"],
                    "start": item["start"],
                    "end": item["end"],
                    "source_offset": item["source_offset"],
                    "gain_db": item["gain_db"],
                }
                for item in ordered
            ],
        })
    return {
        "event_groups": event_groups,
        "relations": relations,
    }


def foreground_overlap_qc(events: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    foreground = [event for event in events if event["role"] != "ambience"]
    points = sorted({event["start"] for event in foreground} | {event["end"] for event in foreground})
    spans = []
    for start, end in zip(points, points[1:]):
        count = sum(event["start"] < end and event["end"] > start for event in foreground)
        if count > 1:
            spans.append({"start": start, "end": end, "layers": count})
    merged: list[list[float]] = []
    for event in sorted(foreground, key=lambda item: (item["start"], item["end"])):
        if not merged or event["start"] > merged[-1][1]:
            merged.append([event["start"], event["end"]])
        else:
            merged[-1][1] = max(merged[-1][1], event["end"])
    gaps: list[dict[str, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            gaps.append({"start": round(cursor, 4), "end": round(start, 4), "duration": round(start - cursor, 4)})
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append({"start": round(cursor, 4), "end": round(duration, 4), "duration": round(duration - cursor, 4)})
    return {
        "max_foreground_layers": max([1] + [span["layers"] for span in spans]),
        "total_foreground_overlap_seconds": round(sum(span["end"] - span["start"] for span in spans), 3),
        "max_overlap_span_seconds": round(max([0.0] + [span["end"] - span["start"] for span in spans]), 3),
        "spans": spans,
        "max_foreground_gap_seconds": round(max([0.0] + [gap["duration"] for gap in gaps]), 3),
        "foreground_gaps": gaps,
    }


def loudness_normalize(raw_path: Path, output_path: Path) -> None:
    analysis = subprocess.run([
        "ffmpeg", "-hide_banner", "-i", str(raw_path),
        "-af", "loudnorm=I=-18:TP=-1.5:LRA=12:print_format=json",
        "-f", "null", "-",
    ], text=True, capture_output=True, check=True)
    matches = re.findall(r"\{[\s\S]*?\}", analysis.stderr)
    if not matches:
        raise RuntimeError(f"No first-pass loudness report for {raw_path}")
    measured = json.loads(matches[-1])
    loudnorm = (
        "loudnorm=I=-18:TP=-1.5:LRA=12:"
        f"measured_I={measured['input_i']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=false"
    )
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(raw_path),
        "-af", loudnorm,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ], check=True)


def loudness_normalize_single_pass(raw_path: Path, output_path: Path) -> dict[str, float]:
    """Normalize and obtain output loudness metrics in one FFmpeg invocation."""
    process = subprocess.run([
        "ffmpeg", "-hide_banner", "-y", "-i", str(raw_path),
        "-af", "loudnorm=I=-18:TP=-1.5:LRA=12:print_format=json",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ], text=True, capture_output=True, check=True)
    matches = re.findall(r"\{[\s\S]*?\}", process.stderr)
    if not matches:
        raise RuntimeError(f"No single-pass loudness report for {raw_path}")
    report = json.loads(matches[-1])
    return {
        "integrated_lufs": float(report["output_i"]),
        "true_peak_dbtp": float(report["output_tp"]),
        "loudness_range_lu": float(report["output_lra"]),
    }


def loudness_metrics(path: Path) -> dict[str, float]:
    process = subprocess.run([
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", "loudnorm=I=-18:TP=-1.5:LRA=12:print_format=json",
        "-f", "null", "-",
    ], text=True, capture_output=True, check=True)
    matches = re.findall(r"\{[\s\S]*?\}", process.stderr)
    if not matches:
        raise RuntimeError(f"No loudness report for {path}")
    report = json.loads(matches[-1])
    return {
        "integrated_lufs": float(report["input_i"]),
        "true_peak_dbtp": float(report["input_tp"]),
        "loudness_range_lu": float(report["input_lra"]),
    }


def silence_metrics(path: Path) -> dict[str, float]:
    """Measure long low-energy spans after final loudness normalization."""
    process = subprocess.run([
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", "silencedetect=noise=-42dB:d=2.5",
        "-f", "null", "-",
    ], text=True, capture_output=True, check=True)
    durations = [
        float(value) for value in re.findall(
            r"silence_duration:\s*([0-9]+(?:\.[0-9]+)?)", process.stderr
        )
    ]
    return {
        "max_detected_silence_seconds": round(max([0.0] + durations), 4),
        "total_detected_silence_seconds": round(sum(durations), 4),
        "silence_threshold_db": -42.0,
        "silence_minimum_duration_seconds": 2.5,
    }


def silence_metrics_numpy(path: Path) -> dict[str, float]:
    """Fast post-normalization silence measurement using 20 ms RMS frames."""
    audio = load_audio(path)
    frame = max(1, round(0.02 * SAMPLE_RATE))
    usable = len(audio) - (len(audio) % frame)
    if usable:
        rms = np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, frame)), axis=1) + 1e-12)
        quiet = rms < 10.0 ** (-42.0 / 20.0)
    else:
        quiet = np.zeros(0, dtype=bool)
    runs: list[float] = []
    start = None
    for index, value in enumerate(quiet):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((index - start) * frame / SAMPLE_RATE)
            start = None
    if start is not None:
        runs.append((len(quiet) - start) * frame / SAMPLE_RATE)
    long_runs = [value for value in runs if value >= 2.5]
    return {
        "max_detected_silence_seconds": round(max([0.0] + long_runs), 4),
        "total_detected_silence_seconds": round(sum(long_runs), 4),
        "silence_threshold_db": -42.0,
        "silence_minimum_duration_seconds": 2.5,
        "silence_detector": "numpy_20ms_rms",
    }


def compose(scene: dict[str, Any], output_dir: Path, fast: bool = False) -> dict[str, Any]:
    duration = float(scene["duration"])
    if not 30.0 <= duration <= 60.0:
        raise ValueError(f"{scene['id']}: duration outside 30–60 seconds")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = audio_dir / f"{scene['id']}.wav"
    reuse_existing = fast and output_path.exists()
    canvas = None if reuse_existing else np.zeros(round(duration * SAMPLE_RATE), dtype=np.float32)
    resolved_events: list[dict[str, Any]] = []

    for raw in scene["events"]:
        base_event = dict(raw)
        occurrences = base_event.pop("occurrences", None)
        placements = occurrences or [{
            "start": base_event["start"],
            "duration": base_event.get("duration"),
            "source_offset": base_event.get("source_offset", 0.0),
            "gain_db_delta": 0.0,
        }]
        role = base_event["role"]
        if role not in ROLE_RMS:
            raise ValueError(f"{scene['id']}: invalid role {role}")
        source_path = Path(base_event["source_path"])
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_duration = float(sf.info(source_path).duration)
        source = None if reuse_existing else load_audio(source_path)
        for occurrence_index, placement in enumerate(placements, start=1):
            event = dict(base_event)
            start = float(placement["start"])
            requested = placement.get("duration", event.get("duration"))
            source_offset = float(placement.get("source_offset", 0.0))
            source_start = max(0, round(source_offset * SAMPLE_RATE))
            available_samples = max(0, round(source_duration * SAMPLE_RATE) - source_start)
            event_duration = (
                available_samples / SAMPLE_RATE if requested is None else float(requested)
            )
            event_duration = min(event_duration, duration - start)
            target_samples = round(event_duration * SAMPLE_RATE)
            if target_samples <= 0:
                raise ValueError(
                    f"{scene['id']}: invalid placement {event['event_id']}"
                )
            if target_samples > available_samples and not event.get("loop"):
                target_samples = available_samples
                event_duration = target_samples / SAMPLE_RATE
            gain_db = float(event.get("gain_db", 0.0)) + float(
                placement.get("gain_db_delta", 0.0)
            )
            start_sample = round(start * SAMPLE_RATE)
            end_sample = min(round(duration * SAMPLE_RATE), start_sample + target_samples)
            if not reuse_existing:
                available = source[source_start:]
                segment = crossfade_loop(available, target_samples)
                segment = role_normalize(segment, role, gain_db)
                segment = add_fades(segment, role)
                canvas[start_sample:end_sample] += segment[: end_sample - start_sample]
            event_group_id = event["event_id"]
            occurrence_id = (
                event_group_id if len(placements) == 1
                else f"{event_group_id}.{occurrence_index}"
            )
            event.pop("start", None)
            event.pop("duration", None)
            event.pop("source_offset", None)
            resolved_events.append({
                **event,
                "event_id": occurrence_id,
                "event_group_id": event_group_id,
                "occurrence_index": occurrence_index,
                "expected_count": len(placements),
                "occurrence_caption_phrase": placement.get("caption_phrase"),
                "start": round(start, 4),
                "end": round(end_sample / SAMPLE_RATE, 4),
                "duration": round((end_sample - start_sample) / SAMPLE_RATE, 4),
                "source_offset": round(source_offset, 4),
                "gain_db": gain_db,
                "source_duration": round(source_duration, 4),
            })

    if reuse_existing:
        loudness = loudness_metrics(output_path)
    else:
        peak = float(np.max(np.abs(canvas)))
        if peak > 0.94:
            canvas *= 0.94 / peak
        raw_path = audio_dir / f".{scene['id']}.raw.wav"
        sf.write(raw_path, canvas, SAMPLE_RATE, subtype="PCM_16")
        if fast:
            loudness = loudness_normalize_single_pass(raw_path, output_path)
        else:
            loudness_normalize(raw_path, output_path)
            loudness = loudness_metrics(output_path)
        raw_path.unlink()

    resolved_events.sort(key=lambda item: (item["start"], item["end"]))
    caption = make_timestamp_caption(scene, resolved_events)
    if PROFANITY.search(caption):
        raise ValueError(f"{scene['id']}: profanity detected")
    for event in resolved_events:
        if event["role"] == "speech" and event["text"] not in caption:
            raise ValueError(f"{scene['id']}: speech text missing from caption")

    languages = sorted({
        event["language"] for event in resolved_events if event["role"] == "speech"
    })
    relations = derive_relations(resolved_events)
    construction_controls = build_construction_controls(
        resolved_events, relations
    )
    row = {
        "id": scene["id"],
        "path_name": "Path2_controltime_audio_caption_pairs",
        "title": scene.get("title") or f"随机组合样本 {scene['id']}",
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "audio_path": str(output_path),
        "background_mode": scene["background_mode"],
        "layers_present": sorted({event["role"] for event in resolved_events}),
        "speech_languages": languages,
        "music_present": any(event["role"] == "music" for event in resolved_events),
        "speech_present": bool(languages),
        "audio_caption": caption,
        "audio_caption_timestamped": caption,
        "audio_caption_natural": None,
        "natural_caption_status": "pending_llm_rewrite",
        "events": resolved_events,
        "construction_controls": construction_controls,
        "overlap_qc": foreground_overlap_qc(resolved_events, duration),
        "mix_policy": scene.get("mix_policy", {}),
        "signal_qc": {
            **loudness,
            **(silence_metrics_numpy(output_path) if fast else silence_metrics(output_path)),
        },
        "construction": "scene_graph_deterministic_composition",
        "generation_mode": scene.get("generation_mode", "manual"),
        "sampling_seed": scene.get("sampling_seed"),
    }
    sidecar_dir = output_dir / "rows"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / f"{scene['id']}.json"
    temporary = sidecar_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    temporary.replace(sidecar_path)
    return row


def validate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = []
    for row in rows:
        errors = []
        if not 30 <= row["duration"] <= 60:
            errors.append("duration")
        if len(row["events"]) < 4:
            errors.append("event_count")
        if row["signal_qc"]["true_peak_dbtp"] > -0.9:
            errors.append("true_peak")
        if not -21 <= row["signal_qc"]["integrated_lufs"] <= -15:
            errors.append("lufs")
        if row["signal_qc"]["max_detected_silence_seconds"] > 2.5:
            errors.append("large_silence")
        max_foreground_gap = float(
            row.get("mix_policy", {}).get("max_foreground_gap_seconds", 3.0)
        )
        if row["overlap_qc"]["max_foreground_gap_seconds"] > max_foreground_gap:
            errors.append("large_foreground_gap")
        for event in row["events"]:
            if event["role"] == "speech" and event["text"] not in row["audio_caption"]:
                errors.append(f"speech_coverage:{event['event_id']}")
            if (
                event.get("expected_count", 1) > 1
                and not event.get("atomic_occurrence_verified", False)
            ):
                errors.append(f"unverified_count_control:{event['event_id']}")
        confusion_families: dict[str, list[str]] = {}
        for event in row["events"]:
            family = event.get("acoustic_confusion_family")
            if family:
                confusion_families.setdefault(str(family), []).append(event["event_id"])
        for family, event_ids in confusion_families.items():
            if len(event_ids) > 1:
                errors.append(
                    f"acoustic_confusion_family:{family}:{','.join(event_ids)}"
                )
        groups: dict[str, list[dict[str, Any]]] = {}
        for event in row["events"]:
            groups.setdefault(event["event_group_id"], []).append(event)
        for group_id, group in groups.items():
            expected = group[0]["expected_count"]
            if len(group) != expected:
                errors.append(f"frequency:{group_id}")
            if expected > 1:
                for event in group:
                    if event_caption_phrase(event) not in row["audio_caption"]:
                        errors.append(
                            f"caption_occurrence:{event['event_id']}"
                        )
        report.append({"id": row["id"], "valid": not errors, "errors": errors})
    speech = [row for row in rows if row["speech_present"]]
    language_counts = {
        lang: sum(lang in row["speech_languages"] for row in speech)
        for lang in ("zh", "en")
    }
    repeat_samples = [
        row for row in rows
        if any(event["expected_count"] > 1 for event in row["events"])
    ]
    summary = {
        "samples": len(rows),
        "valid": sum(item["valid"] for item in report),
        "invalid": sum(not item["valid"] for item in report),
        "speech_samples": len(speech),
        "music_samples": sum(row["music_present"] for row in rows),
        "repeat_control_samples": len(repeat_samples),
        "repeat_groups": sum(
            len({event["event_group_id"] for event in row["events"] if event["expected_count"] > 1})
            for row in rows
        ),
        "overlap_samples": sum(
            row["overlap_qc"]["total_foreground_overlap_seconds"] > 0.08
            for row in rows
        ),
        "foreground_overlap_count_distribution": {
            str(count): sum(
                sum(
                    span["layers"] > 1
                    for span in row["overlap_qc"]["spans"]
                ) == count
                for row in rows
            )
            for count in (0, 1, 2)
        },
        "background_modes": sorted({row["background_mode"] for row in rows}),
        "speech_language_counts": language_counts,
    }
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-graphs", type=Path,
        default=Path(__file__).parent / "demo_scene_graphs.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "runs" / "demo",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Validation-only prefix length; 0 processes every scene. "
            "Normal public runs leave this at 0."
        ),
    )
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = read_jsonl(args.scene_graphs)
    if args.limit:
        scenes = scenes[: args.limit]
    partial_path = args.output_dir / "pairs.partial.jsonl"
    rows = read_jsonl(partial_path) if args.resume and partial_path.exists() else []
    completed = {row["id"] for row in rows}
    if args.resume:
        recovered = []
        for sidecar in sorted((args.output_dir / "rows").glob("*.json")):
            row = json.loads(sidecar.read_text(encoding="utf-8"))
            if row["id"] not in completed:
                rows.append(row)
                recovered.append(row)
                completed.add(row["id"])
        if recovered:
            with partial_path.open("a", encoding="utf-8") as handle:
                for row in recovered:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"recovered_sidecars": len(recovered)}), flush=True)
    pending = [scene for scene in scenes if scene["id"] not in completed]
    if args.workers == 1:
        iterator = (compose(scene, args.output_dir, args.fast) for scene in pending)
        for row in iterator:
            rows.append(row)
            completed.add(row["id"])
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(rows) % 10 == 0:
                print(json.dumps({"composed": len(rows), "total": len(scenes)}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(compose, scene, args.output_dir, args.fast): scene["id"]
                for scene in pending
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed.add(row["id"])
                with partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(rows) % 10 == 0:
                    print(json.dumps({"composed": len(rows), "total": len(scenes)}), flush=True)
    report, summary = validate_rows(rows)
    write_jsonl(args.output_dir / "pairs.jsonl", rows)
    write_jsonl(args.output_dir / "scene_graphs_resolved.jsonl", rows)
    write_jsonl(args.output_dir / "validation_report.jsonl", report)
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
