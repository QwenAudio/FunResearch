#!/usr/bin/env python3
"""Normalize arbitrary video/audio JSONL manifests for Route 1.

Required per row: ``id``, ``dataset``, and at least one of ``audio_path`` or
``video_path``. Relative media paths are resolved from the manifest directory.
All additional annotations are retained as source metadata, not caption truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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


def media_path(value: Any, manifest: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def safe_stem(sample_id: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._-") or "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{prefix[:80]}_{digest}"


def extract_audio(video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.wav")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names to ingest")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    selected = {value.strip().lower() for value in args.datasets.split(",") if value.strip()}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line_number, raw in enumerate(read_jsonl(manifest), start=1):
        sample_id = str(raw.get("id") or "").strip()
        dataset = str(raw.get("dataset") or "").strip().lower()
        if not sample_id or not dataset:
            raise ValueError(f"{manifest}:{line_number}: id and dataset are required")
        if dataset not in selected:
            continue
        audio = media_path(raw.get("audio_path"), manifest)
        video = media_path(raw.get("video_path"), manifest)
        if audio is None and video is None:
            raise ValueError(f"{manifest}:{line_number}: audio_path or video_path is required")
        if audio is None:
            audio = args.run_root / "data" / dataset / "audios" / f"{safe_stem(sample_id)}.wav"
            if not audio.exists():
                extract_audio(video, audio)
        measured_duration = float(raw.get("duration") or duration(audio))
        source_locator = dict(raw.get("source_locator") or {})
        source_locator.update({"input_manifest": str(manifest), "line_number": line_number})
        row = {
            **raw,
            "id": sample_id,
            "dataset": dataset,
            "duration": measured_duration,
            "audio_path": str(audio),
            "video_path": str(video) if video else None,
            "source_group_id": raw.get("source_group_id") or f"{dataset}:{sample_id}",
            "source_locator": source_locator,
            "source_annotations_are_caption_evidence": False,
        }
        grouped.setdefault(dataset, []).append(row)

    for dataset, incoming in grouped.items():
        output = args.run_root / "data" / dataset / "source_manifest.jsonl"
        existing = read_jsonl(output)
        by_id = {str(row["id"]): row for row in existing}
        for row in incoming:
            sample_id = str(row["id"])
            if sample_id in by_id:
                raise ValueError(f"duplicate id in dataset {dataset}: {sample_id}")
            by_id[sample_id] = row
        write_jsonl(output, list(by_id.values()))
        print(json.dumps({"dataset": dataset, "added": len(incoming), "total": len(by_id)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
