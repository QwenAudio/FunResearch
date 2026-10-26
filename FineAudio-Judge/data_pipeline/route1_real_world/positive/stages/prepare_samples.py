from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from common import append_jsonl, write_jsonl
from data_policy import FINEVIDEO_CLIP_SUFFIX
from pipeline_meta import DEFAULT_CONFIG, load_config


def parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return None
    return None


def vript_dataset_time_annotations(
    asr_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for asr_row in asr_rows:
        for segment in asr_row.get("scene_text") or []:
            span = segment.get("span") if isinstance(segment, dict) else None
            if (
                not isinstance(span, list)
                or len(span) != 2
                or not all(isinstance(value, (int, float)) for value in span)
                or float(span[1]) <= float(span[0])
            ):
                continue
            annotations.append({
                "start": float(span[0]),
                "end": float(span[1]),
                "text": segment.get("text"),
                "annotation_type": "speech_span",
                "annotation_source": "dataset_gt",
                "is_dataset_gt": True,
                "dataset": "Vript",
            })
    return annotations


def finevideo_dataset_time_annotations(
    timecoded_asr: list[dict[str, Any]],
    *,
    source_start: float,
    source_end: float,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for segment in timecoded_asr:
        start = parse_timestamp(segment.get("start"))
        end = parse_timestamp(segment.get("end"))
        if start is None or end is None or end <= source_start or start >= source_end:
            continue
        clip_start = max(source_start, start) - source_start
        clip_end = min(source_end, end) - source_start
        if clip_end <= clip_start:
            continue
        annotations.append({
            "start": round(clip_start, 3),
            "end": round(clip_end, 3),
            "source_start": start,
            "source_end": end,
            "text": segment.get("text"),
            "annotation_type": "speech_span",
            "annotation_source": "dataset_gt",
            "is_dataset_gt": True,
            "dataset": "FineVideo",
        })
    return annotations


def load_vript_selection(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        raise ValueError("--vript-selection is required when preparing Vript")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({
                    **row,
                    "clip_id": str(row["clip_id"]),
                    "zip_path": str(row["zip_path"]),
                    "member": str(row["member"]),
                    "caption_jsonl": str(row["caption_jsonl"]),
                    "trusted_time_annotations": row.get("trusted_time_annotations") or [],
                })
    return rows


def load_finevideo_selection(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        raise ValueError("--finevideo-selection is required when preparing FineVideo")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({
                    **row,
                    "parquet": str(row["parquet"]),
                    "row_index": int(row["row_index"]),
                })
    return rows


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def make_audio(video: Path, wav: Path) -> None:
    if wav.exists():
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )


def load_caption_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            rows[str(value["clip_id"])] = value
    return rows


def find_zip_sidecar(archive: zipfile.ZipFile, video_id: str, suffix: str) -> str | None:
    candidates = [name for name in archive.namelist() if name.endswith(f"/{video_id}{suffix}")]
    return candidates[0] if candidates else None


def prepare_vript(
    root: Path,
    selection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    data_dir = root / "data" / "vript"
    video_dir, audio_dir = data_dir / "videos", data_dir / "audios"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    # Discovery records the exact caption JSONL beside each selected archive.
    # Loading those recorded paths keeps preparation portable without changing
    # which frozen source annotation is used.
    caption_paths = {Path(row["caption_jsonl"]) for row in selection}
    caption_cache = {path: load_caption_rows(path) for path in caption_paths}
    rows: list[dict[str, Any]] = []
    archives: dict[Path, zipfile.ZipFile] = {}
    try:
        for selected in selection:
            clip_id = str(selected["clip_id"])
            zip_path = Path(selected["zip_path"])
            member = str(selected["member"])
            caption_path = Path(selected["caption_jsonl"])
            archive = archives.setdefault(zip_path, zipfile.ZipFile(zip_path))
            video_path = video_dir / f"{clip_id}.mp4"
            audio_path = audio_dir / f"{clip_id}.wav"
            if not video_path.exists():
                with archive.open(member) as source, video_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
            make_audio(video_path, audio_path)
            video_id = clip_id.rsplit("-Scene-", 1)[0]
            asr_name = find_zip_sidecar(archive, video_id, "_asr.jsonl")
            asr_rows: list[dict[str, Any]] = []
            if asr_name:
                for line in archive.read(asr_name).decode("utf-8").splitlines():
                    value = json.loads(line)
                    if value.get("clip_id") == clip_id:
                        asr_rows.append(value)
            caption = caption_cache[caption_path][clip_id]
            d = duration(video_path)
            duration_rules = load_config(DEFAULT_CONFIG)["selection"]["duration_seconds"]
            if not (float(duration_rules["min"]) <= d <= float(duration_rules["max"])):
                append_jsonl(root / "outputs" / "vript_preparation_rejected.jsonl", {
                    "id": clip_id,
                    "dataset": "Vript",
                    "stage": "media_preparation",
                    "rejection_reasons": ["actual_duration_outside_configured_range"],
                    "actual_duration": d,
                    "discovery_duration": (selected.get("discovery") or {}).get("duration"),
                    "configured_duration_range": duration_rules,
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "source_locator": {"archive": str(zip_path), "member": member},
                })
                continue
            rows.append(
                {
                    "id": clip_id,
                    "dataset": "Vript",
                    "source_group_id": video_id,
                    "duration": d,
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "source_locator": {
                        "archive": str(zip_path),
                        "member": member,
                        "caption_jsonl": str(caption_path),
                        "clip_id": clip_id,
                        "video_id": video_id,
                        "composition": "native_scene",
                    },
                    "is_composite": False,
                    "source_asr": asr_rows or caption.get("voiceover", ""),
                    "source_visual_caption": caption.get("caption"),
                    "source_voiceover": caption.get("voiceover"),
                    "source_meta": caption.get("meta"),
                    "trusted_time_annotations": (
                        vript_dataset_time_annotations(asr_rows)
                        + (selected.get("trusted_time_annotations") or [])
                    ),
                }
            )
    finally:
        for archive in archives.values():
            archive.close()
    return rows


def prepare_finevideo(
    root: Path,
    selection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    data_dir = root / "data" / "finevideo"
    video_dir, audio_dir = data_dir / "videos", data_dir / "audios"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    current_parquet: str | None = None
    table: Any = None
    for selected in sorted(selection, key=lambda row: (str(row["parquet"]), int(row["row_index"]))):
        parquet_path = str(selected["parquet"])
        row_index = int(selected["row_index"])
        if parquet_path != current_parquet:
            table = pq.read_table(parquet_path)
            current_parquet = parquet_path
        source = table.slice(row_index, 1).to_pylist()[0]
        meta = source["json"]
        filename = str(meta["original_video_filename"])
        source_start = float(selected.get("start_seconds", 0.0))
        source_end = float(selected.get("end_seconds", meta.get("duration_seconds") or 0))
        is_crop = source_start > 0 or source_end < float(meta.get("duration_seconds") or source_end) - 0.01
        sample_id = Path(filename).stem
        if is_crop:
            sample_id += f"_s{round(source_start * 1000):09d}_e{round(source_end * 1000):09d}"
        video_path = video_dir / f"{sample_id}.mp4"
        audio_path = audio_dir / f"{sample_id}.wav"
        if not video_path.exists():
            if not is_crop:
                video_path.write_bytes(source["mp4"])
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp4") as temporary:
                    temporary.write(source["mp4"])
                    temporary.flush()
                    subprocess.run(
                        [
                            "ffmpeg", "-v", "error", "-y",
                            "-ss", f"{source_start:.3f}",
                            "-i", temporary.name,
                            "-t", f"{source_end - source_start:.3f}",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-c:a", "aac", "-movflags", "+faststart",
                            str(video_path),
                        ],
                        check=True,
                    )
        make_audio(video_path, audio_path)
        d = duration(video_path)
        duration_rules = load_config(DEFAULT_CONFIG)["selection"]["duration_seconds"]
        if not (float(duration_rules["min"]) <= d <= float(duration_rules["max"])):
            append_jsonl(root / "outputs" / "finevideo_preparation_rejected.jsonl", {
                "id": sample_id,
                "dataset": "FineVideo",
                "stage": "media_preparation",
                "rejection_reasons": ["actual_duration_outside_configured_range"],
                "actual_duration": d,
                "discovery_duration": source_end - source_start,
                "configured_duration_range": duration_rules,
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "source_locator": {
                    "parquet": parquet_path,
                    "row_index": row_index,
                    "source_start_seconds": source_start,
                    "source_end_seconds": source_end,
                },
            })
            continue
        visual_meta = {
            "content_parent_category": meta.get("content_parent_category"),
            "content_fine_category": meta.get("content_fine_category"),
            "youtube_title": meta.get("youtube_title"),
            "content_metadata": meta.get("content_metadata"),
            "selected_scene": {
                "scene_id": selected.get("scene_id"),
                "scene_title": selected.get("scene_title"),
                "scene_context": selected.get("scene_context"),
            },
        }
        timecoded_asr = meta.get("timecoded_text_to_speech") or []
        if is_crop and timecoded_asr:
            cropped_asr = []
            for segment in timecoded_asr:
                start = parse_timestamp(segment.get("start"))
                end = parse_timestamp(segment.get("end"))
                if start is None or end is None or end <= source_start or start >= source_end:
                    continue
                cropped_asr.append({
                    **segment,
                    "source_start_seconds": start,
                    "source_end_seconds": end,
                    "clip_start_seconds": max(0.0, start - source_start),
                    "clip_end_seconds": min(source_end, end) - source_start,
                })
            source_asr: Any = cropped_asr
        else:
            source_asr = timecoded_asr or meta.get("text_to_speech")
        rows.append(
            {
                "id": sample_id,
                "dataset": "FineVideo",
                "source_group_id": FINEVIDEO_CLIP_SUFFIX.sub("", Path(filename).stem),
                "duration": d,
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "source_locator": {
                    "parquet": parquet_path,
                    "row_index": row_index,
                    "original_video_filename": filename,
                    "source_start_seconds": source_start,
                    "source_end_seconds": source_end,
                    "selection_basis": selected.get("selection_basis", "native_full_video"),
                    "scene_id": selected.get("scene_id"),
                    "composition": (
                        "continuous_source_crop" if is_crop else "native_row"
                    ),
                },
                "is_composite": False,
                "source_asr": source_asr,
                "finevideo_visual_metadata": visual_meta,
                "finevideo_original_json": meta,
                "trusted_time_annotations": (
                    finevideo_dataset_time_annotations(
                        timecoded_asr,
                        source_start=source_start,
                        source_end=source_end,
                    )
                    + (selected.get("trusted_time_annotations") or [])
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--vript-selection",
        type=Path,
        help="JSONL containing clip_id, zip_path, member, and caption_jsonl.",
    )
    parser.add_argument(
        "--finevideo-selection",
        type=Path,
        help="JSONL containing parquet and row_index.",
    )
    parser.add_argument(
        "--dataset",
        choices=("both", "vript", "finevideo"),
        default="both",
        help="Prepare both sources or only one dataset.",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    vript = []
    finevideo = []
    if args.dataset in ("both", "vript"):
        vript = prepare_vript(args.root, load_vript_selection(args.vript_selection))
        write_jsonl(args.root / "data" / "vript" / "source_manifest.jsonl", vript)
    if args.dataset in ("both", "finevideo"):
        finevideo = prepare_finevideo(
            args.root, load_finevideo_selection(args.finevideo_selection)
        )
        write_jsonl(
            args.root / "data" / "finevideo" / "source_manifest.jsonl", finevideo
        )
    print(json.dumps({"vript": len(vript), "finevideo": len(finevideo)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
