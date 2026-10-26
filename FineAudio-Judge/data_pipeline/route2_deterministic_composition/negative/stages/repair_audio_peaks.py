#!/usr/bin/env python3
"""Attenuate only failed true-peak outputs and rebuild deterministic QC files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


POSITIVE_STAGES = Path(__file__).resolve().parents[2] / "positive" / "stages"
sys.path.insert(0, str(POSITIVE_STAGES))
from compose_scene_graphs import loudness_metrics, silence_metrics, validate_rows  # noqa: E402


def write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix-dir", type=Path, required=True)
    parser.add_argument("--target-dbtp", type=float, default=-1.05)
    args = parser.parse_args()
    sidecars = sorted((args.mix_dir / "rows").glob("*.json"))
    rows = []
    repaired = 0
    for sidecar in sidecars:
        row = json.loads(sidecar.read_text(encoding="utf-8"))
        measured = float(row["signal_qc"]["true_peak_dbtp"])
        if measured > args.target_dbtp:
            audio_path = Path(row["audio_path"])
            attenuation = args.target_dbtp - measured
            temporary_audio = audio_path.with_suffix(".peakfix.wav")
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
                "-af", f"volume={attenuation:.4f}dB", "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", str(temporary_audio),
            ], check=True)
            temporary_audio.replace(audio_path)
            row["signal_qc"] = {**loudness_metrics(audio_path), **silence_metrics(audio_path)}
            row["signal_qc"]["true_peak_repair_db"] = round(attenuation, 4)
            row["signal_qc"]["true_peak_repair_target_dbtp"] = args.target_dbtp
            temporary_json = sidecar.with_suffix(".json.tmp")
            temporary_json.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            temporary_json.replace(sidecar)
            repaired += 1
        rows.append(row)

    rows.sort(key=lambda row: row["id"])
    report, summary = validate_rows(rows)
    write_jsonl(args.mix_dir / "pairs.partial.jsonl", rows)
    write_jsonl(args.mix_dir / "pairs.jsonl", rows)
    write_jsonl(args.mix_dir / "validation_report.jsonl", report)
    (args.mix_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sidecars": len(sidecars), "repaired": repaired, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
