#!/usr/bin/env python3
"""Run the same four-round audio analysis for every source dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CORE_CODE = HERE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def whisper_candidate(gate: dict[str, Any], source: dict[str, Any]) -> Any:
    segments = [
        {"start": row.get("start"), "end": row.get("end"), "text": row.get("text"), "reliable": row.get("reliable")}
        for row in gate.get("whisper_segments") or [] if str(row.get("text") or "").strip()
    ]
    return segments or source.get("source_asr") or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(CORE_CODE))
    from prompts import detective_chronology_prompt, detective_sound_prompt, detective_speech_prompt, detective_synthesis_prompt
    from omni_backend import VllmOmniBatchRunner

    config = json.loads(args.config.read_text(encoding="utf-8"))
    generation = config["generation"]
    output = args.run_root / "outputs" / f"{args.dataset}_audio_analysis.jsonl"
    if args.overwrite:
        output.unlink(missing_ok=True)
    done = {str(row["id"]) for row in read_jsonl(output)}
    rows = [
        row for row in read_jsonl(args.run_root / "data" / args.dataset / "selected_manifest.jsonl")
        if str(row["id"]) not in done
    ]
    gates = by_id(read_jsonl(args.run_root / "outputs" / f"{args.dataset}_language_gate.jsonl"))
    runner = VllmOmniBatchRunner(
        generation["omni_model"], int(generation["omni_tensor_parallel"]),
        max_num_seqs=8, max_model_len=32768, submit_batch_size=16,
    )
    provenance = {
        "pipeline_name": config["pipeline_name"], "protocol": config["protocol"],
        "stage": "four_round_audio_analysis", "model": generation["omni_model"],
        "source_annotations_are_caption_evidence": False,
    }
    for start in range(0, len(rows), 16):
        chunk = rows[start:start + 16]
        chronologies = runner.generate([
            ("audio", row["audio_path"], detective_chronology_prompt(), False) for row in chunk
        ], 1000)
        speeches = runner.generate([
            ("audio", row["audio_path"], detective_speech_prompt(
                whisper_candidate(gates.get(str(row["id"]), {}), row),
                row.get("trusted_time_annotations") or [],
            ), False) for row in chunk
        ], 1500)
        sounds = runner.generate([
            ("audio", row["audio_path"], detective_sound_prompt(), False) for row in chunk
        ], 1000)
        observations = [
            {"chronology": chronology, "speech": speech, "sound": sound}
            for chronology, speech, sound in zip(chronologies, speeches, sounds, strict=True)
        ]
        captions = runner.generate([
            ("audio", row["audio_path"], detective_synthesis_prompt(
                observation,
                row.get("trusted_time_annotations") or [],
                (row.get("independent_sound_verification") or {}).get("verified_non_speech_events") or [],
            ), False)
            for row, observation in zip(chunk, observations, strict=True)
        ], 2600)
        for row, observation, caption in zip(chunk, observations, captions, strict=True):
            append_jsonl(output, {
                "id": row["id"], "dataset": row.get("dataset"),
                "pipeline": "route1_audio_analysis",
                "source": row, "investigation_rounds": observation,
                "audio_caption": caption, "model": generation["omni_model"],
                "backend": "vllm_offline_batch", "provenance": provenance,
            })
            print(json.dumps({"id": row["id"], "stage": "detective"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
