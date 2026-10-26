#!/usr/bin/env python3
"""Language-aware Qwen3-Omni screening and independent sound relistening."""

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


def parse(text: str) -> tuple[dict[str, Any], str | None]:
    from common import parse_json_object
    try:
        return parse_json_object(text), None
    except Exception as exc:
        return {"decision": "reject", "reasons": ["json_parse_failed"]}, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=("screen", "verify"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(CORE_CODE))
    from prompts import candidate_screening_prompt, non_speech_verification_prompt
    from omni_backend import VllmOmniBatchRunner

    config = json.loads(args.config.read_text(encoding="utf-8"))
    generation = config["generation"]
    selection_rules = config["selection"]
    runner = VllmOmniBatchRunner(
        generation["omni_model"], int(generation["omni_tensor_parallel"]),
        max_num_seqs=8, max_model_len=32768, submit_batch_size=16,
    )
    gates = by_id(read_jsonl(args.run_root / "outputs" / f"{args.dataset}_language_gate.jsonl"))

    if args.stage == "screen":
        output = args.run_root / "outputs" / f"{args.dataset}_screening.jsonl"
        sources = read_jsonl(args.run_root / "data" / args.dataset / "source_manifest.jsonl")
        rows = [row for row in sources if str(gates.get(str(row["id"]), {}).get("language_policy_decision", "")).startswith("pass_")]
        prompt = lambda row: candidate_screening_prompt(row, selection_rules)
    else:
        output = args.run_root / "outputs" / f"{args.dataset}_sound_verification.jsonl"
        rows = read_jsonl(args.run_root / "outputs" / f"{args.dataset}_screening.jsonl")
        prompt = lambda row: non_speech_verification_prompt(
            row["source"], row.get("assessment"), selection_rules
        )
    if args.overwrite:
        output.unlink(missing_ok=True)
    done = {str(row["id"]) for row in read_jsonl(output)}
    rows = [row for row in rows if str(row["id"]) not in done]

    for start in range(0, len(rows), 16):
        chunk = rows[start:start + 16]
        raw_outputs = runner.generate([
            ("audio", (row if args.stage == "screen" else row["source"])["audio_path"], prompt(row), False)
            for row in chunk
        ], 1800)
        for row, raw in zip(chunk, raw_outputs, strict=True):
            value, error = parse(raw)
            source = row if args.stage == "screen" else row["source"]
            result = {
                "id": row["id"], "dataset": source.get("dataset"), "source": source,
                "raw_model_output": raw, "parse_error": error,
                "provenance": {"protocol": config["protocol"], "model": generation["omni_model"], "stage": args.stage},
            }
            if args.stage == "screen":
                result["assessment"] = value
                result["language_gate"] = gates[str(row["id"])]
            else:
                result["screening_assessment"] = row.get("assessment")
                result["verification"] = value
            append_jsonl(output, result)
            print(json.dumps({"id": row["id"], "stage": args.stage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
