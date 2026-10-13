#!/usr/bin/env python3
"""Wrap a validated Nano training JSONL as a selection manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def has_reference(row: dict) -> bool:
    if str(row.get("text") or row.get("label") or "").strip():
        return True
    return any(
        item.get("role") == "assistant" and str(item.get("content") or "").strip()
        for item in row.get("messages") or []
    )


def has_audio(row: dict) -> bool:
    if row.get("raw_audio_path") or row.get("audio_path"):
        return True
    return any(
        "<|startofspeech|>!" in str(item.get("content") or "")
        for item in row.get("messages") or []
    )


def main() -> int:
    args = parse_args()
    source = args.train_jsonl.resolve(strict=True)
    selected_rows = 0
    selected_seconds = 0.0
    rows_without_duration = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at line {line_number}")
            if not has_reference(row):
                raise ValueError(f"missing transcript at line {line_number}")
            if not has_audio(row):
                raise ValueError(f"missing audio path at line {line_number}")
            if not row.get("asr_hotwords"):
                raise ValueError(f"missing asr_hotwords at line {line_number}")
            duration = row.get("duration") or row.get("raw_audio_duration")
            if duration is None:
                rows_without_duration += 1
            else:
                seconds = float(duration)
                if seconds < 0:
                    raise ValueError(f"negative duration at line {line_number}")
                selected_seconds += seconds
            selected_rows += 1
    if selected_rows == 0:
        raise ValueError("training JSONL has no rows")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, output.parent.resolve())
    payload = {
        "selected_rows": selected_rows,
        "selected_seconds": selected_seconds,
        "selected_hours": selected_seconds / 3600,
        "rows_without_duration": rows_without_duration,
        "shards": [{
            "path": relative_source,
            "start_row": 0,
            "selected_rows": selected_rows,
            "selected_seconds": selected_seconds,
            "selected_hours": selected_seconds / 3600,
        }],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}: {selected_rows} rows, {selected_seconds / 3600:.3f} h")
    if rows_without_duration:
        print(f"warning: {rows_without_duration} rows have no duration; hours are incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
