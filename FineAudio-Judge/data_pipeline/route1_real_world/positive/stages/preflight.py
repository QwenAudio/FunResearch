#!/usr/bin/env python3
"""Validate Route 1 inputs without loading GPU models."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_NORMALIZED_FIELDS = ("id", "dataset", "duration", "audio_path")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def datasets(value: str, root: Path) -> list[str]:
    if value in {"all", "both"}:
        return [path.parent.name for path in sorted((root / "data").glob("*/source_manifest.jsonl"))]
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    minimum = float(config["selection"]["duration_seconds"]["min"])
    maximum = float(config["selection"]["duration_seconds"]["max"])
    report: dict[str, Any] = {
        "protocol": config["protocol"],
        "config": str(args.config.resolve()),
        "run_root": str(args.run_root.resolve()),
        "datasets": {},
        "dependencies": {},
        "errors": [],
        "warnings": [],
        "validation_policy": {
            "id_scope": "unique within each dataset",
            "normalized_manifest_required_fields": list(REQUIRED_NORMALIZED_FIELDS),
            "duration_seconds": {"min": minimum, "max": maximum},
            "duration_bounds_are_configurable": True,
        },
    }

    for module in ("whisper", "speechbrain", "torch", "funasr", "modelscope", "librosa", "soundfile", "vllm"):
        report["dependencies"][module] = importlib.util.find_spec(module) is not None
    if importlib.util.find_spec("demucs") is None:
        report["warnings"].append("demucs is not installed in this Python environment; configure demucs_python")

    for dataset in datasets(args.dataset, args.run_root):
        path = args.run_root / "data" / dataset / "source_manifest.jsonl"
        if not path.is_file():
            report["errors"].append(f"missing manifest: {path}")
            continue
        rows = read_jsonl(path)
        ids = [str(row.get("id")) for row in rows]
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        stats = {
            "rows": len(rows),
            "duplicate_ids_within_dataset": duplicates,
            "missing_media": [],
            "invalid_rows": [],
        }
        for index, row in enumerate(rows, start=1):
            missing = [
                key for key in REQUIRED_NORMALIZED_FIELDS if row.get(key) is None
            ]
            duration = float(row.get("duration") or 0)
            if missing or not minimum <= duration <= maximum:
                stats["invalid_rows"].append({
                    "line": index,
                    "id": row.get("id"),
                    "missing": missing,
                    "duration": duration,
                    "duration_outside_configured_range": not minimum <= duration <= maximum,
                })
            for key in ("audio_path", "video_path"):
                value = row.get(key)
                if value and not Path(value).is_file():
                    stats["missing_media"].append({"id": row.get("id"), "field": key, "path": value})
        if duplicates or stats["invalid_rows"] or stats["missing_media"]:
            report["errors"].append(f"{dataset}: manifest/media validation failed")
        report["datasets"][dataset] = stats

    output = args.run_root / "reports" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
