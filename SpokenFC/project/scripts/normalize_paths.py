#!/usr/bin/env python3
"""Normalize audio paths in processed JSON datasets to repo-relative paths."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


AUDIO_REL_PREFIX = "audio/final_data_audio"


def _extract_audio_relpath(abs_path: str) -> str | None:
    """Extract relative path under final_data_audio from various absolute paths."""
    normalized = abs_path.replace("\\", "/")
    markers = [
        "/final_data_audio/",
        "final_data_audio/",
    ]
    for marker in markers:
        idx = normalized.find(marker)
        if idx != -1:
            suffix = normalized[idx + len(marker):]
            return f"{AUDIO_REL_PREFIX}/{suffix.lstrip('/')}"
    return None


def _normalize_audios(audios: Any, repo_root: Path, dry_run: bool) -> tuple[Any, int]:
    if not isinstance(audios, list):
        return audios, 0

    changed = 0
    new_audios = []
    for item in audios:
        if not isinstance(item, str):
            new_audios.append(item)
            continue

        rel = _extract_audio_relpath(item)
        if rel is None:
            new_audios.append(item)
            continue

        abs_target = repo_root / "data" / rel
        new_path = rel
        if abs_target.exists():
            new_path = rel
        elif (repo_root / "data" / "audio" / "final_data_audio" / rel.split("final_data_audio/", 1)[-1]).exists():
            new_path = rel

        if new_path != item:
            changed += 1
        new_audios.append(new_path)

    return new_audios, changed


def _normalize_item(item: dict[str, Any], repo_root: Path) -> int:
    changed = 0
    if "audios" in item:
        item["audios"], n = _normalize_audios(item["audios"], repo_root, dry_run=False)
        changed += n
    return changed


def normalize_json_file(path: Path, repo_root: Path, dry_run: bool) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")

    total_changed = 0
    for item in data:
        if isinstance(item, dict):
            total_changed += _normalize_item(item, repo_root)

    if total_changed and not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2 if path.suffix == ".json" else None)
            if path.suffix == ".json":
                f.write("\n")

    return total_changed


def iter_json_files(data_root: Path) -> list[Path]:
    files: list[Path] = []
    for sub in ("processed/data_msswift", "processed/data_slu"):
        base = data_root / sub
        if base.exists():
            files.extend(sorted(base.glob("*.json")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize audio paths in dataset JSON files.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report changes without writing files.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = repo_root / "data"
    json_files = iter_json_files(data_root)

    if not json_files:
        print(f"No JSON files found under {data_root}/processed")
        return

    grand_total = 0
    for path in json_files:
        if args.dry_run:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            changed = 0
            for item in data:
                if isinstance(item, dict) and "audios" in item:
                    _, n = _normalize_audios(item["audios"], repo_root, dry_run=True)
                    changed += n
        else:
            changed = normalize_json_file(path, repo_root, dry_run=False)

        print(f"{'[dry-run] ' if args.dry_run else ''}{path.name}: {changed} audio path(s) updated")
        grand_total += changed

    print(f"Total updated paths: {grand_total}")


if __name__ == "__main__":
    main()
