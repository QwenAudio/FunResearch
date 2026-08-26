#!/usr/bin/env python3
"""Generate data/manifest.json with file checksums and sample counts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_json_samples(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else 0


def count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_wav_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.wav"))


def file_entry(path: Path, sample_count: int | None = None) -> dict:
    entry = {
        "path": str(path.relative_to(path.parents[2] if "data" in path.parts else path.parent)),
        "size_bytes": path.stat().st_size,
        "md5": md5_file(path),
    }
    if sample_count is not None:
        entry["num_samples"] = sample_count
    return entry


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    version = (repo_root / "VERSION").read_text(encoding="utf-8").strip()

    manifest: dict = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "files": [],
        "audio": {},
    }

    processed_sets = {
        "data_msswift": data_root / "processed" / "data_msswift",
        "data_slu": data_root / "processed" / "data_slu",
    }

    for name, directory in processed_sets.items():
        split_info = {}
        if directory.exists():
            for json_file in sorted(directory.glob("*.json")):
                n = count_json_samples(json_file)
                split_info[json_file.stem] = n
                manifest["files"].append({
                    "dataset": name,
                    "split": json_file.stem,
                    "relative_path": str(json_file.relative_to(data_root)),
                    "size_bytes": json_file.stat().st_size,
                    "md5": md5_file(json_file),
                    "num_samples": n,
                })
        manifest["splits"][name] = split_info

    raw_root = data_root / "raw" / "final_data_1228"
    if raw_root.exists():
        raw_splits = {}
        for jsonl in sorted(raw_root.rglob("*.jsonl")):
            n = count_jsonl_lines(jsonl)
            rel = jsonl.relative_to(data_root)
            key = str(rel)
            raw_splits[key] = n
            manifest["files"].append({
                "dataset": "final_data_1228",
                "split": key,
                "relative_path": str(rel),
                "size_bytes": jsonl.stat().st_size,
                "md5": md5_file(jsonl),
                "num_samples": n,
            })
        manifest["splits"]["final_data_1228"] = raw_splits

    audio_root = data_root / "audio" / "final_data_audio"
    manifest["audio"] = {
        "relative_path": "audio/final_data_audio",
        "num_wav_files": count_wav_files(audio_root),
    }

    out_path = data_root / "manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote manifest to {out_path}")


if __name__ == "__main__":
    main()
