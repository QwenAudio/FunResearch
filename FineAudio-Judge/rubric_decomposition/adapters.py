from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


def _manifest_row(*, sample_id: str, route: str, variant: str, prompt: str,
                  audio_path: str, source_path: str, source_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "route": route,
        "variant": variant,
        "prompt": prompt.strip(),
        "audio_path": audio_path,
        "source": {"path": source_path, "source_id": source_id},
    }


def load_source(spec: dict[str, Any]) -> list[dict[str, Any]]:
    route, variant = spec["route"], spec["variant"]
    rows: list[dict[str, Any]] = []
    if spec["adapter"] == "jsonl":
        path = Path(spec["path"])
        for raw in read_jsonl(path):
            source_id = str(raw[spec["id_field"]])
            prompt = raw.get(spec["prompt_field"])
            audio = raw.get(spec["audio_field"])
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(audio, str):
                continue
            rows.append(_manifest_row(
                sample_id=f"{variant}:{source_id}", route=route, variant=variant,
                prompt=prompt, audio_path=audio, source_path=str(path), source_id=source_id,
            ))
    elif spec["adapter"] == "oscar_glob":
        for name in sorted(glob.glob(spec["glob"])):
            path = Path(name)
            raw = json.loads(path.read_text(encoding="utf-8"))
            prompt = raw.get(spec["prompt_field"])
            audio = path.parent / spec["audio_relative_path"]
            if not isinstance(prompt, str) or not prompt.strip() or not audio.exists():
                continue
            source_id = path.parent.name
            rows.append(_manifest_row(
                sample_id=f"{variant}:{source_id}", route=route, variant=variant,
                prompt=prompt, audio_path=str(audio), source_path=str(path), source_id=source_id,
            ))
    else:
        raise ValueError(f"Unsupported adapter: {spec['adapter']}")
    return rows


def build_manifest(config: Path) -> list[dict[str, Any]]:
    data = json.loads(config.read_text(encoding="utf-8"))
    rows = [row for spec in data["sources"] for row in load_source(spec)]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in manifest")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_manifest(args.config)
    write_jsonl(args.output, rows)
    print(json.dumps({"output": str(args.output), "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
