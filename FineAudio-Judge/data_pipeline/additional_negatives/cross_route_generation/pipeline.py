"""Generate Route 1/2 prompts with the Route 3 Audio-Oscar process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ROUTE3_RUNNER = ROOT.parents[1] / "route3_prompt_generation/positive/stages/run_audio_oscar.py"
FROZEN_SEED = int.from_bytes(bytes.fromhex("0135282a"), "big")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_id(route: str, source_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("_.-")
    suffix = sha256_text(f"{route}:{source_id}")[:10]
    return f"crossgen_{route}_{clean[:64] or 'sample'}_{suffix}"


def normalize_source(row: dict[str, Any], route: str, index: int, seed: int) -> dict[str, Any]:
    if route == "route1":
        source_id = str(row.get("id") or row.get("sample_id") or "").strip()
        prompt = str(row.get("audio_caption") or row.get("caption") or row.get("prompt") or "").strip()
        language = row.get("primary_caption_language") or row.get("speech_language") or row.get("language")
        duration = row.get("audio_duration") or row.get("audio_duration_seconds")
    else:
        source_id = str(row.get("id") or row.get("sample_id") or "").strip()
        prompt = str(row.get("caption") or row.get("prompt") or "").strip()
        language = row.get("target_caption_language") or row.get("caption_language") or row.get("language")
        duration = row.get("duration") or row.get("audio_duration_seconds")
    if not source_id or not prompt:
        raise ValueError(f"{route} source row {index} lacks an ID or prompt")
    return {
        "id": sample_id(route, source_id),
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "prompt_source_route": route,
        "source_sample_id": source_id,
        "source_row_number": index,
        "source_audio_path": row.get("audio_path"),
        "source_audio_duration_seconds": duration,
        "language": language,
        "audio_generation_route": "route3_prompt_generation",
        "consistency_label": "positive",
        "selection_seed": seed,
        "prompt_characters": len(prompt),
    }


def prepare(args: argparse.Namespace) -> None:
    route1 = read_jsonl(args.route1_source)
    route2 = read_jsonl(args.route2_source)
    if len(route1) != args.expected_source_rows or len(route2) != args.expected_source_rows:
        raise ValueError(
            f"expected {args.expected_source_rows} rows per source, got {len(route1)} and {len(route2)}"
        )
    rng = random.Random(args.seed)
    selected = []
    for route, rows in (("route1", route1), ("route2", route2)):
        indexes = sorted(rng.sample(range(len(rows)), args.per_source))
        selected.extend(normalize_source(rows[index], route, index + 1, args.seed) for index in indexes)
    queues = {
        route: sorted(
            (row for row in selected if row["prompt_source_route"] == route),
            key=lambda row: (-row["prompt_characters"], row["id"]),
        )
        for route in ("route1", "route2")
    }
    ordered = []
    for index in range(args.per_source):
        ordered.extend((queues["route1"][index], queues["route2"][index]))
    for order, row in enumerate(ordered, 1):
        row["generation_order"] = order
    write_jsonl(args.output_root / "manifests/selected_prompts.jsonl", ordered)
    report = {
        "valid": len(ordered) == args.per_source * 2,
        "seed": args.seed,
        "total": len(ordered),
        "distribution": dict(Counter(row["prompt_source_route"] for row in ordered)),
        "source_sha256": {
            "route1": sha256_file(args.route1_source),
            "route2": sha256_file(args.route2_source),
        },
    }
    (args.output_root / "manifests/selection_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def generate(args: argparse.Namespace) -> None:
    manifest = args.output_root / "manifests/selected_prompts.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"run prepare first: {manifest}")
    environment = os.environ.copy()
    environment["AUDIO_OSCAR_RUN_FINAL_EDIT"] = "false"
    subprocess.run([
        sys.executable, str(ROUTE3_RUNNER), "--prompts", str(manifest),
        "--output-root", str(args.output_root / "audio_oscar"), "--skip-complete",
        "--max-sample-attempts", str(args.max_sample_attempts),
    ], check=True, env=environment)


def finalize(args: argparse.Namespace) -> None:
    selected = read_jsonl(args.output_root / "manifests/selected_prompts.jsonl")
    generated = []
    missing = []
    for row in selected:
        sample_root = args.output_root / "audio_oscar/samples" / row["id"]
        try:
            summary = json.loads((sample_root / "run_summary.json").read_text(encoding="utf-8"))
            audio = Path(str(summary["final_audio"]))
            if not (sample_root / "COMPLETE").is_file() or not audio.is_file():
                raise FileNotFoundError
            generated.append({
                **row,
                "audio_path": str(audio.resolve()),
                "audio_sha256": sha256_file(audio),
                "audio_duration_seconds": summary.get("final_duration_seconds"),
                "generation_status": "complete",
                "run_summary_path": str((sample_root / "run_summary.json").resolve()),
            })
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            missing.append(row["id"])
    if missing and not args.allow_incomplete:
        raise ValueError(f"{len(missing)} generated samples are incomplete")
    output = args.output_root / "manifests/generated_pairs.jsonl"
    write_jsonl(output, generated)
    report = {
        "valid": not missing,
        "target": len(selected),
        "generated": len(generated),
        "missing": missing,
        "distribution": dict(Counter(row["prompt_source_route"] for row in generated)),
        "output": str(output.resolve()),
    }
    (args.output_root / "manifests/generation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--route1-source", type=Path, required=True)
    prep.add_argument("--route2-source", type=Path, required=True)
    prep.add_argument("--expected-source-rows", type=int, default=2000)
    prep.add_argument("--per-source", type=int, default=500)
    prep.add_argument("--seed", type=int, default=FROZEN_SEED)
    generation = sub.add_parser("generate")
    generation.add_argument("--max-sample-attempts", type=int, default=3)
    final = sub.add_parser("finalize")
    final.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "generate":
        generate(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
