#!/usr/bin/env python3
"""Run one audio file with the aligned Nano adapter and no reference text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--nano-model-code", type=Path,
        help="Path to model.py in the Fun-ASR repository when Nano is not registered.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--hotword", action="append", required=True)
    parser.add_argument("--history-context", default="")
    parser.add_argument("--candidate-cap", type=int, default=40)
    parser.add_argument("--base-topk-gate", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ctc-temporal-gate", action="store_true")
    parser.add_argument("--ctc-model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def infer(args: argparse.Namespace, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    audio = args.audio.resolve(strict=True)
    source = {
        "raw_audio_path": str(audio),
        "asr_hotwords": args.hotword,
        "hist_context": args.history_context,
    }
    (work_dir / "input.jsonl").write_text(
        json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = [{
        "name": "single_audio",
        "jsonl": "input.jsonl",
        "stages": [4 if args.history_context.strip() else 3],
        "hotword_field": "asr_hotwords",
    }]
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate.py")),
        "--model-dir", str(args.model_dir),
        "--checkpoint", str(args.checkpoint),
        "--manifest", str(work_dir / "manifest.json"),
        "--output-dir", str(work_dir),
        "--candidate-cap", str(args.candidate_cap),
        "--base-topk-gate", str(args.base_topk_gate),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if args.nano_model_code is not None:
        command.extend(["--nano-model-code", str(args.nano_model_code)])
    if args.ctc_temporal_gate:
        if args.ctc_model_dir is None:
            raise ValueError("--ctc-model-dir is required with --ctc-temporal-gate")
        command.extend([
            "--ctc-temporal-gate", "--ctc-model-dir", str(args.ctc_model_dir)
        ])
    subprocess.run(command, check=True)
    rows = (work_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"expected one prediction, found {len(rows)}")
    row = json.loads(rows[0])
    if row.get("error"):
        raise RuntimeError(row["error"])
    return {
        "audio": str(audio),
        "prompt_hotwords": row["prompt_hotwords"],
        "baseline_prediction": row["baseline_prediction"],
        "adapter_prediction": row["adapter_prediction"],
    }


def main() -> int:
    args = parse_args()
    if args.candidate_cap < 1 or args.base_topk_gate < 1:
        raise ValueError("candidate cap and Base Top-K gate must be positive")
    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="hotbias_nano_") as name:
            result = infer(args, Path(name))
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {args.output_dir}")
        result = infer(args, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
