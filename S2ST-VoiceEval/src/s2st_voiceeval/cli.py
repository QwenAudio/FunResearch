"""Small JSON command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Score source-reference voice preservation.")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("score", "rank", "batch"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, help="Export directory or Hugging Face repo ID")
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
        command.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
        command.add_argument("--revision", help="Hub revision; a commit hash pins the exact artifact")
        command.add_argument("--local-files-only", action="store_true")
        if name == "batch":
            command.add_argument("--input", type=Path, required=True, help="JSONL input manifest")
            command.add_argument("--output", type=Path, required=True, help="New JSONL output file")
        else:
            command.add_argument("--reference", required=True)
            command.add_argument("--candidate", nargs="+" if name == "rank" else None, required=True)
    metric = sub.add_parser("pairacc", help="Compute accuracy from labeled score pairs; no model needed")
    metric.add_argument("--input", type=Path, required=True)
    return root


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON") from error
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                yield number, row


def batch(evaluator, input_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile("w", dir=output_path.parent, encoding="utf-8",
                                         delete=False, suffix=".jsonl.tmp") as output:
            temp_name = output.name
            for number, row in read_jsonl(input_path):
                reference, candidates = row.get("reference"), row.get("candidates")
                if not isinstance(reference, str) or not isinstance(candidates, list) or not candidates:
                    raise ValueError(f"{input_path}:{number}: require reference and nonempty candidates")
                if not all(isinstance(item, str) for item in candidates):
                    raise ValueError(f"{input_path}:{number}: candidates must be paths")
                def resolve(value):
                    path = Path(value).expanduser()
                    return path if path.is_absolute() else input_path.parent / path
                ranked = evaluator.rank(resolve(reference), [resolve(item) for item in candidates])
                for item in ranked:
                    item["candidate"] = candidates[item["index"]]
                result = {"id": row.get("id", number), "reference": reference,
                          "ranking": ranked, "precision": evaluator.precision}
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
            if not count:
                raise ValueError("The input manifest contains no records")
        os.link(temp_name, output_path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    print(f"Wrote {count} records to {output_path}", file=sys.stderr)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "pairacc":
            from .metrics import pairwise_accuracy
            result = pairwise_accuracy(row for _, row in read_jsonl(args.input))
        else:
            from .evaluator import VoiceEvaluator
            evaluator = VoiceEvaluator(args.model, device=args.device, precision=args.precision,
                                       revision=args.revision, local_files_only=args.local_files_only)
            if args.command == "batch":
                batch(evaluator, args.input, args.output)
                return 0
            if args.command == "score":
                result = {"reference": args.reference, "candidate": args.candidate,
                          "score": evaluator.score(args.reference, args.candidate),
                          "precision": evaluator.precision}
            else:
                result = {"reference": args.reference,
                          "ranking": evaluator.rank(args.reference, args.candidate),
                          "precision": evaluator.precision}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"s2st-voiceeval: {error}", file=sys.stderr)
        return 2
