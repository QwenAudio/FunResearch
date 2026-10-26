#!/usr/bin/env python3
"""Reproduce FineAudio-Judge batch-rubric preference evaluation.

The script joins the public candidate groups with their prompt rubrics, scores
all 300 candidates with deterministic chunks of at most 80 questions, and
reports the frozen 279-pair and 84-group paper metrics.  Exact score ties count
as incorrect, matching the paper evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


from judge.benchmark import (
    load_preference_candidates,
    preference_report,
    read_jsonl,
)
from judge.judge import BatchRubricJudge


BASE_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", required=True, help="checkpoint-4000 LoRA path or Hub ID")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 runs all 300 candidates")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    jobs = load_preference_candidates(args.dataset_root)
    if args.max_candidates:
        jobs = jobs[: args.max_candidates]
    existing = read_jsonl(args.output) if args.output.is_file() else []
    for row in existing:
        if row.get("status") != "success":
            continue
        if row.get("model") not in {None, args.model} or row.get("adapter") not in {None, args.adapter}:
            raise ValueError("output contains successful rows from a different model or adapter")
    completed = {
        (str(row.get("group_id")), str(row.get("candidate_id")))
        for row in existing if row.get("status") == "success"
    }
    judge = BatchRubricJudge(
        args.model,
        adapter=args.adapter,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.model_batch_size,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        temperature=0.0,
        seed=args.seed,
        ordering_backend="qwen",
        guided_json_fallback=True,
        enforce_eager=args.enforce_eager,
    )

    for job in jobs:
        key = (job["group_id"], job["candidate_id"])
        if key in completed:
            continue
        try:
            result = judge.predict(job["audio_path"], job["questions"], sample_id=job["sample_id"])
            predictions = [prediction.as_dict() for prediction in result.predictions]
            metadata = {str(row["rubric_id"]): row for row in job["questions"]}
            for prediction in predictions:
                source = metadata[prediction["rubric_id"]]
                prediction["axis"] = source.get("axis")
                prediction["subtype"] = source.get("subtype")
            yes_count = sum(row["answer"] == "yes" for row in predictions)
            no_count = sum(row["answer"] == "no" for row in predictions)
            rubric_count = len(predictions)
            if yes_count + no_count != rubric_count:
                raise RuntimeError("batch inference returned an unparsed rubric answer")
            row = {
                "protocol": "joint_rubric",
                "model": args.model,
                "adapter": args.adapter,
                "seed": args.seed,
                "temperature": 0.0,
                "ordering_backend": "qwen",
                "chunk_size": args.chunk_size,
                "guided_json_fallback": True,
                "sample_id": job["sample_id"],
                "group_id": job["group_id"],
                "candidate_id": job["candidate_id"],
                "language": job["language"],
                "category": job["category"],
                "audio": job["audio"],
                "status": "success",
                "yes_count": yes_count,
                "no_count": no_count,
                "rubric_count": rubric_count,
                "score": yes_count / rubric_count,
                "predictions": predictions,
                "chunks": [chunk.as_dict() for chunk in result.chunks],
            }
        except Exception as error:
            row = {
                "protocol": "joint_rubric",
                "model": args.model,
                "adapter": args.adapter,
                "seed": args.seed,
                "temperature": 0.0,
                "ordering_backend": "qwen",
                "chunk_size": args.chunk_size,
                "guided_json_fallback": True,
                "sample_id": job["sample_id"],
                "group_id": job["group_id"],
                "candidate_id": job["candidate_id"],
                "audio": job["audio"],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        append_jsonl(args.output, row)
        if row["status"] == "success":
            completed.add(key)

    all_rows = read_jsonl(args.output)
    report = preference_report(args.dataset_root, all_rows)
    report_path = args.report or args.output.with_name(args.output.stem + ".metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_scores": report["candidate_scores"],
        "pairwise": report["pairwise"]["overall"],
        "ranking": report["ranking"]["overall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
