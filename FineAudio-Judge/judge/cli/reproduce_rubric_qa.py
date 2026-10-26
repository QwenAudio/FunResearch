#!/usr/bin/env python3
"""Reproduce FineAudio-Judge single-question QA inference.

This entry point reads ``rubric_qa/rubric_qa.jsonl`` directly from the public
benchmark.  Each audio-question pair is a separate semantic request; model
batching only improves throughput.  The default decoding settings match the
checkpoint-4000 paper run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


from judge.benchmark import load_rubric_qa, read_jsonl, rubric_qa_report
from judge.judge import SingleRubricJudge


BASE_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def output_row(
    item: dict[str, Any],
    question: dict[str, Any],
    *,
    prediction: Any | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    row = {
        "protocol": "single_rubric",
        "sample_id": item["sample_id"],
        "subset": item["subset"],
        "rubric_set_id": item["rubric_set_id"],
        "language": item["language"],
        "audio": item["audio"],
        "rubric_id": question["rubric_id"],
        "subtype": question.get("subtype"),
        "axis": question.get("axis"),
        "question": question["question"],
        "reference_label": question["reference_label"],
        "vote_pattern": question.get("vote_pattern"),
    }
    if error is not None:
        return {**row, "pred_answer": None, "status": "error", "error": f"{type(error).__name__}: {error}"}
    answer = prediction.answer
    return {
        **row,
        "pred_answer": answer,
        "correct": answer == question["reference_label"],
        "parse_mode": prediction.parse_mode,
        "raw_response": prediction.raw_response,
        "status": "success" if answer in {"yes", "no"} else "unparsed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", required=True, help="checkpoint-4000 LoRA path or Hub ID")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--max-samples", type=int, default=0, help="0 runs all 700 samples")
    parser.add_argument("--max-questions", type=int, default=0, help="0 runs every question")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    items = load_rubric_qa(args.dataset_root)
    if args.max_samples:
        items = items[: args.max_samples]
    existing = read_jsonl(args.output) if args.output.is_file() else []
    for row in existing:
        if row.get("status") != "success":
            continue
        if row.get("model") not in {None, args.model} or row.get("adapter") not in {None, args.adapter}:
            raise ValueError("output contains successful rows from a different model or adapter")
    completed = {
        (str(row.get("sample_id")), str(row.get("rubric_id")))
        for row in existing
        if row.get("status") == "success" and row.get("pred_answer") in {"yes", "no"}
    }
    judge = SingleRubricJudge(
        args.model,
        adapter=args.adapter,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.model_batch_size,
        seed=args.seed,
        guided_yes_no=True,
        enforce_eager=args.enforce_eager,
    )

    remaining = args.max_questions or None
    for item in items:
        pending = [
            row for row in item["questions"]
            if (item["sample_id"], row["rubric_id"]) not in completed
        ]
        if remaining is not None:
            pending = pending[:remaining]
        if not pending:
            continue
        try:
            predictions = judge.predict(item["audio_path"], pending)
            rows = [
                output_row(item, question, prediction=prediction)
                for question, prediction in zip(pending, predictions)
            ]
        except Exception as batch_error:
            rows = []
            for question in pending:
                try:
                    prediction = judge.predict(item["audio_path"], [question])[0]
                    rows.append(output_row(item, question, prediction=prediction))
                except Exception as error:
                    row = output_row(item, question, error=error)
                    row["batch_error"] = f"{type(batch_error).__name__}: {batch_error}"
                    rows.append(row)
        for row in rows:
            row.update({
                "model": args.model,
                "adapter": args.adapter,
                "seed": args.seed,
                "model_batch_size": args.model_batch_size,
                "guided_yes_no": True,
            })
        append_jsonl(args.output, rows)
        completed.update(
            (str(row["sample_id"]), str(row["rubric_id"]))
            for row in rows if row.get("status") == "success"
        )
        if remaining is not None:
            remaining -= len(pending)
            if remaining <= 0:
                break

    all_rows = read_jsonl(args.output)
    report = rubric_qa_report(all_rows)
    report_path = args.report or args.output.with_name(args.output.stem + ".metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
