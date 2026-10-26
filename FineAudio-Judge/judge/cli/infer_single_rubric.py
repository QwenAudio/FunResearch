"""Run the paper Table 2/3 single-rubric protocol over a JSONL manifest.

Input rows contain ``sample_id``, ``audio_path``, and ``questions``.  Every
question becomes one semantic model request; ``--model-batch-size`` only
controls throughput. Output contains one flat JSONL row per rubric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from judge import SingleRubricJudge


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Qwen3-Omni base or merged model")
    parser.add_argument("--adapter", help="FineAudio-Judge LoRA; omit for a merged model")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--no-guided-yes-no", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    source = read_jsonl(args.manifest.resolve())
    existing = read_jsonl(args.output.resolve())
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
        guided_yes_no=not args.no_guided_yes_no,
        enforce_eager=args.enforce_eager,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for item in source:
            sample_id = str(item["sample_id"])
            pending = [
                row for row in item["questions"]
                if (sample_id, str(row.get("rubric_id") or row.get("id"))) not in completed
            ]
            if not pending:
                continue
            try:
                predictions = judge.predict(item["audio_path"], pending)
                output_rows = []
                for question, prediction in zip(pending, predictions):
                    output_rows.append({
                        "protocol": "single_rubric",
                        "sample_id": sample_id,
                        "dataset": item.get("dataset"),
                        "audio_path": item["audio_path"],
                        "rubric_id": prediction.rubric_id,
                        "subtype": question.get("subtype"),
                        "axis": question.get("axis"),
                        "question": prediction.question,
                        "gold_answer": question.get("gold_answer", question.get("expected_answer")),
                        "pred_answer": prediction.answer,
                        "parse_mode": prediction.parse_mode,
                        "raw_response": prediction.raw_response,
                        "status": "success" if prediction.answer in {"yes", "no"} else "unparsed",
                    })
            except Exception as exc:
                output_rows = [{
                    "protocol": "single_rubric",
                    "sample_id": sample_id,
                    "dataset": item.get("dataset"),
                    "audio_path": item.get("audio_path"),
                    "rubric_id": str(row.get("rubric_id") or row.get("id")),
                    "question": row.get("question"),
                    "gold_answer": row.get("gold_answer", row.get("expected_answer")),
                    "pred_answer": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                } for row in pending]
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
