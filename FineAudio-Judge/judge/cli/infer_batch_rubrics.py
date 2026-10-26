"""Run the paper-default joint-rubric protocol over a JSONL manifest.

Each input row contains ``sample_id``, ``audio_path``, and ``questions``.  One
model request contains at most 80 deterministically ordered questions.  Output
is resumable by sample ID and includes both flattened predictions and raw chunk
responses for auditability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from judge import BatchRubricJudge, aggregate_rubric_scores


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
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--ordering-backend", choices=("qwen", "gemini"), default="qwen")
    parser.add_argument("--no-guided-json-fallback", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    source = read_jsonl(args.manifest.resolve())
    existing = read_jsonl(args.output.resolve())
    completed = {str(row["sample_id"]) for row in existing if row.get("status") == "success"}
    judge = BatchRubricJudge(
        args.model,
        adapter=args.adapter,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.model_batch_size,
        chunk_size=args.chunk_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        ordering_backend=args.ordering_backend,
        guided_json_fallback=not args.no_guided_json_fallback,
        enforce_eager=args.enforce_eager,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for item in source:
            sample_id = str(item["sample_id"])
            if sample_id in completed:
                continue
            try:
                result = judge.predict(
                    item["audio_path"], item["questions"], sample_id=sample_id
                )
                predictions = [row.as_dict() for row in result.predictions]
                metadata = {
                    str(row.get("rubric_id") or row.get("id")): row
                    for row in item["questions"]
                }
                for row in predictions:
                    source_rubric = metadata[row["rubric_id"]]
                    row["axis"] = source_rubric.get("axis")
                    row["subtype"] = source_rubric.get("subtype")
                output = {
                    "protocol": "joint_rubric",
                    "sample_id": sample_id,
                    "dataset": item.get("dataset"),
                    "audio_path": item["audio_path"],
                    "status": "success",
                    "predictions": predictions,
                    "chunks": [row.as_dict() for row in result.chunks],
                    "summary": aggregate_rubric_scores(predictions, missing="error"),
                }
            except Exception as exc:
                output = {
                    "protocol": "joint_rubric",
                    "sample_id": sample_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
