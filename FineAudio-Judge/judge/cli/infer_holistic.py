"""Run the paper Table 1 rubric-free holistic 0--100 protocol.

The manifest may contain one candidate per row (``sample_id``, ``prompt``,
``audio_path``), or preference rows containing ``preference_group_id``,
``prompt``, and a ``candidates`` array.  Repeated candidates are deduplicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from judge import HolisticJudge


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expand_jobs(rows: list[dict]) -> list[dict]:
    jobs: dict[str, dict] = {}
    for row in rows:
        if isinstance(row.get("candidates"), list):
            group_id = str(row["preference_group_id"])
            for candidate in row["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                score_id = f"{group_id}::{candidate_id}"
                job = {
                    "score_id": score_id,
                    "sample_id": score_id,
                    "preference_group_id": group_id,
                    "candidate_id": candidate_id,
                    "prompt": row["prompt"],
                    "audio_path": candidate["audio_path"],
                }
                if score_id in jobs and jobs[score_id] != job:
                    raise ValueError(f"conflicting repeated candidate: {score_id}")
                jobs[score_id] = job
        else:
            sample_id = str(row["sample_id"])
            jobs[sample_id] = {
                "score_id": sample_id,
                "sample_id": sample_id,
                "prompt": row["prompt"],
                "audio_path": row["audio_path"],
            }
    return list(jobs.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Qwen3-Omni base or merged model")
    parser.add_argument("--adapter", help="FineAudio-Judge LoRA; omit for a merged model")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--force-use-cache", action="store_true")
    args = parser.parse_args()

    jobs = expand_jobs(read_jsonl(args.manifest.resolve()))
    existing = read_jsonl(args.output.resolve())
    completed = {
        str(row["score_id"]) for row in existing
        if row.get("status") == "success" and isinstance(row.get("score"), int)
    }
    judge = HolisticJudge(
        args.model,
        adapter=args.adapter,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        device_map=args.device_map,
        force_use_cache=args.force_use_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for job in jobs:
            if job["score_id"] in completed:
                continue
            try:
                prediction = judge.predict(job["audio_path"], job["prompt"])
                row = {
                    "protocol": "holistic",
                    **job,
                    **prediction.as_dict(),
                    "status": "success" if prediction.score is not None else "unparsed",
                }
            except Exception as exc:
                row = {
                    "protocol": "holistic",
                    **job,
                    "score": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
