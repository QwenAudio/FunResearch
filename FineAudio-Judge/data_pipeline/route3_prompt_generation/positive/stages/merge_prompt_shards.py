from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from deduplicate import LexicalDedupIndex, normalize_text


ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Path3 prompt shards without modifying their source files."
    )
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--sequence-threshold", type=float, default=0.9)
    parser.add_argument("--jaccard-threshold", type=float, default=0.78)
    parser.add_argument("--expected-total", type=int, default=2000)
    parser.add_argument("--target-chinese", type=int, default=1000)
    parser.add_argument("--target-english", type=int, default=1000)
    parser.add_argument(
        "--source-stage",
        choices=("accepted", "final"),
        default="final",
        help="Merge pre-finalization accepted prompts or finalized prompts.",
    )
    parser.add_argument(
        "--allow-missing-source",
        action="store_true",
        help="For an in-progress diagnostic merge only, treat not-yet-created "
             "source files as empty. Final completion remains strict.",
    )
    args = parser.parse_args()

    unique: list[dict] = []
    duplicate_rows: list[dict] = []
    indexes_by_language: dict[str, LexicalDedupIndex] = {}
    unique_indexes_by_language: dict[str, list[int]] = {}
    normalized_seen: dict[tuple[str, str], int] = {}
    source_counts: Counter[str] = Counter()
    missing_source_run_ids: list[str] = []

    for run_id in args.run_ids:
        source_name = (
            "final_prompts.jsonl"
            if args.source_stage == "final"
            else "accepted_prompts.jsonl"
        )
        source_path = ROOT / "runs" / run_id / "01_prompt_generation" / source_name
        if not source_path.is_file():
            if args.allow_missing_source:
                missing_source_run_ids.append(run_id)
                source_counts[run_id] = 0
                continue
            raise FileNotFoundError(
                f"Required {args.source_stage} prompt file is missing: {source_path}"
            )
        rows = read_jsonl(source_path)
        source_counts[run_id] = len(rows)
        for source_row, row in enumerate(rows):
            prompt = str(row.get("prompt", "")).strip()
            language = str(row.get("language", "unknown"))
            if not prompt:
                duplicate_rows.append(
                    {
                        "source_run_id": run_id,
                        "source_row": source_row,
                        "source_id": row.get("id"),
                        "language": language,
                        "reason": "empty_prompt",
                    }
                )
                continue

            normalized = normalize_text(prompt)
            exact_key = (language, normalized)
            if exact_key in normalized_seen:
                duplicate_rows.append(
                    {
                        "source_run_id": run_id,
                        "source_row": source_row,
                        "source_id": row.get("id"),
                        "language": language,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "reason": "exact_normalized_duplicate",
                        "duplicate_of_merged_index": normalized_seen[exact_key],
                    }
                )
                continue

            language_index = indexes_by_language.setdefault(
                language, LexicalDedupIndex()
            )
            is_duplicate, local_index, scores = language_index.find_duplicate(
                prompt,
                args.sequence_threshold,
                args.jaccard_threshold,
            )
            if is_duplicate and local_index is not None:
                merged_index = unique_indexes_by_language[language][local_index]
                duplicate_rows.append(
                    {
                        "source_run_id": run_id,
                        "source_row": source_row,
                        "source_id": row.get("id"),
                        "language": language,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "reason": "lexical_duplicate",
                        "duplicate_of_merged_index": merged_index,
                        "scores": scores,
                    }
                )
                continue

            merged_index = len(unique)
            merged = dict(row)
            merged["id"] = f"path3_merged_{merged_index:06d}"
            merged["source_shard"] = {
                "run_id": run_id,
                "row": source_row,
                "id": row.get("id"),
            }
            merged["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            unique.append(merged)
            normalized_seen[exact_key] = merged_index
            language_index.add(prompt)
            unique_indexes_by_language.setdefault(language, []).append(merged_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "merged_unique.jsonl", unique)
    write_jsonl(args.output_dir / "duplicate_report.jsonl", duplicate_rows)
    language_counts = Counter(str(row.get("language", "unknown")) for row in unique)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_counts": dict(source_counts),
        "source_total": sum(source_counts.values()),
        "unique_total": len(unique),
        "duplicates_removed": len(duplicate_rows),
        "languages": dict(language_counts),
        "language_backfill_required": {
            "zh": max(0, args.target_chinese - language_counts.get("zh", 0)),
            "en": max(0, args.target_english - language_counts.get("en", 0)),
        },
        "expected_total": args.expected_total,
        "source_stage": args.source_stage,
        "backfill_required": max(0, args.expected_total - len(unique)),
        "sequence_threshold": args.sequence_threshold,
        "jaccard_threshold": args.jaccard_threshold,
        "source_files_unchanged": True,
        "missing_source_run_ids": missing_source_run_ids,
        "diagnostic_allow_missing_source": args.allow_missing_source,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
