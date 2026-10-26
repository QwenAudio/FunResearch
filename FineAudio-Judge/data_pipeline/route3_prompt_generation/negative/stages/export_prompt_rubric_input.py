from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def export(pairs_path: Path, output_path: Path) -> list[dict[str, Any]]:
    pairs = read_jsonl(pairs_path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, pair in enumerate(pairs, 1):
        sample_id = str(pair.get("sample_id") or "")
        prompt = str(pair.get("prompt_prime") or pair.get("evaluation_prompt") or "").strip()
        if not sample_id or not prompt:
            raise ValueError(f"{pairs_path}:{index}: missing sample_id or P-prime prompt")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append({
            "sample_id": sample_id,
            "route": "route3_prompt_side_negative",
            "language": pair.get("language"),
            "prompt": prompt,
        })
    if not rows:
        raise ValueError("prompt-side pair manifest is empty")
    write_jsonl(output_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Route 3 P-prime prompts for canonical rubric decomposition."
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = export(args.pairs, args.output)
    print(json.dumps({"samples": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
