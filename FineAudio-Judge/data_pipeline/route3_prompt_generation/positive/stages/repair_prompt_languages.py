from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

from client import ChatClient, SamplingConfig, prompt_from_response  # noqa: E402
from provenance import (  # noqa: E402
    atomic_write_jsonl,
    sha256_json,
    utc_now,
)


def prompt_from_repair_response(response: dict[str, Any]) -> str:
    """Accept the requested key and harmless translation-key variants."""

    for key in ("prompt", "translated_prompt", "translation", "output", "text"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    string_values = [value.strip() for value in response.values() if isinstance(value, str) and value.strip()]
    if len(string_values) == 1:
        return string_values[0]
    return prompt_from_response(response)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def language_ratio(text: str) -> float:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return cjk / max(1, cjk + latin)


def needs_repair(record: dict[str, Any]) -> bool:
    return record.get("language") == "zh" and language_ratio(str(record.get("prompt", ""))) < 0.5


def chinese_quotations(text: str) -> list[str]:
    spans: list[str] = []
    # Quotes must be real delimiters, not apostrophes inside English words.
    patterns = [
        r"“([^”\n]+)”",
        r'(?<![A-Za-z])"([^"\n]+)"(?![A-Za-z])',
        r"(?<![A-Za-z])'([^'\n]+)'(?![A-Za-z])",
    ]
    for pattern in patterns:
        for value in re.findall(pattern, text):
            cjk = len(re.findall(r"[\u3400-\u9fff]", value))
            latin = len(re.findall(r"[A-Za-z]", value))
            if cjk >= 2 and latin == 0:
                spans.append(value.strip())
    return list(dict.fromkeys(spans))


def chinese_text_only(text: str) -> str:
    """Normalize Chinese text for quote preservation while ignoring punctuation."""

    return "".join(re.findall(r"[\u3400-\u9fff]", text))


def stable_seed(record_id: str, attempt: int) -> int:
    digest = hashlib.sha256(f"{record_id}:{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


TRANSLATE_PROMPT = """Your output narration must be written in Chinese. Convert the following audio-generation prompt into coherent, natural Chinese.

This is a faithful language repair, not a rewrite of the scene. Preserve every audible event, speaker role, quoted utterance meaning, temporal order, overlap, repetition, sustained sound, and ending. Keep existing Chinese quoted speech verbatim. Translate English narration and any English quoted speech into Chinese. Do not add or remove events, explanations, timestamps, scene graphs, rubrics, or metadata. Keep the direct, moderately detailed Complex Audio Scenes style of the source.

The source was mislabeled as Chinese because it contains Chinese dialogue, but its narration may still be English. Translate all English narration. The returned prompt must be predominantly Chinese rather than English.

{retry_feedback}

Return JSON only:
{{"prompt": "..."}}

Source record:
{record_json}
"""


def repair_worker(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    targets = [
        record
        for index, record in enumerate(record for record in records if needs_repair(record))
        if index % args.worker_count == args.worker_index
    ]
    targets = [
        record
        for index, record in enumerate(targets)
        if index % args.sub_count == args.sub_index
    ]
    completed = {str(record["id"]): record for record in read_jsonl(args.output)}
    overrides = {
        str(record["id"]): str(record["prompt"])
        for record in read_jsonl(args.override)
        if str(record.get("id", "")) and str(record.get("prompt", "")).strip()
    } if args.override else {}
    client = ChatClient(args.api_base, args.model)
    sampling = SamplingConfig(temperature=0.1, top_p=0.8, max_tokens=4000)
    for source in targets:
        record_id = str(source["id"])
        if record_id in completed:
            continue
        source_prompt = str(source["prompt"])
        required_quotes = chinese_quotations(source_prompt)
        if record_id in overrides:
            repaired = overrides[record_id].strip()
            ratio = language_ratio(repaired)
            repaired_chinese = chinese_text_only(repaired)
            missing_quotes = [
                value
                for value in required_quotes
                if chinese_text_only(value) not in repaired_chinese
            ]
            if ratio < 0.5 or missing_quotes:
                raise ValueError(
                    f"Invalid manual override for {record_id}: ratio={ratio:.3f}, "
                    f"missing_quotes={missing_quotes}"
                )
            result = {
                **source,
                "prompt": repaired,
                "prompt_sha256": sha256_json(repaired),
                "language_repair": {
                    "created_at": utc_now(),
                    "model": "manual_faithful_translation_override",
                    "source_prompt_sha256": sha256_json(source_prompt),
                    "source_script_ratio": language_ratio(source_prompt),
                    "repaired_script_ratio": ratio,
                    "preserved_chinese_quotations": required_quotes,
                },
            }
            completed[record_id] = result
            atomic_write_jsonl(args.output, [completed[key] for key in sorted(completed)])
            print(f"repaired {record_id} by override ({len(completed)}/{len(targets)})", flush=True)
            continue
        last_error = ""
        for attempt in range(1, 9):
            try:
                request = TRANSLATE_PROMPT.format(
                    retry_feedback=(
                        ""
                        if attempt == 1
                        else "The previous response failed validation. Ensure all narration is Chinese and preserve the scene exactly."
                    ),
                    record_json=json.dumps(
                        {
                            "id": record_id,
                            "target_language": "Chinese",
                            "prompt": source_prompt,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                response = client.json_completion(
                    request,
                    sampling,
                    seed=stable_seed(record_id, attempt),
                )
                repaired = prompt_from_repair_response(response)
                ratio = language_ratio(repaired)
                repaired_chinese = chinese_text_only(repaired)
                missing_quotes = [
                    value
                    for value in required_quotes
                    if chinese_text_only(value) not in repaired_chinese
                ]
                if ratio < 0.5:
                    raise ValueError(f"Chinese script ratio too low: {ratio:.3f}")
                if missing_quotes:
                    raise ValueError(f"Existing Chinese quotations changed: {missing_quotes}")
                result = {
                    **source,
                    "prompt": repaired,
                    "prompt_sha256": sha256_json(repaired),
                    "language_repair": {
                        "created_at": utc_now(),
                        "model": args.model,
                        "source_prompt_sha256": sha256_json(source_prompt),
                        "source_script_ratio": language_ratio(source_prompt),
                        "repaired_script_ratio": ratio,
                        "preserved_chinese_quotations": required_quotes,
                    },
                }
                completed[record_id] = result
                ordered = [completed[key] for key in sorted(completed)]
                atomic_write_jsonl(args.output, ordered)
                print(f"repaired {record_id} ({len(completed)}/{len(targets)})", flush=True)
                break
            except Exception as exc:  # request/parse/validation retry is intentional
                last_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"retry {record_id} attempt={attempt}: {last_error}",
                    flush=True,
                )
        else:
            raise RuntimeError(f"Could not repair {record_id}: {last_error}")
    return 0


def merge_parts(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    repaired: dict[str, dict[str, Any]] = {}
    for path in args.part:
        for record in read_jsonl(path):
            record_id = str(record["id"])
            if record_id in repaired:
                raise RuntimeError(f"Duplicate repaired ID across parts: {record_id}")
            repaired[record_id] = record
    expected = {str(record["id"]) for record in records if needs_repair(record)}
    if set(repaired) != expected:
        missing = sorted(expected - set(repaired))
        extra = sorted(set(repaired) - expected)
        raise RuntimeError(f"Repair parts incomplete: missing={missing[:10]}, extra={extra[:10]}")
    merged = [repaired.get(str(record["id"]), record) for record in records]
    atomic_write_jsonl(args.output, merged)
    print(json.dumps({"total": len(merged), "repaired": len(repaired)}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--override", type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--sub-index", type=int, default=0)
    parser.add_argument("--sub-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--part", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge:
        return merge_parts(args)
    if not args.api_base or not args.model:
        raise SystemExit("--api-base and --model are required for language repair")
    if not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker index must be in [0, worker_count)")
    if not 0 <= args.sub_index < args.sub_count:
        raise SystemExit("sub index must be in [0, sub_count)")
    return repair_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
