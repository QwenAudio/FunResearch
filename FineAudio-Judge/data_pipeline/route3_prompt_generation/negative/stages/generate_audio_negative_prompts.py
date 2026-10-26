#!/usr/bin/env python3
"""Generate exact Path3 counterfactual prompt patches with deferred collateral handling."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


EXPLICIT_TIME = re.compile(
    r"(?:\b\d{1,2}:\d{2}\b|\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|秒)\b|"
    r"\bfrom\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b)", re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_text(row: dict[str, Any]) -> str:
    changed = [
        {key: item.get(key) for key in ("rubric_id", "rubric_type", "question", "basis", "parent_ids", "target_ids")}
        for item in row["changed_rubrics"]
    ]
    retained = [
        {key: item.get(key) for key in ("rubric_id", "rubric_type", "question", "basis")}
        for item in row["retained_rubrics"]
    ]
    return f"""Propose LOCAL patches for a frozen audio QA prompt. Do not rewrite the whole prompt.

Contradict every PLANNED CHANGE fact while preserving unrelated text and facts as much as possible.

Output rules:
- Return JSON only with key `patches`.
- Each patch contains rubric_ids, scope, original_span, replacement_span, replacement_facts, audible_test.
- `rubric_ids` should list PLANNED CHANGE IDs covered by that patch.
- If the same necessary edit may also affect a RETAINED rubric, put that ID in optional
  `collateral_rubric_ids`; do not hide the effect and do not put it in `rubric_ids`.
- original_span must be copied verbatim from FROZEN PROMPT and occur exactly once.
- Use the smallest natural word, phrase, clause, or sentence that can express the change.
- The union of planned IDs covered by all patches must equal PLANNED CHANGE exactly.
- Patches must not overlap. Do not alter text outside patches.
- Do not introduce numeric timestamps or durations.
- Keep intelligible speech in quotation marks.
- Replacements must remain feasible for Audio-Oscar and audibly distinguishable.
- Keep the answer compact: group compatible IDs into one patch; use at most three short
  replacement_facts and one short audible_test sentence per patch.

FROZEN PROMPT:
{row['natural_prompt']}

PLANNED CHANGE:
{json.dumps(changed, ensure_ascii=False, indent=2)}

PLANNED RETAIN:
{json.dumps(retained, ensure_ascii=False, indent=2)}
"""


def parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


def apply_patches(row: dict[str, Any], patches: Any) -> tuple[str, list[dict[str, Any]], list[str]]:
    if not isinstance(patches, list) or not patches:
        raise ValueError("patches must be a non-empty array")
    prompt = str(row["natural_prompt"])
    expected = {str(value) for value in row["changed_rubric_ids"]}
    known = expected | {str(value) for value in row["retained_rubric_ids"]}
    covered: list[str] = []
    collateral: set[str] = set()
    located: list[dict[str, Any]] = []
    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            raise ValueError(f"patch {index} is not an object")
        original = str(patch.get("original_span") or "")
        replacement = str(patch.get("replacement_span") or "")
        if not original or not replacement or original == replacement:
            raise ValueError(f"patch {index} has an empty or identical span")
        if prompt.count(original) != 1:
            raise ValueError(f"patch {index} original_span occurs {prompt.count(original)} times")
        if EXPLICIT_TIME.search(replacement):
            raise ValueError(f"patch {index} introduces explicit numeric time")
        declared = [str(value) for value in patch.get("rubric_ids") or []]
        declared += [str(value) for value in patch.get("collateral_rubric_ids") or []]
        unknown = set(declared) - known
        if unknown:
            raise ValueError(f"patch {index} declares unknown rubric IDs: {sorted(unknown)}")
        planned_here = [value for value in declared if value in expected]
        collateral_here = sorted(set(declared) - expected)
        if not planned_here:
            raise ValueError(f"patch {index} covers no planned rubric ID")
        covered.extend(planned_here)
        collateral.update(collateral_here)
        start = prompt.index(original)
        located.append({
            **patch,
            "rubric_ids": planned_here,
            "declared_collateral_rubric_ids": collateral_here,
            "start_char": start,
            "end_char": start + len(original),
        })
    if set(covered) != expected:
        raise ValueError(
            f"planned rubric union mismatch missing={sorted(expected-set(covered))} "
            f"extra={sorted(set(covered)-expected)}"
        )
    duplicated = sorted({value for value in covered if covered.count(value) > 1})
    if duplicated:
        raise ValueError(f"planned rubric IDs covered more than once: {duplicated}")
    located.sort(key=lambda item: item["start_char"])
    if any(left["end_char"] > right["start_char"] for left, right in zip(located, located[1:])):
        raise ValueError("patches overlap")
    pieces: list[str] = []
    cursor = 0
    new_cursor = 0
    for patch in located:
        untouched = prompt[cursor:patch["start_char"]]
        pieces.append(untouched)
        new_cursor += len(untouched)
        patch["new_start_char"] = new_cursor
        pieces.append(str(patch["replacement_span"]))
        new_cursor += len(str(patch["replacement_span"]))
        patch["new_end_char"] = new_cursor
        cursor = patch["end_char"]
    pieces.append(prompt[cursor:])
    changed = "".join(pieces)
    restored = changed
    for patch in reversed(located):
        restored = restored[:patch["new_start_char"]] + patch["original_span"] + restored[patch["new_end_char"]:]
    if restored != prompt:
        raise AssertionError("outside-patch text changed")
    return changed, located, sorted(collateral)


def repair_instruction(error: str, raw: str) -> str:
    previous = raw[-10000:] if raw else "(previous output was not parseable)"
    return f"""

REPAIR THE PREVIOUS OUTPUT instead of starting over blindly.
Validator error: {error}

Repair checklist:
- If patches overlap, merge all edits touching the same source span into one patch and
  union their planned rubric_ids.
- Every replacement_span must differ from its original_span.
- A patch must cover at least one PLANNED CHANGE ID; never emit a collateral-only patch.
- Each PLANNED CHANGE ID must occur exactly once across rubric_ids. Do not put a planned
  ID in collateral_rubric_ids.
- Return the complete corrected JSON object, not a diff or explanation.

PREVIOUS INVALID OUTPUT:
{previous}
"""


def whole_prompt_fallback_instruction(row: dict[str, Any]) -> str:
    """Use a single full-prompt patch when a complex low-band row resists local repair."""
    return f"""

FINAL FALLBACK FOR A STRUCTURALLY DIFFICULT LOW-CONSISTENCY SAMPLE:
- Return exactly ONE patch.
- Copy the entire FROZEN PROMPT verbatim into original_span.
- replacement_span must be a complete, natural rewritten prompt, not a short fragment.
- Contradict every PLANNED CHANGE while retaining other facts where feasible.
- Set rubric_ids to exactly this list, once each:
  {json.dumps(row['changed_rubric_ids'], ensure_ascii=False)}
- Do not include collateral_rubric_ids and do not introduce numeric timestamps.
"""


async def call(client: AsyncOpenAI, model: str, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = str(response.choices[0].message.content or "")
    return raw


async def run(args: argparse.Namespace) -> None:
    plan = read_jsonl(args.plan)
    if args.limit:
        plan = plan[: args.limit]
    latest = {str(row.get("sample_id")): row for row in read_jsonl(args.output)}
    pending = [row for row in plan if latest.get(row["sample_id"], {}).get("status") != "ok"]
    clients = [AsyncOpenAI(base_url=url, api_key="EMPTY", timeout=args.timeout) for url in args.endpoint]
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(index: int, row: dict[str, Any]) -> None:
        async with semaphore:
            raw = ""
            prior = latest.get(row["sample_id"], {})
            previous_raw = str(prior.get("raw_model_output") or "") if prior.get("status") == "error" else ""
            previous_error = str(prior.get("error") or "") if prior.get("status") == "error" else ""
            error: Exception | None = None
            for attempt in range(1, args.retries + 1):
                try:
                    prompt = request_text(row)
                    if previous_error:
                        prompt += repair_instruction(previous_error, previous_raw)
                    use_full_fallback = attempt == args.retries and row.get("target_band") == "low"
                    if use_full_fallback:
                        prompt += whole_prompt_fallback_instruction(row)
                    raw = await call(clients[index % len(clients)], args.model, prompt)
                    value = parse_json(raw)
                    patches = value.get("patches")
                    if use_full_fallback and isinstance(patches, list) and patches:
                        replacement = str(patches[0].get("replacement_span") or "")
                        original = str(row["natural_prompt"])
                        if len(replacement) >= max(20, int(len(original) * 0.35)):
                            patches = [{
                                "rubric_ids": list(row["changed_rubric_ids"]),
                                "scope": "full_prompt_fallback",
                                "original_span": original,
                                "replacement_span": replacement,
                                "replacement_facts": patches[0].get("replacement_facts") or [],
                                "audible_test": patches[0].get("audible_test") or "Full-prompt counterfactual rewrite.",
                            }]
                    patched, audit, collateral = apply_patches(row, patches)
                    result = {
                        "sample_id": row["sample_id"],
                        "status": "ok",
                        "attempt": attempt,
                        "original_prompt": row["natural_prompt"],
                        "patched_prompt": patched,
                        "original_prompt_sha256": sha256_text(row["natural_prompt"]),
                        "patched_prompt_sha256": sha256_text(patched),
                        "changed_rubric_ids": row["changed_rubric_ids"],
                        "retained_rubric_ids": row["retained_rubric_ids"],
                        "declared_collateral_rubric_ids": collateral,
                        "patches": audit,
                        "outside_patch_exact": True,
                        "raw_model_output": raw,
                    }
                    break
                except Exception as exc:
                    error = exc
                    previous_error = repr(exc)
                    previous_raw = raw
            else:
                result = {
                    "sample_id": row["sample_id"], "status": "error",
                    "error": repr(error), "raw_model_output": raw,
                }
            async with lock:
                append_jsonl(args.output, result)

    await asyncio.gather(*(one(index, row) for index, row in enumerate(pending)))
    latest = {str(row.get("sample_id")): row for row in read_jsonl(args.output)}
    good = [latest[row["sample_id"]] for row in plan if latest.get(row["sample_id"], {}).get("status") == "ok"]
    by_id = {row["sample_id"]: row for row in plan}
    pprime = []
    for proposal in good:
        source = by_id[proposal["sample_id"]]
        if source["pair_type"] != "P_prime_A":
            continue
        pprime.append({
            "sample_id": f"{proposal['sample_id']}__pprime",
            "prompt": proposal["patched_prompt"],
            "rubric_source_prompt": proposal["patched_prompt"],
            "rubric_source_prompt_sha256": proposal["patched_prompt_sha256"],
            "audio_path": None,
            "route": "path3_prompt_side_negative",
            "variant": f"{source['target_band']}_{source['language']}",
            "metadata": {"source_sample_id": source["source_pair_id"], "construction_id": source["sample_id"]},
        })
    write_jsonl(args.pprime_output, pprime)
    print(json.dumps({"requested": len(plan), "completed": len(good), "remaining": len(plan)-len(good)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pprime-output", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Validation-only prefix length; 0 processes the complete audio-side plan.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
