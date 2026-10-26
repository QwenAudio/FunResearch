#!/usr/bin/env python3
"""Build 1,200 deterministic random-mismatch pairs from the three base positive routes."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RUN = Path(".")
BASE_INPUT = Path("positive_pairs.jsonl")
RAW_OUTPUT = RUN / "mismatch_pairs.jsonl"
SEED = int.from_bytes(bytes.fromhex("0135282b"), "big")

PATH_ROUTES = {
    "path1": "path1_whole_real_audio",
    "path2": "path2_real_clip_composition",
    "path3": "path3_fully_generated_audio",
}

# Each cross-path unordered pair contributes 100 examples in each direction.
SPECS = (
    ("cross_path2P_path1A", "path2", "path1", 100, "cross_path"),
    ("cross_path1P_path2A", "path1", "path2", 100, "cross_path"),
    ("cross_path3P_path1A", "path3", "path1", 100, "cross_path"),
    ("cross_path1P_path3A", "path1", "path3", 100, "cross_path"),
    ("cross_path3P_path2A", "path3", "path2", 100, "cross_path"),
    ("cross_path2P_path3A", "path2", "path3", 100, "cross_path"),
    ("within_path1", "path1", "path1", 200, "within_path"),
    ("within_path2", "path2", "path2", 200, "within_path"),
    ("within_path3", "path3", "path3", 200, "within_path"),
)

WORD_RE = re.compile(r"[a-z0-9']+", re.I)
CJK_RE = re.compile(r"[\u3400-\u9fff]")
SPEECH_HINT_RE = re.compile(
    r"\b(speak|speaker|speech|voice|says?|said|talk|narrat|accent|utter)\w*\b|"
    r"说话|讲话|语音|嗓音|旁白|口音|发言",
    re.I,
)
STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "then", "before", "after",
    "sound", "audio", "there", "into", "while", "over", "under", "have", "has",
    "says", "said", "speaks", "speaker", "voice", "woman", "man", "male", "female",
    "comes", "occurs", "background", "followed", "overlaps", "continuous", "throughout",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return rows


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_caption(value: str) -> str:
    return " ".join(value.lower().split())


def content_tokens(value: str) -> frozenset[str]:
    tokens = [token.lower() for token in WORD_RE.findall(value)]
    return frozenset(token for token in tokens if len(token) > 2 and token not in STOPWORDS)


def inferred_language(row: dict[str, Any]) -> str:
    explicit = str(row.get("language") or "").lower()
    if explicit in {"en", "zh"}:
        return explicit
    questions = " ".join(str(question.get("question") or "") for question in row.get("questions") or [])
    cjk = len(CJK_RE.findall(questions))
    latin = len(re.findall(r"[A-Za-z]", questions))
    return "zh" if cjk >= 8 and cjk > latin * 0.2 else "en"


def has_speech(row: dict[str, Any]) -> bool:
    questions = row.get("questions") or []
    if any(str(question.get("subtype") or "").startswith("speech_") for question in questions):
        return True
    probe = str(row.get("caption") or "")[:8000]
    return bool(SPEECH_HINT_RE.search(probe))


def duration_bucket(seconds: float) -> int:
    for index, boundary in enumerate((10.0, 20.0, 40.0, 60.0)):
        if seconds < boundary:
            return index
    return 4


def profile(row: dict[str, Any], path_name: str) -> dict[str, Any]:
    caption = str(row.get("caption") or row.get("prompt") or "")
    duration = float(row.get("audio_duration_seconds") or row.get("duration") or 0)
    return {
        "row": row,
        "path": path_name,
        "sample_id": str(row["sample_id"]),
        "caption_norm": normalized_caption(caption),
        "caption_hash": text_sha256(caption),
        "tokens": content_tokens(caption),
        "language": inferred_language(row),
        "has_speech": has_speech(row),
        "duration": duration,
        "duration_bucket": duration_bucket(duration),
    }


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def choose_audio(
    prompt: dict[str, Any],
    available: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = list(available)
    rng.shuffle(candidates)
    candidates = candidates[: min(256, len(candidates))]
    ranked: list[tuple[tuple[float, ...], dict[str, Any], float]] = []
    for audio in candidates:
        if prompt["sample_id"] == audio["sample_id"]:
            continue
        if prompt["caption_norm"] == audio["caption_norm"]:
            continue
        similarity = jaccard(prompt["tokens"], audio["tokens"])
        if similarity >= 0.55:
            continue
        duration_ratio = abs(math.log(max(prompt["duration"], 0.1) / max(audio["duration"], 0.1)))
        score = (
            float(prompt["language"] != audio["language"]),
            float(prompt["has_speech"] != audio["has_speech"]),
            float(abs(prompt["duration_bucket"] - audio["duration_bucket"])),
            duration_ratio,
            similarity,
            rng.random(),
        )
        ranked.append((score, audio, similarity))
    if not ranked:
        raise RuntimeError(f"no compatible audio donor for {prompt['sample_id']}")
    score, audio, similarity = min(ranked, key=lambda item: item[0])
    matching = {
        "prompt_language": prompt["language"],
        "audio_language": audio["language"],
        "language_matched": prompt["language"] == audio["language"],
        "prompt_has_speech": prompt["has_speech"],
        "audio_has_speech": audio["has_speech"],
        "speech_presence_matched": prompt["has_speech"] == audio["has_speech"],
        "prompt_duration_bucket": prompt["duration_bucket"],
        "audio_duration_bucket": audio["duration_bucket"],
        "duration_bucket_distance": abs(prompt["duration_bucket"] - audio["duration_bucket"]),
        "caption_token_jaccard": round(similarity, 6),
    }
    return audio, matching


def main() -> None:
    global RUN, BASE_INPUT, RAW_OUTPUT, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-frozen-pool-check", action="store_true",
        help="Allow larger or alternative positive pools while retaining all pairing constraints.",
    )
    args = parser.parse_args()
    RUN = args.output_dir
    BASE_INPUT = args.positive_pairs
    RAW_OUTPUT = RUN / "mismatch_pairs.jsonl"
    SEED = args.seed
    base_rows = read_jsonl(BASE_INPUT)
    base_ids = {str(row["sample_id"]) for row in base_rows}
    pools: dict[str, list[dict[str, Any]]] = {}
    for path_name, route in PATH_ROUTES.items():
        aliases = {
            "path1": {route, "route1_real_world"},
            "path2": {route, "route2_deterministic_composition"},
            "path3": {route, "route3_prompt_generation"},
        }[path_name]
        selected = [row for row in base_rows if row.get("route") in aliases and row.get("positive_pair") is not False]
        pools[path_name] = [profile(row, path_name) for row in selected]
    expected_pools = {"path1": 1959, "path2": 2000, "path3": 1916}
    actual_pools = {key: len(value) for key, value in pools.items()}
    if not args.skip_frozen_pool_check and actual_pools != expected_pools:
        raise ValueError(f"unexpected positive pools: {actual_pools}")
    if any(actual_pools[name] < 800 for name in PATH_ROUTES):
        raise ValueError(f"each route needs at least 800 unique donors: {actual_pools}")

    rng = random.Random(SEED)
    used: dict[str, set[str]] = defaultdict(set)
    outputs: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []

    for code, prompt_path, audio_path, count, scope in SPECS:
        prompt_candidates = [row for row in pools[prompt_path] if row["sample_id"] not in used[prompt_path]]
        prompt_candidates.sort(key=lambda row: row["sample_id"])
        rng.shuffle(prompt_candidates)
        prompts = prompt_candidates[:count]
        if len(prompts) != count:
            raise RuntimeError(f"insufficient prompt donors for {code}")
        used[prompt_path].update(row["sample_id"] for row in prompts)

        for index, prompt in enumerate(prompts, 1):
            audio_candidates = [row for row in pools[audio_path] if row["sample_id"] not in used[audio_path]]
            audio, matching = choose_audio(prompt, audio_candidates, rng)
            used[audio_path].add(audio["sample_id"])
            prompt_row = prompt["row"]
            audio_row = audio["row"]
            identity = f"{code}|{prompt['sample_id']}|{audio['sample_id']}|{SEED}"
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
            sample_id = f"random_mismatch_{code}_{index:04d}_{suffix}"
            source_audio = str(audio_row.get("source_audio_path") or audio_row["audio_path"])
            questions = prompt_row.get("questions") or []
            output = {
                "schema_version": "random_mismatch_input",
                "sample_id": sample_id,
                "source_id": sample_id,
                "route": f"random_mismatch_{code}",
                "mismatch_scope": scope,
                "pair_form": f"{prompt_path}P_{audio_path}A",
                "positive_pair": False,
                "audio_path": str(audio_row["audio_path"]),
                "source_audio_path": source_audio,
                "audio_duration_seconds": float(audio_row["audio_duration_seconds"]),
                "caption": prompt_row.get("caption") or prompt_row.get("prompt"),
                "language": prompt["language"],
                "questions": questions,
                "prompt_parent_id": prompt["sample_id"],
                "prompt_parent_route": prompt_row["route"],
                "prompt_parent_path": prompt_path,
                "audio_parent_id": audio["sample_id"],
                "audio_parent_route": audio_row["route"],
                "audio_parent_path": audio_path,
                "prompt_caption_sha256": prompt["caption_hash"],
                "audio_parent_caption_sha256": audio["caption_hash"],
                "random_seed": SEED,
                "matching": matching,
                "source_manifest": str(BASE_INPUT),
                "source_rubrics": prompt_row.get("source_rubrics"),
            }
            if sample_id in base_ids:
                raise ValueError(f"sample ID collision with base: {sample_id}")
            outputs.append(output)
            provenance.append({
                "sample_id": sample_id,
                "prompt_parent_id": prompt["sample_id"],
                "audio_parent_id": audio["sample_id"],
                "route": output["route"],
                "matching": matching,
            })

    if len(outputs) != 1200 or len({row["sample_id"] for row in outputs}) != 1200:
        raise ValueError("1,200-pair output invariant failed")
    expected_routes = {f"random_mismatch_{code}": count for code, _, _, count, _ in SPECS}
    actual_routes = Counter(row["route"] for row in outputs)
    if dict(actual_routes) != expected_routes:
        raise ValueError(f"route quota mismatch: {dict(actual_routes)}")
    for path_name in PATH_ROUTES:
        if len(used[path_name]) != 800:
            raise ValueError(f"{path_name}: expected 800 unique donor rows, got {len(used[path_name])}")

    atomic_jsonl(RAW_OUTPUT, outputs)
    atomic_jsonl(RUN / "provenance/pair_sources.jsonl", provenance)
    summary = {
        "schema_version": "random_mismatch_build_summary",
        "created_at": now(),
        "seed": SEED,
        "base_input": str(BASE_INPUT),
        "base_input_sha256": sha256(BASE_INPUT),
        "positive_pools": actual_pools,
        "pairs": len(outputs),
        "questions": sum(len(row["questions"]) for row in outputs),
        "pairs_by_route": dict(sorted(actual_routes.items())),
        "questions_by_route": dict(sorted(Counter({
            route: sum(len(row["questions"]) for row in outputs if row["route"] == route)
            for route in actual_routes
        }).items())),
        "unique_donor_rows_by_path": {path_name: len(used[path_name]) for path_name in PATH_ROUTES},
        "matching": {
            "language_matched": sum(bool(row["matching"]["language_matched"]) for row in outputs),
            "speech_presence_matched": sum(bool(row["matching"]["speech_presence_matched"]) for row in outputs),
            "same_duration_bucket": sum(row["matching"]["duration_bucket_distance"] == 0 for row in outputs),
            "max_caption_token_jaccard": max(row["matching"]["caption_token_jaccard"] for row in outputs),
        },
        "raw_output": str(RAW_OUTPUT),
        "raw_output_sha256": sha256(RAW_OUTPUT),
    }
    atomic_json(RUN / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
