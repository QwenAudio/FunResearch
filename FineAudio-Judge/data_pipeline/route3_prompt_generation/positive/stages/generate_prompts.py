from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

from client import ChatClient, SamplingConfig, prompt_from_response  # noqa: E402
from deduplicate import LexicalDedupIndex  # noqa: E402
from provenance import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    init_run,
    read_jsonl,
    sha256_json,
    update_run_stage,
    utc_now,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sampling(config: dict[str, Any], key: str) -> SamplingConfig:
    section = config[key]
    return SamplingConfig(
        temperature=float(section["temperature"]),
        top_p=float(section["top_p"]),
        max_tokens=int(section["max_tokens"]),
    )


def render(template: str, key: str, value: Any) -> str:
    return template.replace(
        "{" + key + "}", json.dumps(value, ensure_ascii=False, indent=2)
    )


LANGUAGE_INSTRUCTIONS = {
    "zh": (
        "Write the complete audio description in natural Chinese. When the scene "
        "contains intelligible speech, keep the quoted words in Chinese as well."
    ),
    "en": (
        "Write the complete audio description in natural English. When the scene "
        "contains intelligible speech, keep the quoted words in English as well."
    ),
}


def prompt_language_matches(prompt: str, target: str | None) -> bool:
    if target is None:
        return True
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", prompt))
    latin_count = len(re.findall(r"[A-Za-z]", prompt))
    if target == "zh":
        # A few Chinese quotations inside otherwise English narration must not
        # satisfy the Chinese lane.  This is a script-level language check, not
        # a content/scene vocabulary filter.
        return cjk_count >= 20 and cjk_count / max(1, cjk_count + latin_count) >= 0.5
    return cjk_count == 0


def recent_prompt_context(
    accepted: list[dict[str, Any]],
    *,
    max_items: int = 12,
    max_prompt_chars: int = 5000,
) -> list[dict[str, str]]:
    """Return a bounded recent-history window for LLM novelty checks.

    Lexical duplicate detection still compares against every accepted prompt.
    This window only bounds the text embedded in generation and semantic-audit
    requests so their input plus requested output stays within model context.
    """
    selected: list[dict[str, str]] = []
    used_chars = 0
    for record in reversed(accepted):
        prompt = str(record.get("prompt", ""))
        if selected and used_chars + len(prompt) > max_prompt_chars:
            break
        selected.append({"id": str(record.get("id", "")), "prompt": prompt})
        used_chars += len(prompt)
        if len(selected) >= max_items:
            break
    selected.reverse()
    return selected


def generation_seed(run_id: str, persistent_attempt_index: int) -> int:
    """Return a stable, shard-specific seed that survives process restarts."""
    payload = f"{run_id}:{persistent_attempt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def primary_language_lane(run_id: str) -> str:
    """Assign a stable per-shard language lane without changing quotas."""
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return "en" if digest[-1] & 1 else "zh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate traceable Path 3 audio-scene prompts."
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/generation_config.json"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs",
        help="Parent directory for immutable run records.",
    )
    parser.add_argument(
        "--run-id",
        help="Explicit run ID. Reusing it resumes the same prompt batch.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=ROOT / "prompts",
        help="Directory holding generate/rewrite/audit/deduplicate prompt "
             "templates; use a variant dir (e.g. prompts_variants/diverse_motifs) "
             "for regeneration batches with extra constraints.",
    )
    parser.add_argument(
        "--generate-template",
        type=Path,
        help="Override only generate.txt while keeping the shared rewrite/audit rules.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--chinese-count",
        type=int,
        default=0,
        help="Number of the newly requested prompts that must be Chinese.",
    )
    parser.add_argument(
        "--english-count",
        type=int,
        default=0,
        help="Number of the newly requested prompts that must be English.",
    )
    parser.add_argument(
        "--external-dedup-prompts",
        type=Path,
        help="Optional JSONL corpus whose prompts must not be duplicated. "
             "The corpus is used only for lexical duplicate checks and is "
             "not included in this run's output or generation context.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.chinese_count < 0 or args.english_count < 0:
        raise SystemExit("language counts must be non-negative")
    language_controlled = bool(args.chinese_count or args.english_count)
    if language_controlled and args.chinese_count + args.english_count != args.count:
        raise SystemExit("--chinese-count + --english-count must equal --count")
    if args.dry_run and args.run_id:
        raise SystemExit("--dry-run does not create or resume a run")

    config = load_json(args.config)
    prompts_dir = args.prompts_dir
    prompts = {
        "generate": load_text(args.generate_template or prompts_dir / "generate.txt"),
        "rewrite": load_text(prompts_dir / "rewrite_audio_only.txt"),
        "audit": load_text(prompts_dir / "audit.txt"),
        "deduplicate": load_text(prompts_dir / "deduplicate.txt"),
    }
    client = ChatClient(config["api_base"], config["model"])

    run_root: Path | None = None
    paths: dict[str, Path] = {}
    if not args.dry_run:
        oscar_config = load_json(ROOT / "config/oscar_generation_config.json")
        configured_oscar_root = os.environ.get("AUDIO_OSCAR_ROOT") or oscar_config["audio_oscar_root"]
        run_id, run_root = init_run(
            ROOT,
            args.runs_root,
            args.run_id,
            Path(configured_oscar_root),
        )
        stage_root = run_root / "01_prompt_generation"
        paths = {
            "accepted": stage_root / "accepted_prompts.jsonl",
            "candidates": stage_root / "candidates.jsonl",
            "rewrites": stage_root / "rewrites.jsonl",
            "audits": stage_root / "audits.jsonl",
            "rejections": stage_root / "rejections.jsonl",
            "state": stage_root / "state.json",
        }
        accepted = read_jsonl(paths["accepted"])
        persistent_candidate_count = len(read_jsonl(paths["candidates"]))
        update_run_stage(
            run_root,
            "01_prompt_generation",
            "running",
            model=config["model"],
            api_base=config["api_base"],
            requested_new_count=args.count,
            requested_total_count=len(accepted) + args.count,
            prompts_dir=str(prompts_dir),
            external_dedup_prompts=(
                str(args.external_dedup_prompts)
                if args.external_dedup_prompts else None
            ),
        )
        print(f"run_id={run_id}")
    else:
        accepted = []
        persistent_candidate_count = 0

    accepted_prompts = [str(record["prompt"]) for record in accepted]
    accepted_indexes: dict[str, LexicalDedupIndex] = {}
    for record in accepted:
        accepted_indexes.setdefault(
            str(record.get("language", "unknown")), LexicalDedupIndex()
        ).add(str(record["prompt"]))
    external_indexes: dict[str, LexicalDedupIndex] = {}
    if args.external_dedup_prompts:
        if not args.external_dedup_prompts.is_file():
            raise FileNotFoundError(args.external_dedup_prompts)
        for record in read_jsonl(args.external_dedup_prompts):
            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                continue
            external_indexes.setdefault(
                str(record.get("language", "unknown")), LexicalDedupIndex()
            ).add(prompt)
    target_total = len(accepted) + args.count
    initial_language_counts = {
        "zh": sum(record.get("language") == "zh" for record in accepted),
        "en": sum(record.get("language") == "en" for record in accepted),
    }
    target_language_counts = {
        "zh": initial_language_counts["zh"] + args.chinese_count,
        "en": initial_language_counts["en"] + args.english_count,
    }
    validation_config = config["validation"]
    dedup_config = config["deduplication"]
    attempts_for_current = 0
    total_attempts = 0
    primary_language = primary_language_lane(run_id if run_root else "dry_run")

    while len(accepted) < target_total:
        attempts_for_current += 1
        total_attempts += 1
        if attempts_for_current > int(validation_config["maximum_attempts_per_sample"]):
            if run_root:
                update_run_stage(
                    run_root,
                    "01_prompt_generation",
                    "failed",
                    accepted_count=len(accepted),
                    total_attempts=total_attempts,
                    reason="maximum_attempts_exceeded",
                )
            raise RuntimeError(
                "Maximum attempts exceeded for the next sample; inspect rejections.jsonl"
            )

        attempt_id = f"attempt_{total_attempts:06d}"
        target_language: str | None = None
        if language_controlled:
            current_zh = sum(record.get("language") == "zh" for record in accepted)
            current_en = sum(record.get("language") == "en" for record in accepted)
            zh_remaining = target_language_counts["zh"] - current_zh
            en_remaining = target_language_counts["en"] - current_en
            if zh_remaining > 0 and en_remaining > 0:
                target_language = primary_language
            elif zh_remaining > 0:
                target_language = "zh"
            elif en_remaining > 0:
                target_language = "en"
        rejection: dict[str, Any] = {
            "attempt_id": attempt_id,
            "created_at": utc_now(),
            "accepted_count": len(accepted),
        }
        try:
            recent_context = recent_prompt_context(accepted)
            generation_prompt = render(
                prompts["generate"],
                "accepted_json",
                recent_context,
            )
            generation_prompt = generation_prompt.replace(
                "{target_language_instruction}",
                LANGUAGE_INSTRUCTIONS.get(target_language, "Choose the language naturally."),
            )
            candidate_seed = generation_seed(
                run_id if run_root else "dry_run",
                persistent_candidate_count + total_attempts,
            )
            draft = client.json_completion(
                generation_prompt,
                sampling(config, "generation"),
                seed=candidate_seed,
            )
            if run_root:
                append_jsonl(
                    paths["candidates"],
                    {
                        "attempt_id": attempt_id,
                        "created_at": utc_now(),
                        "model": config["model"],
                        "seed": candidate_seed,
                        "raw_response": draft,
                        "response_sha256": sha256_json(draft),
                    },
                )

            rewrite_prompt = render(prompts["rewrite"], "draft_json", draft)
            rewritten = client.json_completion(
                rewrite_prompt, sampling(config, "rewrite")
            )
            candidate = prompt_from_response(rewritten)
            if not prompt_language_matches(candidate, target_language):
                rejection.update(
                    stage="language_mismatch",
                    errors=[f"expected_language={target_language}"],
                    candidate=candidate,
                )
                if run_root:
                    append_jsonl(paths["rejections"], rejection)
                continue
            if run_root:
                append_jsonl(
                    paths["rewrites"],
                    {
                        "attempt_id": attempt_id,
                        "created_at": utc_now(),
                        "model": config["model"],
                        "rewritten_response": rewritten,
                        "prompt_sha256": sha256_json(candidate),
                    },
                )

            audit_prompt = render(
                prompts["audit"], "candidate_json", {"prompt": candidate}
            )
            audit = client.json_completion(audit_prompt, sampling(config, "audit"))
            if run_root:
                append_jsonl(
                    paths["audits"],
                    {
                        "attempt_id": attempt_id,
                        "audit_type": "audio_only_quality",
                        "created_at": utc_now(),
                        "result": audit,
                    },
                )
            if audit.get("pass") is not True:
                try:
                    corrected = prompt_from_response(audit, "corrected_prompt")
                except ValueError as exc:
                    rejection.update(
                        stage="llm_audio_only_correction",
                        errors=list(audit.get("violations", [])) + [str(exc)],
                        unsupported_phrases=audit.get("unsupported_phrases", []),
                        candidate=candidate,
                    )
                    if run_root:
                        append_jsonl(paths["rejections"], rejection)
                    continue

                second_audit_prompt = render(
                    prompts["audit"],
                    "candidate_json",
                    {"prompt": corrected},
                )
                second_audit = client.json_completion(
                    second_audit_prompt, sampling(config, "audit")
                )
                if run_root:
                    append_jsonl(
                        paths["audits"],
                        {
                            "attempt_id": attempt_id,
                            "audit_type": "audio_only_quality_after_correction",
                            "created_at": utc_now(),
                            "result": second_audit,
                        },
                    )
                if second_audit.get("pass") is not True:
                    rejection.update(
                        stage="llm_audit_after_correction",
                        errors=second_audit.get(
                            "violations", ["second_audio_only_audit_failed"]
                        ),
                        unsupported_phrases=second_audit.get(
                            "unsupported_phrases", []
                        ),
                        candidate=corrected,
                    )
                    if run_root:
                        append_jsonl(paths["rejections"], rejection)
                    continue
                candidate = corrected

            language_index = accepted_indexes.setdefault(
                str(target_language or "unknown"), LexicalDedupIndex()
            )
            is_duplicate, duplicate_index, scores = language_index.find_duplicate(
                candidate,
                float(dedup_config["sequence_similarity_threshold"]),
                float(dedup_config["token_jaccard_threshold"]),
            )
            if is_duplicate:
                rejection.update(
                    stage="lexical_deduplication",
                    duplicate_index=duplicate_index,
                    scores=scores,
                    candidate=candidate,
                )
                if run_root:
                    append_jsonl(paths["rejections"], rejection)
                continue

            external_index = external_indexes.get(str(target_language or "unknown"))
            if external_index is not None:
                is_external_duplicate, external_duplicate_index, external_scores = (
                    external_index.find_duplicate(
                        candidate,
                        float(dedup_config["sequence_similarity_threshold"]),
                        float(dedup_config["token_jaccard_threshold"]),
                    )
                )
                if is_external_duplicate:
                    rejection.update(
                        stage="external_lexical_deduplication",
                        duplicate_index=external_duplicate_index,
                        scores=external_scores,
                        candidate=candidate,
                    )
                    if run_root:
                        append_jsonl(paths["rejections"], rejection)
                    continue

            semantic: dict[str, Any] = {
                "duplicate": False,
                "matching_ids": [],
                "reason": "no previously accepted prompt",
            }
            if accepted_prompts:
                dedup_prompt = render(
                    prompts["deduplicate"],
                    "accepted_json",
                    recent_context,
                )
                dedup_prompt = render(
                    dedup_prompt, "candidate_json", {"prompt": candidate}
                )
                semantic = client.json_completion(
                    dedup_prompt, sampling(config, "deduplication")
                )
                if run_root:
                    append_jsonl(
                        paths["audits"],
                        {
                            "attempt_id": attempt_id,
                            "audit_type": "semantic_deduplication",
                            "created_at": utc_now(),
                            "result": semantic,
                        },
                    )
                if semantic.get("duplicate") is True:
                    rejection.update(
                        stage="semantic_deduplication",
                        errors=[semantic.get("reason", "semantic_duplicate")],
                        matching_ids=semantic.get("matching_ids", []),
                        candidate=candidate,
                    )
                    if run_root:
                        append_jsonl(paths["rejections"], rejection)
                    continue

            record = {
                "id": f"path3_{len(accepted):06d}",
                "prompt": candidate,
                "prompt_sha256": sha256_json(candidate),
                "source_attempt_id": attempt_id,
                "source": "qwen3.5_natural_scene_generation",
                "pipeline_stage": "accepted_prompt",
                "created_at": utc_now(),
                "model": config["model"],
                "language": target_language,
                "validation": {
                    "llm_audit": "passed",
                    "deduplication": "passed",
                },
            }
            accepted.append(record)
            accepted_prompts.append(candidate)
            language_index.add(candidate)
            attempts_for_current = 0

            if args.dry_run:
                print(json.dumps(record, ensure_ascii=False, indent=2))
            else:
                atomic_write_jsonl(paths["accepted"], accepted)
                atomic_write_json(
                    paths["state"],
                    {
                        "updated_at": utc_now(),
                        "accepted_count": len(accepted),
                        "target_total": target_total,
                        "attempts_in_this_invocation": total_attempts,
                        "last_accepted_id": record["id"],
                        "language_counts": {
                            "zh": sum(item.get("language") == "zh" for item in accepted),
                            "en": sum(item.get("language") == "en" for item in accepted),
                        },
                    },
                )
                print(f"accepted {record['id']} ({len(accepted)}/{target_total})")
        except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            rejection.update(stage="exception", errors=[f"{type(exc).__name__}: {exc}"])
            if run_root:
                append_jsonl(paths["rejections"], rejection)
            if isinstance(exc, RuntimeError) and "LLM request failed" in str(exc):
                time.sleep(5.0)

    if run_root:
        update_run_stage(
            run_root,
            "01_prompt_generation",
            "completed",
            accepted_count=len(accepted),
            attempts_in_this_invocation=total_attempts,
            output="01_prompt_generation/accepted_prompts.jsonl",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
