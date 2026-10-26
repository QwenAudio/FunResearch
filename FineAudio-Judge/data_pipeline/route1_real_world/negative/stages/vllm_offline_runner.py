#!/usr/bin/env python3
"""Run Route 1 P-prime generation directly with a local vLLM offline engine.

Each GPU owns one deterministic shard and one append-only checkpoint.  The
``merge`` command revalidates successful rows and writes the canonical
``model/patch_attempts.jsonl`` consumed by the Route 1 negative pipeline.
No HTTP client or serving process is used.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import socket
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import counterfactual_pipeline as pipeline
except ImportError:  # Direct script execution.
    import counterfactual_pipeline as pipeline


BACKEND = "vllm_offline"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    append_jsonl_batch(path, [row])


def append_jsonl_batch(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def specialized_feedback(error: str) -> str:
    """Translate validator codes into a concrete one-step repair instruction."""
    guidance = []
    mappings = (
        ("original_span_not_unique", "Extend original_span with adjacent verbatim context until it occurs exactly once."),
        ("noncontiguous_ellipsis", "Use one contiguous literal source substring; never join spans with ellipses."),
        ("overlapping_patches", "Merge overlapping edits into one patch and attach the union of their direct rubric IDs."),
        ("patch_rubric_union_mismatch", "Return every DIRECT TARGET ID exactly once. If several IDs share one source span, attach them to one patch; never repeat an ID across patches."),
        ("edit_kind_target_mismatch", "Prefer one non-overlapping patch per direct target kind. Do not attach a relation ID to an event patch when the relation connective can be edited separately; use mixed only for one inseparable span carrying multiple direct kinds."),
        ("explicit_absence_shortcut", "Use a positively present alternative sound. Do not use no/not/without/silence/stop/cease/absent/missing, and do not describe removing the original event."),
        ("visual_leakage", "Delete visual wording and describe only what can be heard."),
        ("music_genre_near_neighbor", "Do not substitute one music genre for another. Follow the category rule: preserve event identity and change the requested property for an attribute target; choose a different coarse family for an event target."),
        ("accent_near_neighbor", "Use a mutually exclusive non-accent audible property or a coarse event replacement."),
        ("patched_source_fraction_too_large", "Shorten each original_span to the smallest unique target clause. Keep surrounding source text outside the patch byte-identical and never rewrite the whole prompt."),
        ("explanation_missing", "Name every patch rubric ID verbatim in contradiction_explanation."),
        ("not_high_distance", "Set semantic_distance to high and make the contradiction coarse and mutually exclusive."),
        ("event_family", "Use the controlled event-family fields and make them agree with the rubric domain and replacement text."),
        ("original_family_text_mismatch", "Set original_event_family to the coarse family directly supported by the original target text; sound_effect may serve as the umbrella for impact, mechanical, vehicle, or footsteps sounds."),
        ("replacement_family_text_mismatch", "Make replacement_event_family agree with the replacement wording, or rewrite the replacement as the intended coarse family."),
        ("stale_event_coreference", "Expand the same contiguous event patch through the adjacent temporal reference and update that reference to the replacement family; never leave phrases such as 'after the speech ends' after replacing speech with a non-speech event."),
        ("event_identity", "For an attribute edit, preserve the same coarse audible event family."),
        ("same_coarse_identity", "For an event edit, choose a different coarse audible event family."),
        ("relation_endpoint", "Return two unchanged relation_endpoints that occur verbatim in both the original and edited prompt."),
        ("relation_patch_subtype", "Use an explicit relation phrase that is a permitted flip of the source subtype; for before, use an unambiguous after or overlap construction rather than prior to or another before construction."),
        ("incidental_relation_change", "When updating a temporal reference to an event replacement, preserve the original relation subtype unless that relation is also a direct target."),
        ("stale_speech_content", "When replacing speech with a non-speech event, remove the complete explanation, topic, quotations, measurements, and other propositions carried by that speech."),
        ("invented_proper_names", "Remove invented names and narrative context; edit only audible event content."),
        ("article_boundary_mismatch", "Include the adjacent a/an article in original_span and make the replacement phrase grammatical."),
        ("empty_or_unchanged", "Return a non-empty replacement that clearly differs from the exact source span."),
        ("no JSON object", "Return exactly one JSON object with top-level key patches and no prose."),
    )
    for marker, instruction in mappings:
        if marker in error:
            guidance.append(instruction)
    if not guidance:
        guidance.append("Repair exactly the validator defect while preserving all other valid patch fields.")
    return f"{error}\nTARGETED REPAIR: {' '.join(guidance)}"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_rows(plan_rows: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError(f"invalid shard {shard_id}/{num_shards}")
    ordered = sorted(plan_rows, key=lambda row: int(row["execution_rank"]))
    return [row for row in ordered if int(row["execution_rank"]) % num_shards == shard_id]


def default_shard_path(run: Path, shard_id: int) -> Path:
    return run / f"model/offline_patch_attempts/shard_{shard_id:02d}.jsonl"


def shard_attempt_paths(run: Path, shard_id: int) -> list[Path]:
    """Return the primary and any independent shadow checkpoints for a shard."""
    directory = run / "model/offline_patch_attempts"
    return sorted(directory.glob(f"shard_{shard_id:02d}*.jsonl"))


def output_lock_path(run: Path, shard_id: int, output: Path) -> Path:
    primary = default_shard_path(run, shard_id).resolve()
    if output.resolve() == primary:
        return run / f"state/offline_shard_{shard_id:02d}.lock"
    suffix = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:16]
    return run / f"state/offline_shadow_{shard_id:02d}_{suffix}.lock"


def valid_successes(path: Path, plan_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    successes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if row.get("status") != "ok" or sample_id not in plan_by_id:
            continue
        try:
            pipeline.validate_stored_patch(plan_by_id[sample_id], row)
        except Exception:
            continue
        successes[sample_id] = row
    return successes


def valid_successes_across(
    paths: list[Path], plan_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    successes: dict[str, dict[str, Any]] = {}
    for path in paths:
        successes.update(valid_successes(path, plan_by_id))
    return successes


def latest_failure_feedback(
    path: Path, plan_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Resume each pending sample with its most recent validator diagnosis."""
    latest: dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in plan_by_id:
            continue
        if row.get("status") == "ok":
            latest.pop(sample_id, None)
            continue
        error = str(row.get("error") or "")
        if error:
            latest[sample_id] = specialized_feedback(error)
    return latest


def latest_failure_feedback_across(
    paths: list[Path], plan_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    rows = [row for path in paths for row in read_jsonl(path)]
    rows.sort(key=lambda row: str(row.get("timestamp") or ""))
    latest: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in plan_by_id:
            continue
        if row.get("status") == "ok":
            latest.pop(sample_id, None)
            continue
        error = str(row.get("error") or "")
        if error:
            latest[sample_id] = specialized_feedback(error)
    return latest


def output_texts(request_output: Any, maximum: int | None = None) -> list[str]:
    candidates = getattr(request_output, "outputs", None) or []
    if not candidates:
        raise ValueError("vLLM returned no candidate output")
    selected = candidates if maximum is None else candidates[:maximum]
    return [str(candidate.text or "") for candidate in selected]


def output_text(request_output: Any) -> str:
    """Backward-compatible single-candidate accessor used by existing tests/tools."""
    return output_texts(request_output, 1)[0]


def tail_candidate_fanout(
    active_count: int, batch_size: int, remaining_attempts: int, maximum: int = 8,
) -> int:
    """Spend otherwise-idle decode slots on independent hard-tail candidates."""
    if active_count < 1 or batch_size < 1 or remaining_attempts < 1:
        raise ValueError("fanout inputs must be positive")
    return max(1, min(maximum, batch_size // active_count, remaining_attempts))


def compact_patch_request(plan: dict[str, Any], previous_error: str | None) -> str:
    """Drop large non-target rubric inventories while preserving edit semantics."""
    retry = f"\nPREVIOUS VALIDATOR ERROR: {previous_error}\nRepair that defect.\n" if previous_error else ""
    category = str(plan["disturbance_category"])
    override = ""
    if category == "event":
        override = """
EVENT HARD-TAIL OVERRIDE:
- Replace each target event with a completely unrelated, positively present environmental or mechanical sound.
- Good alternatives include thunder, a vehicle horn, glass shattering, an alarm, or heavy machinery.
- Do not preserve the original speaker, source, action, object, quoted words, music genre, or their synonyms.
"""
    elif category == "relation":
        override = """
RELATION HARD-TAIL OVERRIDE:
- Keep both related sounds present and explicitly reverse before/after, overlap/separation, or foreground/background.
- Never express the reversal by stopping, silencing, removing, or negating either sound.
"""
    elif category == "attribute":
        override = """
ATTRIBUTE HARD-TAIL OVERRIDE:
- Keep the audible event identity but choose a mutually exclusive coarse property.
- Do not merely make a small intensity change and do not switch music genre or accent to a near neighbour.
"""
    return f"""Create one controlled audio-description negative using exact-span patches only.

The source is positive by construction. No labels or audio are available. Change P only.
Return JSON only.

Rules:
- Contradict every DIRECT TARGET rubric exactly once.
- Event: replace it with a clearly different coarse audible event.
- Attribute: keep event identity and use a mutually exclusive audible property.
- Relation: keep both event identities and flip their temporal or foreground/background relation.
- Do not use absence, silence, no/not/without, visual facts, timestamps, causality, or near neighbours.
- original_span is one unique verbatim contiguous substring of FROZEN P; patches do not overlap.
- Text outside patches remains byte-identical and the original language is preserved.
- Every patch has edit_kind, semantic_distance=high, contradiction_explanation, and audible_test.
- Each explanation names every attached rubric ID.
- Do not invent names, organizations, locations, quotations, or narrative context.
- Event and attribute patches include original_event_family and replacement_event_family.
- Relation patches include two relation_endpoints found in the edited prompt. Endpoints not directly
  replaced by an event target must also remain unchanged in the source; an event-replaced endpoint
  uses its replacement anchor from the edited prompt.
- Each replacement remains grammatical with its unchanged left and right context.
{override}

LANGUAGE: {plan['language']}
CATEGORY: {plan['disturbance_category']}
DIRECT TARGET IDS: {json.dumps(plan['direct_target_rubric_ids'], ensure_ascii=False)}
DIRECT TARGET RUBRICS:
{json.dumps([pipeline.compact_rubric(row) for row in plan['direct_target_rubrics']], ensure_ascii=False)}

TARGET-SPECIFIC INSTRUCTIONS:
{pipeline.target_specific_instructions(plan)}

FROZEN P:
{plan['original_prompt']}
{retry}
Return exactly:
{{"patches":[{{"rubric_ids":["R..."],"original_span":"exact source substring",
"replacement_span":"natural counterfactual text","edit_kind":"event|attribute|relation|mixed",
"original_event_family":"controlled coarse family","replacement_event_family":"controlled coarse family",
"relation_endpoints":["unchanged event anchor 1","unchanged event anchor 2"],
"semantic_distance":"high","contradiction_explanation":"...","audible_test":"..."}}]}}"""


def chat_token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True,
    )
    return len(encoded["input_ids"])


def bounded_patch_request(
    plan: dict[str, Any], previous_error: str | None, tokenizer: Any, max_input_tokens: int,
) -> tuple[str, int, bool]:
    # Once a full request has failed, keep only the frozen prompt, direct
    # targets, and targeted repair diagnosis.  The complete validator still
    # enforces retained text, edit distance, and every structural invariant.
    if previous_error:
        prompt = compact_patch_request(plan, previous_error)
        tokens = chat_token_count(tokenizer, prompt)
        if tokens > max_input_tokens:
            raise ValueError(
                f"compact_prompt_too_long:{tokens}>{max_input_tokens}; sample={plan['sample_id']}"
            )
        return prompt, tokens, True
    prompt = pipeline.patch_request(plan, previous_error)
    tokens = chat_token_count(tokenizer, prompt)
    if tokens <= max_input_tokens:
        return prompt, tokens, False
    prompt = compact_patch_request(plan, previous_error)
    tokens = chat_token_count(tokenizer, prompt)
    if tokens > max_input_tokens:
        raise ValueError(
            f"compact_prompt_too_long:{tokens}>{max_input_tokens}; sample={plan['sample_id']}"
        )
    return prompt, tokens, True


def engine_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "model_path": str(args.model_path.resolve()),
        "model_path_config_sha256": file_sha256(args.model_path / "config.json"),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "retries": args.retries,
        "tail_candidate_fanout_maximum": 8,
    }


def run_shard(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    run = args.run.resolve()
    output = args.output.resolve() if args.output else default_shard_path(run, args.shard_id)
    shard_lock_path = output_lock_path(run, args.shard_id, output)
    shard_lock_path.parent.mkdir(parents=True, exist_ok=True)
    shard_lock = shard_lock_path.open("a", encoding="utf-8")
    # Acquire before loading the model.  This allows a freed GPU on another
    # host to take over a queued shard without duplicate inference or writes.
    fcntl.flock(shard_lock.fileno(), fcntl.LOCK_EX)
    plan_path = run / "plan/counterfactual_plan.jsonl"
    all_plan = read_jsonl(plan_path)
    selected = shard_rows(all_plan, args.shard_id, args.num_shards)
    if args.limit:
        selected = selected[: args.limit]
    plan_by_id = {str(row["sample_id"]): row for row in selected}
    inputs = shard_attempt_paths(run, args.shard_id)
    completed = valid_successes_across(inputs, plan_by_id)
    pending = [row for row in selected if row["sample_id"] not in completed]
    resumed_feedback = latest_failure_feedback_across(inputs, plan_by_id)

    metadata = engine_metadata(args)
    metadata.update({
        "timestamp": now(),
        "plan_path": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "output_path": str(output),
        "shard_lock_path": str(shard_lock_path),
        "selected": len(selected),
        "resumed_successes": len(completed),
        "pending": len(pending),
    })
    metadata_path = output.with_suffix(".metadata.json")
    atomic_write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    if not pending:
        print(json.dumps({"shard": args.shard_id, "completed": len(completed), "remaining": 0}))
        return

    llm = LLM(
        model=str(args.model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
        safetensors_load_strategy="eager",
        skip_mm_profiling=True,
    )
    tokenizer = llm.get_tokenizer()
    max_input_tokens = args.max_model_len - args.max_tokens - 32

    # Keep every inference batch full.  The previous nested loop exhausted all
    # retries for one fixed batch before admitting new samples, so successful
    # rows made later retry calls progressively shrink to only one or two
    # sequences.  A FIFO retry queue admits untouched samples immediately and
    # sends failures to the back with their targeted validator feedback.
    work = deque(
        {
            "plan": row,
            "previous_error": resumed_feedback.get(str(row["sample_id"])),
            "attempt": 1,
        }
        for row in pending
    )
    while work:
        active = [work.popleft() for _ in range(min(args.batch_size, len(work)))]
        bounded = [
            bounded_patch_request(item["plan"], item["previous_error"], tokenizer, max_input_tokens)
            for item in active
        ]
        prompts = [item[0] for item in bounded]
        input_tokens = [item[1] for item in bounded]
        compact_flags = [item[2] for item in bounded]
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        started = [now() for _ in active]
        fanout = tail_candidate_fanout(
            len(active), args.batch_size,
            min(args.retries - int(item["attempt"]) + 1 for item in active),
        )
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=fanout,
        )
        outputs = llm.chat(
            conversations,
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        if len(outputs) != len(active):
            raise RuntimeError(f"vLLM output count mismatch: {len(outputs)} != {len(active)}")
        attempt_records: list[dict[str, Any]] = []
        for item, prompt, prompt_tokens, compacted, started_at, generated in zip(
            active, prompts, input_tokens, compact_flags, started, outputs,
        ):
            plan = item["plan"]
            base_attempt = int(item["attempt"])
            accepted = False
            last_error = ""
            consumed = 0
            for candidate_index, raw in enumerate(output_texts(generated, fanout)):
                consumed += 1
                attempt = base_attempt + candidate_index
                common = {
                    "schema_version": pipeline.SCHEMA_VERSION,
                    "timestamp": now(),
                    "started_at": started_at,
                    "stage": "generate",
                    "backend": BACKEND,
                    "sample_id": plan["sample_id"],
                    "parent_positive_id": plan["parent_positive_id"],
                    "status": "error",
                    "attempt": attempt,
                    "candidate_index": candidate_index,
                    "candidate_fanout": fanout,
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "model": args.model_name,
                    "model_path": str(args.model_path.resolve()),
                    "request_sha256": pipeline.text_sha256(prompt),
                    "input_tokens": prompt_tokens,
                    "compact_request_used": compacted,
                    "raw_model_output": raw,
                    "label_visibility": "none",
                    "label_status": "pending",
                }
                try:
                    parsed = pipeline.extract_json(raw, "patches")
                    patches = list(parsed.get("patches") or [])
                    prompt_prime, normalized = pipeline.apply_exact_patches(
                        plan["original_prompt"], patches, set(plan["direct_target_rubric_ids"]),
                    )
                    pipeline.validate_patch_result(plan, prompt_prime, normalized)
                    record = {
                        **common,
                        "status": "ok",
                        "prompt_prime": prompt_prime,
                        "prompt_prime_sha256": pipeline.text_sha256(prompt_prime),
                        "patches": normalized,
                        "outside_patch_exact": True,
                    }
                    completed[str(plan["sample_id"])] = record
                    attempt_records.append(record)
                    accepted = True
                    break
                except Exception as exc:
                    last_error = repr(exc)
                    attempt_records.append({**common, "error": last_error})
            if not accepted:
                next_attempt = base_attempt + consumed
                if next_attempt <= args.retries:
                    work.append({
                        "plan": plan,
                        "previous_error": specialized_feedback(last_error),
                        "attempt": next_attempt,
                    })
        append_jsonl_batch(output, attempt_records)

    remaining = len(selected) - len(
        valid_successes_across(shard_attempt_paths(run, args.shard_id), plan_by_id)
    )
    print(json.dumps({
        "shard": args.shard_id,
        "selected": len(selected),
        "completed": len(selected) - remaining,
        "remaining": remaining,
        "output": str(output),
    }, ensure_ascii=False))
    if remaining and not args.allow_partial:
        raise SystemExit(2)


def merge(args: argparse.Namespace) -> None:
    run = args.run.resolve()
    plan_path = run / "plan/counterfactual_plan.jsonl"
    plan_rows = sorted(read_jsonl(plan_path), key=lambda row: int(row["execution_rank"]))
    plan_by_id = {str(row["sample_id"]): row for row in plan_rows}
    successes: dict[str, dict[str, Any]] = {}
    attempt_counts: Counter[str] = Counter()
    inputs: list[dict[str, Any]] = []
    for shard_id in range(args.num_shards):
        for path in shard_attempt_paths(run, shard_id):
            rows = read_jsonl(path)
            inputs.append({"path": str(path), "rows": len(rows), "sha256": file_sha256(path)})
            for row in rows:
                attempt_counts[str(row.get("status"))] += 1
                sample_id = str(row.get("sample_id") or "")
                if row.get("status") != "ok" or sample_id not in plan_by_id:
                    continue
                if int(plan_by_id[sample_id]["execution_rank"]) % args.num_shards != shard_id:
                    raise ValueError(f"sample {sample_id} appears in wrong shard {shard_id}")
                pipeline.validate_stored_patch(plan_by_id[sample_id], row)
                successes[sample_id] = row

    missing = [row["sample_id"] for row in plan_rows if row["sample_id"] not in successes]
    if missing and not args.allow_partial:
        raise ValueError(f"cannot merge: {len(missing)} of {len(plan_rows)} samples are incomplete")
    canonical_rows = [successes[row["sample_id"]] for row in plan_rows if row["sample_id"] in successes]
    canonical = run / "model/patch_attempts.jsonl"
    atomic_write(
        canonical,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in canonical_rows),
    )
    summary = {
        "timestamp": now(),
        "backend": BACKEND,
        "num_shards": args.num_shards,
        "planned": len(plan_rows),
        "merged_successes": len(canonical_rows),
        "missing": len(missing),
        "missing_sample_ids": missing,
        "attempt_status_counts": dict(attempt_counts),
        "inputs": inputs,
        "canonical_path": str(canonical),
        "canonical_sha256": file_sha256(canonical),
        "runner_sha256": file_sha256(Path(__file__)),
    }
    summary_path = run / "model/offline_merge_summary.json"
    atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    run = args.run.resolve()
    plan_rows = read_jsonl(run / "plan/counterfactual_plan.jsonl")
    plan_by_id = {str(row["sample_id"]): row for row in plan_rows}
    payload: dict[str, Any] = {"planned": len(plan_rows), "num_shards": args.num_shards, "shards": []}
    total = 0
    for shard_id in range(args.num_shards):
        selected = shard_rows(plan_rows, shard_id, args.num_shards)
        selected_ids = {str(row["sample_id"]) for row in selected}
        scoped_plan = {sample_id: plan_by_id[sample_id] for sample_id in selected_ids}
        paths = shard_attempt_paths(run, shard_id)
        completed = len(valid_successes_across(paths, scoped_plan))
        total += completed
        payload["shards"].append({
            "shard_id": shard_id,
            "planned": len(selected),
            "completed": completed,
            "remaining": len(selected) - completed,
            "attempt_records": sum(len(read_jsonl(path)) for path in paths),
            "checkpoint_files": len(paths),
        })
    payload["completed"] = total
    payload["remaining"] = len(plan_rows) - total
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-shard")
    run.add_argument("--run", type=Path, required=True)
    run.add_argument("--shard-id", type=int, required=True)
    run.add_argument("--num-shards", type=int, default=8)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--model-name", required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--max-num-batched-tokens", type=int, default=8192)
    run.add_argument("--max-model-len", type=int, default=16384)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--max-tokens", type=int, default=3200)
    run.add_argument("--retries", type=int, default=4)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--allow-partial", action="store_true")

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--run", type=Path, required=True)
    merge_parser.add_argument("--num-shards", type=int, default=8)
    merge_parser.add_argument("--allow-partial", action="store_true")

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--run", type=Path, required=True)
    status_parser.add_argument("--num-shards", type=int, default=8)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "run-shard":
        run_shard(args)
    elif args.command == "merge":
        merge(args)
    elif args.command == "status":
        status(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
