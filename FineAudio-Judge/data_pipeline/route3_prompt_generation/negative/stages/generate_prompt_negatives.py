#!/usr/bin/env python3
"""Localized, resumable Route 3 prompt-side negative construction with offline local vLLM.

The program fixes exact source spans and rubric membership deterministically;
the selected local text model writes replacement text for one overlapping span group at a time.
All group and sample attempts are append-only.  Audio and perceptual labels are
never read, and no HTTP/API service is used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Any

import prompt_negative_planner as planner


ROOT = Path(__file__).resolve().parent.parent
WORK = planner.WORK
GROUP_PLAN = WORK / "group_plan/groups.jsonl"
GROUP_SUMMARY = WORK / "group_plan/summary.json"
GROUP_ATTEMPTS = WORK / "model/group_attempts"
SAMPLE_ATTEMPTS = WORK / "model/grouped_sample_attempts"
MANIFEST = WORK / "manifests/prompt_side_pairs.jsonl"
SCHEMA_VERSION = "route3-prompt-negative-grouped"
TIME_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|sec(?:onds?)?|minutes?)\b|"
    r"\b(?:at|around)\s+\d{1,2}:\d{2}\b|\d+(?:\.\d+)?\s*(?:毫秒|秒|分钟)",
    re.I,
)
RELATION_FOCUS_RE = re.compile(
    r"\b(?:before|after|then|while|meanwhile|followed by|precedes|follows|overlaps?|"
    r"simultaneous(?:ly)?|foreground|background|interrupts?|together|along with)\b|"
    r"之前|之后|随后|接着|然后|同时|重叠|前景|背景|打断|只剩下|随之|交织|伴随|"
    r"结束后|才|先.+后",
    re.I,
)
CLAUSE_RE = re.compile(r"[^。！？!?；;.\n]+[。！？!?；;.]?", re.M)


def group_attempt_path(shard_id: int) -> Path:
    return GROUP_ATTEMPTS / f"shard_{shard_id:02d}.jsonl"


def sample_attempt_path(shard_id: int) -> Path:
    return SAMPLE_ATTEMPTS / f"shard_{shard_id:02d}.jsonl"


def append_jsonl_batch(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append one inference batch with a single flush/fsync."""
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


def specialized_feedback(error: str | None) -> str | None:
    if not error:
        return None
    mappings = (
        ("relation_not_explicit", "Use an explicit before/after, overlap/separation, foreground/background, interruption, or same/different-source marker and keep both event identities."),
        ("event_too_lexically_similar", "Choose a completely different coarse sound source and action; avoid reusing the original event nouns and verbs."),
        ("absence_shortcut", "Describe a present alternative sound; never negate, remove, stop, silence, or make the target inaudible."),
        ("music_genre_near_neighbor", "Replace music with a clearly non-music coarse sound event, not another genre or instrument style."),
        ("empty_or_unchanged", "Return a non-empty replacement that is plainly different from TARGET SPAN."),
        ("no JSON object", "Return exactly one JSON object with replacement_span, contradiction_explanation, and audible_test."),
        ("replacement_contains_ellipsis", "Write the complete replacement literally without ellipses."),
        ("visual_leakage", "Use only audible wording and remove all visual or camera language."),
        ("timestamp_leakage", "Remove numeric timing and express only a qualitative audible relation."),
        ("replacement_too_long", "Make the replacement a compact local clause that fits the adjacent context."),
        ("event_family", "Return valid original_event_family and replacement_event_family values; attributes preserve the family and events change it."),
        ("relation_endpoints", "Return exactly two distinct audible endpoint anchors that occur verbatim in the completed P PRIME."),
    )
    guidance = [instruction for marker, instruction in mappings if marker in error]
    if not guidance:
        guidance = ["Repair exactly this validator defect while preserving every other hard rule."]
    return f"{error}\nTARGETED REPAIR: {' '.join(guidance)}"


def _unique_span(source: str, basis: str, rubric_id: str) -> tuple[int, int]:
    if not basis or source.count(basis) != 1:
        raise ValueError(f"{rubric_id}:basis_not_unique_verbatim")
    start = source.index(basis)
    return start, start + len(basis)


def focus_span(source: str, rubric: dict[str, Any]) -> tuple[int, int]:
    """Return a deterministic, local source interval for one direct rubric."""
    rubric_id = str(rubric["rubric_id"])
    basis = str(rubric.get("basis") or "")
    basis_start, basis_end = _unique_span(source, basis, rubric_id)
    if planner.route1.rubric_kind(rubric) != "relation":
        return basis_start, basis_end
    candidates = []
    for match in CLAUSE_RE.finditer(basis):
        clause = match.group(0).strip()
        if not clause or not RELATION_FOCUS_RE.search(clause):
            continue
        relative = basis.find(clause, match.start())
        candidates.append((len(clause), -relative, relative, clause))
    if not candidates:
        return basis_start, basis_end
    _, _, relative, clause = min(candidates)
    return basis_start + relative, basis_start + relative + len(clause)


def make_groups(plan: dict[str, Any]) -> list[dict[str, Any]]:
    source = plan["original_prompt"]
    intervals: list[dict[str, Any]] = []
    for rubric in plan["direct_target_rubrics"]:
        start, end = focus_span(source, rubric)
        intervals.append({"start": start, "end": end, "rubric": rubric})
    intervals.sort(key=lambda x: (x["start"], x["end"], str(x["rubric"]["rubric_id"])))
    merged: list[dict[str, Any]] = []
    for interval in intervals:
        if merged and interval["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
            merged[-1]["rubrics"].append(interval["rubric"])
        else:
            merged.append({
                "start": interval["start"], "end": interval["end"],
                "rubrics": [interval["rubric"]],
            })
    result: list[dict[str, Any]] = []
    for index, item in enumerate(merged):
        rubrics = sorted(item["rubrics"], key=lambda x: str(x["rubric_id"]))
        kinds = sorted({planner.route1.rubric_kind(x) for x in rubrics} - {None})
        rubric_ids = [str(x["rubric_id"]) for x in rubrics]
        start, end = int(item["start"]), int(item["end"])
        overlap_retained = []
        for rubric in plan["retained_rubrics"]:
            basis = str(rubric.get("basis") or "")
            if not basis or basis not in source:
                continue
            cursor = 0
            while True:
                found = source.find(basis, cursor)
                if found < 0:
                    break
                if found < end and found + len(basis) > start:
                    overlap_retained.append(str(rubric["rubric_id"]))
                    break
                cursor = found + 1
        result.append({
            "schema_version": SCHEMA_VERSION,
            "sample_id": plan["sample_id"],
            "parent_positive_id": plan["parent_positive_id"],
            "execution_rank": plan["execution_rank"],
            "group_index": index,
            "group_id": f"{plan['sample_id']}::g{index:02d}",
            "language": plan["language"],
            "target_band": plan["target_band"],
            "start_char": start,
            "end_char": end,
            "original_span": source[start:end],
            "left_context": source[max(0, start - 220):start],
            "right_context": source[end:end + 220],
            "rubric_ids": rubric_ids,
            "rubrics": rubrics,
            "expected_kinds": kinds,
            "overlap_retained_rubric_ids": sorted(set(overlap_retained)),
        })
    actual_ids = [rid for group in result for rid in group["rubric_ids"]]
    expected_ids = list(plan["direct_target_rubric_ids"])
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ValueError(f"group_union_mismatch:{actual_ids}!={expected_ids}")
    for left, right in zip(result, result[1:]):
        if int(left["end_char"]) > int(right["start_char"]):
            raise ValueError("group_overlap_after_merge")
    return result


def prepare_groups(_: argparse.Namespace) -> None:
    plans = planner.read_jsonl(planner.PLAN)
    # The original production run contained 500 rows, while traceable repair
    # batches intentionally contain only the persistent failures being
    # replaced.  Cardinality belongs to the caller's plan, not this generic
    # grouping primitive.
    if not plans:
        raise ValueError("expected a non-empty planner plan")
    groups = [group for plan in plans for group in make_groups(plan)]
    planner.write_jsonl(GROUP_PLAN, groups)
    summary = {
        "timestamp": planner.now(), "schema_version": SCHEMA_VERSION,
        "samples": len(plans), "groups": len(groups),
        "groups_per_sample": dict(Counter(str(len(make_groups(p))) for p in plans)),
        "kind_sets": dict(Counter("+".join(g["expected_kinds"]) for g in groups)),
        "groups_with_retained_overlap": sum(bool(g["overlap_retained_rubric_ids"]) for g in groups),
        "plan": str(planner.PLAN), "plan_sha256": planner.file_sha256(planner.PLAN),
        "group_plan": str(GROUP_PLAN), "group_plan_sha256": planner.file_sha256(GROUP_PLAN),
        "labels_read": False, "audio_model_used": False, "http_api_used": False,
    }
    planner.atomic_write(GROUP_SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def generation_prompt(group: dict[str, Any], feedback: str | None) -> str:
    retry = f"\nVALIDATOR FAILURE FROM THE LAST TRY:\n{feedback}\nRepair it.\n" if feedback else ""
    kinds = ", ".join(group["expected_kinds"])
    event_only = ""
    if set(group["expected_kinds"]) == {"event"}:
        event_only = """
EVENT-ONLY HARD-TAIL OVERRIDE:
- Replace the whole target span with one unrelated environmental or mechanical sound event.
- Do not preserve the original speaker, subject, action, object, quoted words, or their synonyms.
- Good coarse alternatives include a thunderclap, a vehicle horn, glass shattering, an engine revving,
  a fire alarm, or heavy machinery; express the chosen alternative naturally in the source language.
- Do not merely change what the same person says or how the same object sounds.
"""
    relation = ""
    if "relation" in set(group["expected_kinds"]):
        relation = """
RELATION HARD-TAIL OVERRIDE:
- Keep both related sounds present and reverse their order, overlap, or foreground/background relation.
- For a sound that originally continues after another, place that sound entirely before the other event.
- Never express a relation reversal by saying a sound stops, ceases, disappears, becomes silent, or is absent.
"""
    return f"""Rewrite only TARGET SPAN to create an audible counterfactual. Return JSON only.

The source P is assumed positive. Audio and labels are unavailable. The program, not you, fixes
the span and rubric IDs. Write only a replacement for this one span in {group['language']}.

Required edit kinds: {kinds}
Target rubrics:
{json.dumps([planner.route1.compact_rubric(x) for x in group['rubrics']], ensure_ascii=False)}

Hard rules:
- Contradict every target rubric in this group.
- event: use a clearly different coarse audible event/source/action, not a nearby subtype.
- attribute: keep event identity but use a mutually exclusive audible property.
- relation: keep event identities and explicitly reverse before/after, overlap/separation,
  foreground/background, interruption, or same/different source.
- Never use absence, silence, no/not/without, 没有/不再/未出现/停止/消失.
- Add no visual fact, timestamp, causality, genre swap, accent swap, or unverifiable context.
- Keep the replacement compact and grammatical between LEFT and RIGHT context.
- For event or attribute edits, report original_event_family and replacement_event_family using
  speech|music|sound_effect|vocalization|animal|ambience|mechanical|vehicle|impact|footsteps|other.
  Attribute edits preserve the family; event edits change it.
- For relation edits, report exactly two relation_endpoints. Each endpoint must be an audible
  event anchor that will occur verbatim in the completed P PRIME. Otherwise return an empty list.
{event_only}
{relation}

LEFT CONTEXT:
{group['left_context']}
TARGET SPAN (replace all of this):
{group['original_span']}
RIGHT CONTEXT:
{group['right_context']}
{retry}
Return exactly:
{{"replacement_span":"...","original_event_family":"... or null",
"replacement_event_family":"... or null","relation_endpoints":["...","..."],
"contradiction_explanation":"...","audible_test":"..."}}"""


def validate_group(group: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    replacement = str(value.get("replacement_span") or "").strip()
    explanation = str(value.get("contradiction_explanation") or "").strip()
    audible_test = str(value.get("audible_test") or "").strip()
    original = str(group["original_span"])
    if not replacement or replacement == original:
        raise ValueError("empty_or_unchanged_replacement")
    original_ellipses = Counter({"...": original.count("..."), "…": original.count("…")})
    replacement_ellipses = Counter({
        "...": replacement.count("..."), "…": replacement.count("…"),
    })
    if replacement_ellipses - original_ellipses:
        raise ValueError("replacement_contains_ellipsis")
    if len(replacement) > max(500, 3 * len(original)):
        raise ValueError("replacement_too_long")
    # Preserve an absence-like token when it was already literal source text
    # (for example the station name phrase "Next stop").  Reject only newly
    # introduced occurrences; otherwise some valid relation flips are
    # impossible even though no shortcut was added by the model.
    original_absence = Counter(
        match.group(0).casefold() for match in planner.UNIVERSAL_ABSENCE.finditer(original)
    )
    replacement_absence = Counter(
        match.group(0).casefold() for match in planner.UNIVERSAL_ABSENCE.finditer(replacement)
    )
    introduced_absence = replacement_absence - original_absence
    if introduced_absence:
        raise ValueError(f"absence_shortcut:{next(iter(introduced_absence))!r}")
    visual = planner.route1.VISUAL_RE.search(replacement) or planner.route1.VISUAL_RE.search(audible_test)
    if visual:
        raise ValueError(f"visual_leakage:{visual.group(0)!r}")
    timestamp = TIME_RE.search(replacement)
    if timestamp:
        raise ValueError(f"timestamp_leakage:{timestamp.group(0)!r}")
    if not explanation or not audible_test:
        raise ValueError("missing_explanation_or_audible_test")
    reason = planner.route1.near_neighbor_reason(original, replacement)
    if reason:
        raise ValueError(reason)
    kinds = set(group["expected_kinds"])
    original_family = None
    replacement_family = None
    if kinds & {"event", "attribute"}:
        original_family = planner.route1.canonical_event_family(
            value.get("original_event_family")
        )
        replacement_family = planner.route1.canonical_event_family(
            value.get("replacement_event_family")
        )
        if not original_family or not replacement_family:
            raise ValueError("missing_or_invalid_event_family")
        if "event" not in kinds and original_family != replacement_family:
            raise ValueError("attribute_changes_event_family")
        if "event" in kinds and original_family == replacement_family:
            raise ValueError("event_keeps_same_event_family")
    relation_endpoints = [
        str(item).strip() for item in value.get("relation_endpoints") or []
    ]
    if "relation" in kinds:
        if (
            len(relation_endpoints) != 2
            or len(set(relation_endpoints)) != 2
            or any(not item for item in relation_endpoints)
        ):
            raise ValueError("invalid_relation_endpoints")
    else:
        relation_endpoints = []
    if kinds == {"event"}:
        similarity = planner.lexical_similarity(original, replacement)
        if similarity > 0.55:
            raise ValueError(f"event_too_lexically_similar:{similarity:.3f}")
    if "relation" in kinds and not planner.relation_is_explicit(original, replacement):
        raise ValueError("relation_not_explicit")
    return {
        "replacement_span": replacement,
        "original_event_family": original_family,
        "replacement_event_family": replacement_family,
        "relation_endpoints": relation_endpoints,
        "contradiction_explanation": explanation,
        "audible_test": audible_test,
    }


def _latest_rows(path: Path, key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in planner.read_jsonl(path):
        value = str(row.get(key) or "")
        if value:
            latest[value] = row
    return latest


def _valid_group_successes(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for group_id, row in _latest_rows(path, "group_id").items():
        if row.get("status") == "ok":
            result[group_id] = row
    return result


def _edit_kind(group: dict[str, Any]) -> str:
    kinds = list(group["expected_kinds"])
    return kinds[0] if len(kinds) == 1 else "mixed"


def tail_candidate_fanout(
    active_count: int, batch_size: int, remaining_attempts: int, maximum: int = 8,
) -> int:
    """Use otherwise-idle decode slots for independent hard-tail candidates."""
    if active_count < 1 or batch_size < 1 or remaining_attempts < 1:
        raise ValueError("fanout inputs must be positive")
    return max(1, min(maximum, batch_size // active_count, remaining_attempts))


def build_sample(
    plan: dict[str, Any], groups: list[dict[str, Any]], generated: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    raw_patches = []
    for group in groups:
        row = generated[group["group_id"]]
        raw_patches.append({
            "rubric_ids": group["rubric_ids"],
            "original_span": group["original_span"],
            "replacement_span": row["replacement_span"],
            "edit_kind": _edit_kind(group),
            "semantic_distance": "high",
            "contradiction_explanation": (
                f"{' '.join(group['rubric_ids'])}: {row['contradiction_explanation']}"
            ),
            "audible_test": row["audible_test"],
            "original_event_family": row.get("original_event_family"),
            "replacement_event_family": row.get("replacement_event_family"),
            "relation_endpoints": row.get("relation_endpoints") or [],
            "group_id": group["group_id"],
        })
    prime, patches = planner.route1.apply_exact_patches(
        plan["original_prompt"], raw_patches, set(plan["direct_target_rubric_ids"]),
    )
    planner.validate_counterfactual_patch_result(plan, prime, patches)
    return prime, patches


def effective_sets(
    plan: dict[str, Any], patches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    affected_ids = set(plan["hierarchy_affected_rubric_ids"])
    for rubric in plan["retained_rubrics"]:
        if planner.route1.rubric_basis_overlaps_patches(rubric, plan["original_prompt"], patches):
            affected_ids.add(str(rubric["rubric_id"]))
    all_map = {str(x["rubric_id"]): x for x in plan["all_original_rubrics"]}
    affected = [all_map[x] for x in sorted(affected_ids)]
    retained = [all_map[x] for x in sorted(set(all_map) - affected_ids)]
    return affected, retained


def construction_audit(patches: list[dict[str, Any]]) -> dict[str, Any]:
    """Record deterministic direct-ID coverage; this is not an audio label."""
    direct = []
    for patch in patches:
        evidence = str(patch["replacement_span"])
        for rubric_id in patch["rubric_ids"]:
            direct.append({
                "rubric_id": str(rubric_id), "contradicted": True,
                "evidence": evidence, "basis": "controlled_exact_patch",
            })
    direct.sort(key=lambda x: x["rubric_id"])
    return {"direct": direct, "audit_type": "deterministic_construction_coverage_not_audio_label"}


def validate_sample_success(plan: dict[str, Any], row: dict[str, Any]) -> None:
    planner.validate_success(plan, row)


def sample_successes(path: Path, plans: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for sample_id, row in _latest_rows(path, "sample_id").items():
        if row.get("status") != "ok" or sample_id not in plans:
            continue
        try:
            validate_sample_success(plans[sample_id], row)
        except Exception:
            continue
        result[sample_id] = row
    return result


def run_shard(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    plans_all = planner.read_jsonl(planner.PLAN)
    selected = planner.shard_rows(plans_all, args.shard_id, args.num_shards)
    if args.limit_samples:
        selected = selected[:args.limit_samples]
    plans = {str(x["sample_id"]): x for x in selected}
    groups_all = [g for g in planner.read_jsonl(GROUP_PLAN) if str(g["sample_id"]) in plans]
    by_sample: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in plans}
    for group in groups_all:
        by_sample[str(group["sample_id"])].append(group)
    group_path = group_attempt_path(args.shard_id)
    sample_path = sample_attempt_path(args.shard_id)
    completed_samples = sample_successes(sample_path, plans)
    pending_ids = [x["sample_id"] for x in selected if x["sample_id"] not in completed_samples]
    latest_samples = _latest_rows(sample_path, "sample_id")
    generated = _valid_group_successes(group_path)
    # Missing-group assembly failures are monotonic: preserve every valid
    # group and retry only the missing ones.  Reconstruct all groups only for
    # a genuine post-assembly structural failure.
    for sample_id in pending_ids:
        latest = latest_samples.get(sample_id)
        if (
            latest
            and latest.get("status") != "ok"
            and not (
                latest.get("stage") == "assemble"
                and str(latest.get("error") or "").startswith("missing_group_successes:")
            )
        ):
            for group in by_sample[sample_id]:
                generated.pop(group["group_id"], None)
    pending_groups = [
        group for sample_id in pending_ids for group in by_sample[sample_id]
        if group["group_id"] not in generated
    ]
    run_id = str(uuid.uuid4())
    metadata = {
        "timestamp": planner.now(), "run_id": run_id, "schema_version": SCHEMA_VERSION,
        "backend": planner.BACKEND, "model": planner.MODEL_NAME,
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": planner.file_sha256(args.model_path / "config.json"),
        "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256": planner.file_sha256(Path(__file__)),
        "validator_sha256": planner.file_sha256(Path(planner.__file__)),
        "plan_sha256": planner.file_sha256(planner.PLAN), "group_plan_sha256": planner.file_sha256(GROUP_PLAN),
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "selected_samples": len(selected), "completed_samples": len(completed_samples),
        "pending_samples": len(pending_ids), "pending_groups": len(pending_groups),
        "batch_size": args.batch_size, "retries": args.retries,
        "temperature": args.temperature,
        "labels_read": False, "audio_model_used": False, "http_api_used": False,
    }
    planner.atomic_write(
        group_path.with_suffix(".metadata.json"),
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    if not pending_ids:
        print(json.dumps({"shard": args.shard_id, "completed": len(completed_samples), "remaining": 0}))
        return
    llm = LLM(
        model=str(args.model_path), trust_remote_code=True, dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True, safetensors_load_strategy="eager", skip_mm_profiling=True,
    )
    tokenizer = llm.get_tokenizer()
    audit_sampling = SamplingParams(temperature=0.0, max_tokens=args.audit_max_tokens)
    max_input = args.max_model_len - max(args.max_tokens, args.audit_max_tokens) - 32
    invalidated: set[str] = set()
    for sample_id in pending_ids:
        latest = latest_samples.get(sample_id)
        if (
            latest
            and latest.get("status") != "ok"
            and not (
                latest.get("stage") == "assemble"
                and str(latest.get("error") or "").startswith("missing_group_successes:")
            )
        ):
            invalidated.update(group["group_id"] for group in by_sample[sample_id])

    disk_generated = _valid_group_successes(group_path)
    generated.update({k: value for k, value in disk_generated.items() if k not in invalidated})
    pending_groups = [
        group for sample_id in pending_ids for group in by_sample[sample_id]
        if group["group_id"] not in generated
    ]
    latest_groups = _latest_rows(group_path, "group_id")
    max_attempts = args.repair_rounds * args.retries
    work = deque({
        "group": group,
        "feedback": specialized_feedback(
            str(latest_groups.get(group["group_id"], {}).get("error") or "")
        ),
        "total_attempt": 1,
    } for group in pending_groups)
    # Retry failures at the back of one global queue.  This prevents a fixed
    # batch from collapsing after its easy groups succeed and keeps vLLM fed
    # with a full batch throughout the hard tail.
    while work:
        active = [work.popleft() for _ in range(min(args.batch_size, len(work)))]
        prompts = [generation_prompt(x["group"], x["feedback"]) for x in active]
        sizes = [planner.chat_tokens(tokenizer, x) for x in prompts]
        if any(x > max_input for x in sizes):
            raise ValueError(f"group prompt exceeds context:{max(sizes)}>{max_input}")
        fanout = tail_candidate_fanout(
            len(active), args.batch_size,
            min(max_attempts - int(x["total_attempt"]) + 1 for x in active),
        )
        generate_sampling = SamplingParams(
            temperature=args.temperature, max_tokens=args.max_tokens, n=fanout,
        )
        outputs = llm.chat(
            [[{"role": "user", "content": p}] for p in prompts],
            sampling_params=generate_sampling, use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        attempt_rows = []
        for state, prompt, size, output in zip(active, prompts, sizes, outputs):
            group = state["group"]
            base_attempt = int(state["total_attempt"])
            candidates_out = list(getattr(output, "outputs", None) or [])[:fanout]
            if not candidates_out:
                raise ValueError("vLLM returned no candidate output")
            accepted = False
            last_error = ""
            consumed = 0
            for candidate_index, candidate in enumerate(candidates_out):
                consumed += 1
                total_attempt = base_attempt + candidate_index
                repair_round = (total_attempt - 1) // args.retries + 1
                attempt = (total_attempt - 1) % args.retries + 1
                raw = str(candidate.text or "")
                common = {
                    "timestamp": planner.now(), "run_id": run_id, "schema_version": SCHEMA_VERSION,
                    "group_id": group["group_id"], "sample_id": group["sample_id"],
                    "repair_round": repair_round, "attempt": attempt,
                    "candidate_index": candidate_index, "candidate_fanout": fanout,
                    "shard_id": args.shard_id, "model": planner.MODEL_NAME,
                    "backend": planner.BACKEND,
                    "generation_request_sha256": planner.sha256_text(prompt),
                    "input_tokens": size, "raw_model_output": raw,
                    "label_visibility": "none", "label_status": "pending",
                }
                try:
                    value = validate_group(group, planner.extract_json(raw, "replacement_span"))
                    row = {**common, "stage": "group_generate", "status": "ok", **value}
                    generated[group["group_id"]] = row
                    invalidated.discard(group["group_id"])
                    attempt_rows.append(row)
                    accepted = True
                    break
                except Exception as exc:
                    last_error = repr(exc)
                    attempt_rows.append({
                        **common, "stage": "group_generate", "status": "error",
                        "error": last_error,
                    })
            if not accepted:
                generated.pop(group["group_id"], None)
                next_attempt = base_attempt + consumed
                if next_attempt <= max_attempts:
                    work.append({
                        "group": group,
                        "feedback": specialized_feedback(last_error),
                        "total_attempt": next_attempt,
                    })
        append_jsonl_batch(group_path, attempt_rows)

    generated = {
        k: value for k, value in _valid_group_successes(group_path).items()
        if k not in invalidated
    }

    candidates = []
    assembly_errors = []
    for sample_id in pending_ids:
        plan = plans[sample_id]
        groups = by_sample[sample_id]
        missing = [g["group_id"] for g in groups if g["group_id"] not in generated]
        if missing:
            assembly_errors.append({
                "timestamp": planner.now(), "run_id": run_id, "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id, "stage": "assemble", "status": "error",
                "error": f"missing_group_successes:{missing}",
            })
            continue
        try:
            prime, patches = build_sample(plan, groups, generated)
            candidates.append({"plan": plan, "prime": prime, "patches": patches})
        except Exception as exc:
            assembly_errors.append({
                "timestamp": planner.now(), "run_id": run_id, "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id, "stage": "assemble", "status": "error", "error": repr(exc),
            })
    append_jsonl_batch(sample_path, assembly_errors)

    if args.skip_model_audit:
        accepted_rows = []
        for item in candidates:
            plan, prime, patches = item["plan"], item["prime"], item["patches"]
            affected, retained = effective_sets(plan, patches)
            row = {
                "timestamp": planner.now(), "run_id": run_id, "schema_version": SCHEMA_VERSION,
                "sample_id": plan["sample_id"], "parent_positive_id": plan["parent_positive_id"],
                "stage": "deterministic_construction_audit", "status": "ok",
                "model": planner.MODEL_NAME, "backend": planner.BACKEND,
                "prompt_prime": prime, "patches": patches,
                "semantic_audit": construction_audit(patches),
                "model_semantic_audit": None, "model_semantic_audit_valid": None,
                "model_semantic_audit_warning": "skipped_pending_post_generation_labeling",
                "prompt_prime_sha256": planner.sha256_text(prime),
                "effective_affected_rubrics": affected,
                "effective_retained_rubrics": retained,
                "outside_patch_exact": True, "audio_byte_identical_to_parent": True,
                "label_visibility": "none", "label_status": "pending",
            }
            validate_sample_success(plan, row)
            accepted_rows.append(row)
        append_jsonl_batch(sample_path, accepted_rows)
    for batch_start in range(0 if not args.skip_model_audit else len(candidates), len(candidates), args.audit_batch_size):
        batch = candidates[batch_start:batch_start + args.audit_batch_size]
        active = [{"item": x, "feedback": None} for x in batch]
        for audit_attempt in range(1, args.audit_retries + 1):
            if not active:
                break
            prompts = [planner.audit_prompt(x["item"]["plan"], x["item"]["prime"], x["item"]["patches"]) for x in active]
            sizes = [planner.chat_tokens(tokenizer, x) for x in prompts]
            if any(x > max_input for x in sizes):
                raise ValueError(f"audit prompt exceeds context:{max(sizes)}>{max_input}")
            outputs = llm.chat(
                [[{"role": "user", "content": p}] for p in prompts],
                sampling_params=audit_sampling, use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            retry = []
            audit_rows = []
            for state, prompt, size, output in zip(active, prompts, sizes, outputs):
                item = state["item"]
                plan, prime, patches = item["plan"], item["prime"], item["patches"]
                raw = planner.output_text(output)
                common = {
                    "timestamp": planner.now(), "run_id": run_id, "schema_version": SCHEMA_VERSION,
                    "sample_id": plan["sample_id"], "parent_positive_id": plan["parent_positive_id"],
                    "stage": "semantic_audit", "audit_attempt": audit_attempt,
                    "model": planner.MODEL_NAME, "backend": planner.BACKEND,
                    "audit_request_sha256": planner.sha256_text(prompt), "audit_input_tokens": size,
                    "raw_audit_output": raw, "prompt_prime": prime, "patches": patches,
                    "label_visibility": "none", "label_status": "pending",
                }
                model_audit = None
                model_audit_valid = False
                model_audit_warning = None
                try:
                    model_audit = planner.extract_json(raw, "direct")
                    planner.validate_audit(plan, prime, patches, model_audit)
                    model_audit_valid = True
                except Exception as exc:
                    # The requested post-generation audio labeling is a later
                    # stage.  Preserve this text-only model opinion as a
                    # diagnostic, but do not let omitted IDs masquerade as a
                    # construction failure or prevent the 500-row run.
                    model_audit_warning = repr(exc)
                audit = construction_audit(patches)
                affected, retained = effective_sets(plan, patches)
                row = {
                    **common, "status": "ok", "semantic_audit": audit,
                    "model_semantic_audit": model_audit,
                    "model_semantic_audit_valid": model_audit_valid,
                    "model_semantic_audit_warning": model_audit_warning,
                    "prompt_prime_sha256": planner.sha256_text(prime),
                    "effective_affected_rubrics": affected,
                    "effective_retained_rubrics": retained,
                    "outside_patch_exact": True, "audio_byte_identical_to_parent": True,
                }
                validate_sample_success(plan, row)
                audit_rows.append(row)
            append_jsonl_batch(sample_path, audit_rows)
            active = retry
    complete = sample_successes(sample_path, plans)
    payload = {
        "shard": args.shard_id, "planned": len(selected), "completed": len(complete),
        "remaining": len(selected) - len(complete), "group_attempts": str(group_path),
        "sample_attempts": str(sample_path),
    }
    print(json.dumps(payload, ensure_ascii=False))
    if payload["remaining"] and not args.allow_partial:
        raise SystemExit(2)


def status(args: argparse.Namespace) -> None:
    plans = planner.read_jsonl(planner.PLAN)
    total = 0
    shards = []
    for shard_id in range(args.num_shards):
        selected = planner.shard_rows(plans, shard_id, args.num_shards)
        scoped = {str(x["sample_id"]): x for x in selected}
        complete = len(sample_successes(sample_attempt_path(shard_id), scoped))
        total += complete
        shards.append({
            "shard_id": shard_id, "planned": len(selected), "completed": complete,
            "remaining": len(selected) - complete,
            "group_attempt_records": len(planner.read_jsonl(group_attempt_path(shard_id))),
            "sample_attempt_records": len(planner.read_jsonl(sample_attempt_path(shard_id))),
        })
    print(json.dumps({
        "planned": len(plans), "completed": total, "remaining": len(plans) - total,
        "shards": shards,
    }, ensure_ascii=False, indent=2))


def merge(args: argparse.Namespace) -> None:
    plans = planner.read_jsonl(planner.PLAN)
    plan_map = {str(x["sample_id"]): x for x in plans}
    accepted: dict[str, dict[str, Any]] = {}
    inputs = []
    for shard_id in range(args.num_shards):
        path = sample_attempt_path(shard_id)
        rows = planner.read_jsonl(path)
        accepted.update(sample_successes(path, plan_map))
        inputs.append({
            "path": str(path), "rows": len(rows),
            "sha256": planner.file_sha256(path) if path.is_file() else None,
        })
    missing = [x["sample_id"] for x in plans if x["sample_id"] not in accepted]
    if missing and not args.allow_partial:
        raise ValueError(f"cannot merge incomplete grouped run:{len(accepted)}/{len(plans)}")
    pairs = []
    for plan in sorted(plans, key=lambda x: int(x["execution_rank"])):
        row = accepted.get(plan["sample_id"])
        if row is None:
            continue
        pairs.append({
            "schema_version": SCHEMA_VERSION,
            "sample_id": plan["sample_id"], "source_pair_id": plan["parent_positive_id"],
            "language": plan["language"], "target_band": plan["target_band"],
            "pair_type": "P_prime_A", "route": "route3_prompt_side_negative",
            "original_prompt": plan["original_prompt"], "prompt_prime": row["prompt_prime"],
            "source_audio_path": plan["audio_path"], "source_audio_sha256": plan["audio_sha256"],
            "evaluation_prompt": row["prompt_prime"],
            "evaluation_audio_path": plan["audio_path"],
            "rubric_source_prompt": row["prompt_prime"],
            "rubric_source_prompt_sha256": planner.sha256_text(row["prompt_prime"]),
            "evaluation_rubrics_status": "pending_regeneration_from_P_prime",
            "direct_target_rubrics": plan["direct_target_rubrics"],
            "affected_rubrics": row["effective_affected_rubrics"],
            "retained_rubrics": row["effective_retained_rubrics"],
            "patches": row["patches"], "semantic_audit": row["semantic_audit"],
            "model_semantic_audit": row.get("model_semantic_audit"),
            "model_semantic_audit_valid": row.get("model_semantic_audit_valid"),
            "model_semantic_audit_warning": row.get("model_semantic_audit_warning"),
            "outside_patch_exact": True, "audio_byte_identical_to_parent": True,
            "label_visibility": "none", "label_status": "pending",
            "generator": {"backend": planner.BACKEND, "model": planner.MODEL_NAME, "run_id": row["run_id"]},
        })
    planner.write_jsonl(MANIFEST, pairs)
    summary = {
        "timestamp": planner.now(), "schema_version": SCHEMA_VERSION,
        "planned": len(plans), "accepted": len(pairs), "missing": len(missing),
        "missing_sample_ids": missing, "output": str(MANIFEST),
        "output_sha256": planner.file_sha256(MANIFEST),
        "distribution": dict(Counter(f"{x['language']}|{x['target_band']}" for x in pairs)),
        "plan_sha256": planner.file_sha256(planner.PLAN), "group_plan_sha256": planner.file_sha256(GROUP_PLAN),
        "script_sha256": planner.file_sha256(Path(__file__)), "inputs": inputs,
        "labels_read": False, "audio_model_used": False, "http_api_used": False,
        "legacy_output_included": False, "one_shot_output_included": False,
    }
    planner.atomic_write(WORK / "manifests/summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-groups")
    run = sub.add_parser("run-shard")
    run.add_argument("--shard-id", type=int, required=True)
    run.add_argument("--num-shards", type=int, default=2)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--audit-batch-size", type=int, default=16)
    run.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run.add_argument("--max-model-len", type=int, default=8192)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    run.add_argument("--max-tokens", type=int, default=900)
    run.add_argument("--temperature", type=float, default=0.15)
    run.add_argument("--audit-max-tokens", type=int, default=1800)
    run.add_argument("--retries", type=int, default=4)
    run.add_argument("--repair-rounds", type=int, default=8)
    run.add_argument("--audit-retries", type=int, default=2)
    run.add_argument("--skip-model-audit", action="store_true")
    run.add_argument("--limit-samples", type=int, default=0)
    run.add_argument("--allow-partial", action="store_true")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--num-shards", type=int, default=2)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--num-shards", type=int, default=2)
    merge_parser.add_argument("--allow-partial", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-groups":
        prepare_groups(args)
    elif args.command == "run-shard":
        run_shard(args)
    elif args.command == "status":
        status(args)
    elif args.command == "merge":
        merge(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
