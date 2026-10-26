#!/usr/bin/env python3
"""Dependency-aware Route 3 prompt-side negative generation with local offline vLLM.

The source pair is assumed positive by construction.  This program never reads
perceptual labels, never edits audio, and never calls an HTTP API.  It borrows
Path1's direct-target/dependency-closure planning and exact-patch validators,
then adds a text-only semantic construction audit before accepting a sample.
All model attempts are append-only and resumable.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("FINEAUDIO_ROUTE3_NEGATIVE_RUN", Path(__file__).resolve().parent.parent))
ROUTE1_CODE = Path(__file__).resolve().parents[3] / "route1_real_world/negative/stages"
sys.path.insert(0, str(ROUTE1_CODE))
import counterfactual_pipeline as route1  # noqa: E402


SOURCE_PLAN = ROOT / "plan/prompt_side_plan.jsonl"
WORK = ROOT / "prompt_side"
PLAN = WORK / "plan/counterfactual_plan.jsonl"
MODEL_NAME = "reproducer_selected_text_model"
SCHEMA_VERSION = "route3-prompt-negative"
BACKEND = "vllm_offline"
FROZEN_SAMPLE_PREFIX = bytes.fromhex("70617468336e6567316b5f").decode("ascii")
UNIVERSAL_ABSENCE = re.compile(
    r"\b(?:no|not|without|absence|absent|omit(?:s|ted)?|silent|silence|inaudible|"
    r"stops?|ceases?|drowns?\s+out)\b|没有|不存在|不再|未出现|缺少|听不见|静音|停止|消失|掩盖",
    re.I,
)
RELATION_MARKER = re.compile(
    r"\b(?:before|after|then|followed|precedes|follows|while|overlap|simultaneous|sequential|"
    r"foreground|background|interrupt|separate|different source|another source)\b|"
    r"之前|之后|随后|接着|先.*后|同时|重叠|前景|背景|打断|分离|不同声源|另一个声源|结束后|才",
    re.I,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(value: str) -> str:
    return sha256_text(value)


def rank_sample_id(row: dict[str, Any]) -> str:
    """Return the frozen ranking ID while keeping public IDs clean."""
    sample_id = str(row["sample_id"])
    public_prefix = "route3neg_"
    if sample_id.startswith(public_prefix):
        return FROZEN_SAMPLE_PREFIX + sample_id[len(public_prefix):]
    return sample_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def descendant_set(root: str, children: dict[str, set[str]]) -> set[str]:
    found = {root}
    pending = [root]
    while pending:
        for child in children[pending.pop()]:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def band_choice(row: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Select few direct targets whose dependency closure reaches the band."""
    prompt = str(row["natural_prompt"])
    questions = {str(item["rubric_id"]): item for item in row["source_questions"]}
    children: dict[str, set[str]] = defaultdict(set)
    for question in questions.values():
        for parent in question.get("parent_ids") or []:
            if str(parent) in questions:
                children[str(parent)].add(str(question["rubric_id"]))

    candidates: list[dict[str, Any]] = []
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rubric_id, question in questions.items():
        kind = route1.rubric_kind(question)
        if kind is None or route1.target_rejection(question, prompt):
            continue
        candidate = {
            "rubric_id": rubric_id,
            "kind": kind,
            "closure": descendant_set(rubric_id, children),
        }
        candidates.append(candidate)
        by_kind[kind].append(candidate)
    for kind in by_kind:
        by_kind[kind].sort(key=lambda item: stable_key(f"{rank_sample_id(row)}:{kind}:{item['rubric_id']}"))

    required_kinds = {
        "attribute": ["attribute"],
        "relation": ["relation"],
        "event": ["event"],
        "all_three": ["event", "attribute", "relation"],
    }[category]
    if any(not by_kind[kind] for kind in required_kinds):
        return None

    fraction = 0.25 if row["target_band"] == "mid" else 0.50
    low, high = ((0.10, 0.40) if row["target_band"] == "mid" else (0.35, 0.85))
    target = max(1, round(len(questions) * fraction))
    minimum = max(1, math.ceil(len(questions) * low))
    maximum = max(minimum, math.floor(len(questions) * high))

    chosen: list[dict[str, Any]] = []
    affected: set[str] = set()
    for kind in required_kinds:
        options = []
        for candidate in by_kind[kind]:
            new_size = len(affected | candidate["closure"])
            options.append((
                max(0, new_size - target),
                abs(new_size - target),
                -len(candidate["closure"] - affected),
                stable_key(f"required:{rank_sample_id(row)}:{candidate['rubric_id']}"),
                candidate,
            ))
        candidate = min(options)[-1]
        if candidate not in chosen:
            chosen.append(candidate)
            affected.update(candidate["closure"])

    while len(affected) < target:
        options = []
        chosen_ids = {item["rubric_id"] for item in chosen}
        for candidate in candidates:
            if candidate["rubric_id"] in chosen_ids:
                continue
            gain = len(candidate["closure"] - affected)
            if not gain:
                continue
            new_size = len(affected | candidate["closure"])
            if new_size > maximum:
                continue
            options.append((
                abs(new_size - target),
                -gain,
                0 if candidate["kind"] == "event" else 1,
                stable_key(f"fill:{rank_sample_id(row)}:{candidate['rubric_id']}"),
                candidate,
            ))
        if not options:
            break
        candidate = min(options)[-1]
        chosen.append(candidate)
        affected.update(candidate["closure"])

    if not minimum <= len(affected) <= maximum:
        return None
    direct = {item["rubric_id"] for item in chosen}
    return {
        "direct_target_rubric_ids": sorted(direct),
        "hierarchy_affected_rubric_ids": sorted(affected),
        "retained_rubric_ids": sorted(set(questions) - affected),
        "direct_target_count": len(direct),
        "affected_count": len(affected),
        "total_rubrics": len(questions),
        "construction_changed_fraction": round(len(affected) / len(questions), 6),
        "intended_perturbation_strength": "medium" if row["target_band"] == "mid" else "strong",
    }


def prepare(_: argparse.Namespace) -> None:
    source_rows = read_jsonl(SOURCE_PLAN)
    if len(source_rows) != 500:
        raise ValueError(f"expected 500 Route 3 prompt-side sources, got {len(source_rows)}")
    assigned_counts: Counter[tuple[str, str, str]] = Counter()
    planned: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for language, band in (("zh", "mid"), ("zh", "low"), ("en", "mid"), ("en", "low")):
        cell = sorted(
            (row for row in source_rows if row["language"] == language and row["target_band"] == band),
            key=lambda row: stable_key(f"cell:{rank_sample_id(row)}"),
        )
        if len(cell) != 125:
            raise ValueError(f"unexpected cell size {language}/{band}: {len(cell)}")
        preferred = ("attribute", "relation") if band == "mid" else ("event", "all_three")
        for row in cell:
            options = []
            for category in preferred:
                choice = band_choice(row, category)
                if choice is not None:
                    options.append((
                        assigned_counts[(language, band, category)],
                        choice["direct_target_count"],
                        stable_key(f"assign:{rank_sample_id(row)}:{category}"),
                        category,
                        choice,
                    ))
            if not options:
                rejected.append({"sample_id": row["sample_id"], "reason": "no_dependency_safe_choice"})
                continue
            _, _, _, category, choice = min(options)
            assigned_counts[(language, band, category)] += 1
            qmap = {str(item["rubric_id"]): item for item in row["source_questions"]}
            direct = set(choice["direct_target_rubric_ids"])
            affected = set(choice["hierarchy_affected_rubric_ids"])
            planned.append({
                "schema_version": SCHEMA_VERSION,
                "sample_id": str(row["sample_id"]),
                "negative_id": str(row["sample_id"]),
                "parent_positive_id": str(row["source_pair_id"]),
                "source_group_id": str(row["source_pair_id"]),
                "route": "route3_prompt_side_negative",
                "pair_type": "P_prime_A",
                "positive_pair": False,
                "source_assumption": "positive_by_construction",
                "label_visibility": "none",
                "label_status": "pending",
                "selection_policy": "dependency_closure_no_labels",
                "disturbance_category": category,
                "target_band": band,
                "intended_perturbation_strength": choice["intended_perturbation_strength"],
                "construction_changed_fraction": choice["construction_changed_fraction"],
                "language": language,
                "original_prompt": str(row["natural_prompt"]),
                "original_prompt_sha256": sha256_text(str(row["natural_prompt"])),
                "audio_path": str(row["source_audio_path"]),
                "audio_sha256": row.get("source_audio_sha256"),
                "audio_duration_seconds": row.get("source_audio_duration_seconds"),
                "direct_target_rubric_ids": sorted(direct),
                "hierarchy_affected_rubric_ids": sorted(affected),
                "retained_rubric_ids": sorted(set(qmap) - affected),
                "direct_target_rubrics": [qmap[rid] for rid in sorted(direct)],
                "affected_rubrics": [qmap[rid] for rid in sorted(affected)],
                "retained_rubrics": [qmap[rid] for rid in sorted(set(qmap) - affected)],
                "all_original_rubrics": list(row["source_questions"]),
                "initial_changed_rubric_ids": list(row["changed_rubric_ids"]),
            })
    if rejected or len(planned) != 500:
        raise ValueError(f"planner planning incomplete: planned={len(planned)} rejected={len(rejected)}")
    planned.sort(key=lambda row: stable_key(f"execute:{rank_sample_id(row)}"))
    for execution_rank, row in enumerate(planned):
        row["execution_rank"] = execution_rank
    write_jsonl(PLAN, planned)
    write_jsonl(WORK / "plan/rejected.jsonl", rejected)
    summary = {
        "timestamp": now(),
        "planned": len(planned),
        "source_plan": str(SOURCE_PLAN),
        "source_plan_sha256": file_sha256(SOURCE_PLAN),
        "plan_sha256": file_sha256(PLAN),
        "distribution": dict(Counter(f"{r['language']}|{r['target_band']}" for r in planned)),
        "categories": dict(Counter(r["disturbance_category"] for r in planned)),
        "direct_target_counts": dict(Counter(str(len(r["direct_target_rubric_ids"])) for r in planned)),
        "labels_read": False,
        "audio_model_used": False,
        "http_api_used": False,
    }
    atomic_write(WORK / "plan/summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def shard_rows(rows: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError(f"invalid shard {shard_id}/{num_shards}")
    return [row for row in rows if int(row["execution_rank"]) % num_shards == shard_id]


def shard_output(shard_id: int) -> Path:
    return WORK / f"model/attempts/shard_{shard_id:02d}.jsonl"


def extract_json(raw: str, required_key: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_key in value:
            found.append(value)
    if not found:
        raise ValueError(f"no JSON object containing {required_key!r}")
    return found[-1]


def output_text(request_output: Any) -> str:
    outputs = getattr(request_output, "outputs", None) or []
    if not outputs:
        raise ValueError("vLLM returned no output")
    return str(outputs[0].text or "")


def generation_prompt(plan: dict[str, Any], feedback: str | None) -> str:
    retry = f"\nPREVIOUS FAILURE:\n{feedback}\nRepair all listed defects.\n" if feedback else ""
    retained_guards = sorted({
        str(rubric.get("basis") or "") for rubric in plan["retained_rubrics"]
        if str(rubric.get("basis") or "")
        and plan["original_prompt"].count(str(rubric.get("basis") or "")) == 1
    })
    return f"""Create one controlled Route 3 prompt-side negative using exact-span patches.

The source (P,A) is positive by construction. No labels or audio are available. Change P only.
Return JSON only.

Rules:
1. Contradict every DIRECT TARGET exactly once. Do not claim dependent rubric IDs as direct edits.
2. Event: replace with a clearly different coarse audible event, never absence or silence.
   Do not keep the same sound family, source class, or action with only a nearby subtype change.
3. Attribute: keep event identity and replace the property with a mutually exclusive one.
4. Relation: keep both event identities and explicitly invert before/after, overlap/separation,
   foreground/background, or same-source/different-source.
5. original_span is one unique verbatim contiguous substring of FROZEN P. Patches do not overlap.
6. Preserve text outside patches byte-for-byte and preserve unrelated facts.
7. No numeric times, visual facts, unverifiable causality, no/not/without/没有/停止 shortcuts,
   genre swaps, accent swaps, or nearby intensity/count changes.
8. Each explanation names every rubric ID attached to that patch.
9. Event/attribute patches report original_event_family and replacement_event_family using
   speech|music|sound_effect|vocalization|animal|ambience|mechanical|vehicle|impact|footsteps|other.
   Attributes preserve the family; events change it.
10. Relation patches report exactly two distinct relation_endpoints that occur verbatim in
    P PRIME. Non-relation patches use an empty relation_endpoints list.

LANGUAGE: {plan['language']}
TARGET BAND: {plan['target_band']}
CATEGORY: {plan['disturbance_category']}
DIRECT TARGET IDS: {json.dumps(plan['direct_target_rubric_ids'], ensure_ascii=False)}
DIRECT TARGET RUBRICS:
{json.dumps([route1.compact_rubric(x) for x in plan['direct_target_rubrics']], ensure_ascii=False)}

DEPENDENCY-AFFECTED IDS (consequences, not extra direct edits):
{json.dumps(plan['hierarchy_affected_rubric_ids'], ensure_ascii=False)}

RETAINED BASIS STRINGS (each must remain verbatim in P PRIME):
{json.dumps(retained_guards, ensure_ascii=False)}

FROZEN P:
{plan['original_prompt']}
{retry}
Return exactly:
{{"patches":[{{"rubric_ids":["R..."],"original_span":"exact source substring",
"replacement_span":"natural counterfactual text","edit_kind":"event|attribute|relation|mixed",
"original_event_family":"speech|music|sound_effect|vocalization|animal|ambience|mechanical|vehicle|impact|footsteps|other",
"replacement_event_family":"same controlled vocabulary value",
"relation_endpoints":["audible endpoint 1","audible endpoint 2"],
"semantic_distance":"high","contradiction_explanation":"...","audible_test":"..."}}]}}"""


def chat_tokens(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True,
    )
    return len(encoded["input_ids"])


def at_risk_retained(plan: dict[str, Any], patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        rubric for rubric in plan["retained_rubrics"]
        if route1.rubric_basis_overlaps_patches(rubric, plan["original_prompt"], patches)
    ]


def relation_is_explicit(original: str, replacement: str) -> bool:
    if RELATION_MARKER.search(replacement):
        return True
    chunks = [
        value.strip(" \t\r\n\"'“”‘’")
        for value in re.split(r"[。！？!?；;]|(?<=[.])\s+|,\s+", original)
    ]
    chunks = [value for value in chunks if len(value) >= 8 and value in replacement]
    positions = [replacement.index(value) for value in chunks]
    return any(left > right for left, right in zip(positions, positions[1:]))


def lexical_similarity(left: str, right: str) -> float:
    def normalize(value: str) -> str:
        return "".join(char.lower() for char in value if char.isalnum())

    return difflib.SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def validate_counterfactual_patch_result(
    plan: dict[str, Any], prime: str, patches: list[dict[str, Any]],
) -> None:
    qmap = {str(item["rubric_id"]): item for item in plan["direct_target_rubrics"]}
    for index, patch in enumerate(patches):
        replacement = str(patch["replacement_span"])
        original = str(patch["original_span"])
        original_absence = Counter(
            match.group(0).casefold() for match in UNIVERSAL_ABSENCE.finditer(original)
        )
        replacement_absence = Counter(
            match.group(0).casefold() for match in UNIVERSAL_ABSENCE.finditer(replacement)
        )
        introduced_absence = replacement_absence - original_absence
        if introduced_absence:
            raise ValueError(
                f"patch_{index}_universal_absence_shortcut:"
                f"{next(iter(introduced_absence))!r}"
            )
        expected_kinds = {route1.rubric_kind(qmap[rid]) for rid in patch["rubric_ids"]}
        expected_kinds.discard(None)
        expected_declared = next(iter(expected_kinds)) if len(expected_kinds) == 1 else "mixed"
        declared = str(patch.get("edit_kind") or "").lower()
        if declared != expected_declared:
            patch["model_edit_kind"] = declared
            patch["edit_kind"] = expected_declared
        explanation = str(patch.get("contradiction_explanation") or "")
        missing_ids = [rid for rid in patch["rubric_ids"] if rid not in explanation]
        if missing_ids:
            patch["model_contradiction_explanation"] = explanation
            patch["contradiction_explanation"] = f"{' '.join(missing_ids)}: {explanation}"
    # Path1's medium-band cap is 0.45, while this Path3 design explicitly
    # permits a whole audible relation clause up to 0.60.  Run all shared
    # checks with the broader base ceiling, then enforce Path3's tighter
    # route-specific cap immediately below.
    shared_plan = dict(plan)
    shared_plan["intended_perturbation_strength"] = "strong"
    route1.validate_patch_result(shared_plan, prime, patches)
    changed_chars = sum(int(patch["end_char"]) - int(patch["start_char"]) for patch in patches)
    changed_fraction = changed_chars / max(1, len(plan["original_prompt"]))
    # Some Path3 relation rubrics use a whole audible clause as their only
    # faithful edit unit.  Keep a hard cap, but allow those deterministic
    # clause spans rather than making otherwise valid plans impossible.
    maximum = 0.60 if plan["target_band"] == "mid" else 0.85
    if changed_fraction > maximum:
        raise ValueError(f"counterfactual_patched_source_fraction:{changed_fraction:.4f}>{maximum:.4f}")
    # Match Path1's deterministic overlap policy: text outside every patch is
    # frozen byte-for-byte.  A retained rubric whose basis intersects a direct
    # patch cannot be promised literal, so the grouped runner records it in the
    # effective affected set instead of making the construction impossible.
    for rubric in plan["retained_rubrics"]:
        basis = str(rubric.get("basis") or "")
        if (
            basis
            and basis in plan["original_prompt"]
            and basis not in prime
            and not route1.rubric_basis_overlaps_patches(
                rubric, plan["original_prompt"], patches,
            )
        ):
            raise ValueError(f"retained_basis_not_literal_outside_patches:{rubric['rubric_id']}")
    for index, patch in enumerate(patches):
        replacement = str(patch["replacement_span"])
        expected_kinds = {route1.rubric_kind(qmap[rid]) for rid in patch["rubric_ids"]}
        expected_kinds.discard(None)
        if expected_kinds == {"event"}:
            similarity = lexical_similarity(str(patch["original_span"]), replacement)
            if similarity > 0.55:
                raise ValueError(f"patch_{index}_event_too_lexically_similar:{similarity:.3f}>0.550")
        if "relation" in expected_kinds and not relation_is_explicit(
            str(patch["original_span"]), replacement,
        ):
            raise ValueError(f"patch_{index}_relation_not_explicit")


def audit_prompt(plan: dict[str, Any], prime: str, patches: list[dict[str, Any]]) -> str:
    return f"""Audit a text-only Path3 counterfactual. No audio or labels are available.

Do not trust declared patch coverage. For every DIRECT rubric, contradicted=true only when P PRIME
clearly makes its original yes/no claim false using the required event/attribute/relation edit type.
Evidence must be a non-empty verbatim substring of P PRIME. Return JSON only.

Dependent rubrics are validated separately by the deterministic dependency closure. Retained
rubrics are validated separately by exact literal-basis preservation. Do not enumerate them.

P PRIME:
{prime}

DIRECT:
{json.dumps([route1.compact_rubric(x) for x in plan['direct_target_rubrics']], ensure_ascii=False)}

Return exactly:
{{"direct":[{{"rubric_id":"R...","contradicted":true,"evidence":"exact P PRIME span"}}]}}"""


def validate_audit(
    plan: dict[str, Any], prime: str, patches: list[dict[str, Any]], audit: dict[str, Any],
) -> None:
    direct_expected = set(plan["direct_target_rubric_ids"])
    failures: list[str] = []
    values = audit.get("direct")
    if not isinstance(values, list):
        failures.append("direct_array_missing")
    else:
        ids = [str(x.get("rubric_id")) for x in values if isinstance(x, dict)]
        if len(ids) != len(set(ids)) or set(ids) != direct_expected:
            failures.append(f"direct_ids:{sorted(set(ids))}!={sorted(direct_expected)}")
        else:
            for item in values:
                evidence = str(item.get("evidence") or "")
                if item.get("contradicted") is not True or not evidence or evidence not in prime:
                    failures.append(f"direct_{item.get('rubric_id')}_contradicted_failed")
    if failures:
        raise ValueError("semantic_audit:" + ";".join(failures))


def validate_success(plan: dict[str, Any], row: dict[str, Any]) -> None:
    rebuilt, patches = route1.apply_exact_patches(
        plan["original_prompt"], row["patches"], set(plan["direct_target_rubric_ids"]),
    )
    if rebuilt != row.get("prompt_prime"):
        raise ValueError("stored_prompt_prime_mismatch")
    validate_counterfactual_patch_result(plan, rebuilt, patches)
    validate_audit(plan, rebuilt, patches, row["semantic_audit"])


def successes(path: Path, plan_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if row.get("status") != "ok" or sample_id not in plan_by_id:
            continue
        try:
            validate_success(plan_by_id[sample_id], row)
        except Exception:
            continue
        result[sample_id] = row
    return result


def run_shard(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    all_plan = read_jsonl(PLAN)
    selected = shard_rows(all_plan, args.shard_id, args.num_shards)
    if args.limit:
        selected = selected[:args.limit]
    plan_by_id = {str(row["sample_id"]): row for row in selected}
    output = shard_output(args.shard_id)
    completed = successes(output, plan_by_id)
    pending = [row for row in selected if row["sample_id"] not in completed]
    metadata = {
        "timestamp": now(), "schema_version": SCHEMA_VERSION, "backend": BACKEND,
        "model": MODEL_NAME, "model_path": str(args.model_path.resolve()),
        "model_config_sha256": file_sha256(args.model_path / "config.json"),
        "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "plan": str(PLAN), "plan_sha256": file_sha256(PLAN),
        "script_sha256": file_sha256(Path(__file__)),
        "counterfactual_validator_sha256": file_sha256(ROUTE1_CODE / "counterfactual_pipeline.py"),
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "selected": len(selected), "resumed": len(completed), "pending": len(pending),
        "batch_size": args.batch_size, "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_tokens": args.max_tokens, "audit_max_tokens": args.audit_max_tokens,
        "retries": args.retries, "labels_read": False, "audio_model_used": False,
        "http_api_used": False,
    }
    atomic_write(output.with_suffix(".metadata.json"), json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if not pending:
        print(json.dumps({"shard": args.shard_id, "completed": len(completed), "remaining": 0}))
        return

    llm = LLM(
        model=str(args.model_path), trust_remote_code=True, dtype="bfloat16",
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size, max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True, safetensors_load_strategy="eager", skip_mm_profiling=True,
    )
    generation_sampling = SamplingParams(temperature=0.1, max_tokens=args.max_tokens)
    audit_sampling = SamplingParams(temperature=0.0, max_tokens=args.audit_max_tokens)
    tokenizer = llm.get_tokenizer()
    max_generation_input = args.max_model_len - args.max_tokens - 32
    max_audit_input = args.max_model_len - args.audit_max_tokens - 32

    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]
        active = [{"plan": row, "feedback": None} for row in batch]
        for attempt in range(1, args.retries + 1):
            if not active:
                break
            prompts = [generation_prompt(x["plan"], x["feedback"]) for x in active]
            sizes = [chat_tokens(tokenizer, prompt) for prompt in prompts]
            if any(size > max_generation_input for size in sizes):
                raise ValueError(f"generation prompt exceeds context: {max(sizes)}>{max_generation_input}")
            generated = llm.chat(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                sampling_params=generation_sampling, use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            structurally_valid: list[dict[str, Any]] = []
            retry: list[dict[str, Any]] = []
            for state, prompt, input_tokens, model_output in zip(active, prompts, sizes, generated):
                plan = state["plan"]
                raw = output_text(model_output)
                common = {
                    "timestamp": now(), "schema_version": SCHEMA_VERSION, "backend": BACKEND,
                    "sample_id": plan["sample_id"], "parent_positive_id": plan["parent_positive_id"],
                    "attempt": attempt, "shard_id": args.shard_id, "num_shards": args.num_shards,
                    "model": MODEL_NAME, "model_path": str(args.model_path.resolve()),
                    "generation_request_sha256": sha256_text(prompt), "input_tokens": input_tokens,
                    "raw_model_output": raw, "label_visibility": "none", "label_status": "pending",
                }
                try:
                    parsed = extract_json(raw, "patches")
                    prime, patches = route1.apply_exact_patches(
                        plan["original_prompt"], list(parsed.get("patches") or []),
                        set(plan["direct_target_rubric_ids"]),
                    )
                    validate_counterfactual_patch_result(plan, prime, patches)
                    structurally_valid.append({
                        "plan": plan, "prompt": prompt, "prime": prime, "patches": patches,
                        "raw": raw, "common": common,
                    })
                except Exception as exc:
                    append_jsonl(output, {**common, "stage": "generate", "status": "error", "error": repr(exc)})
                    if attempt < args.retries:
                        retry.append({"plan": plan, "feedback": repr(exc)})

            if structurally_valid:
                audit_prompts = [audit_prompt(x["plan"], x["prime"], x["patches"]) for x in structurally_valid]
                audit_sizes = [chat_tokens(tokenizer, prompt) for prompt in audit_prompts]
                if any(size > max_audit_input for size in audit_sizes):
                    raise ValueError(f"audit prompt exceeds context: {max(audit_sizes)}>{max_audit_input}")
                audited = llm.chat(
                    [[{"role": "user", "content": prompt}] for prompt in audit_prompts],
                    sampling_params=audit_sampling, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
                for state, prompt, input_tokens, model_output in zip(
                    structurally_valid, audit_prompts, audit_sizes, audited,
                ):
                    plan = state["plan"]
                    raw_audit = output_text(model_output)
                    common = state["common"]
                    try:
                        audit = extract_json(raw_audit, "direct")
                        validate_audit(plan, state["prime"], state["patches"], audit)
                        append_jsonl(output, {
                            **common, "stage": "generate_and_audit", "status": "ok",
                            "audit_request_sha256": sha256_text(prompt), "audit_input_tokens": input_tokens,
                            "raw_audit_output": raw_audit, "prompt_prime": state["prime"],
                            "prompt_prime_sha256": sha256_text(state["prime"]),
                            "patches": state["patches"], "semantic_audit": audit,
                            "outside_patch_exact": True, "audio_byte_identical_to_parent": True,
                        })
                    except Exception as exc:
                        append_jsonl(output, {
                            **common, "stage": "semantic_audit", "status": "error",
                            "audit_request_sha256": sha256_text(prompt), "audit_input_tokens": input_tokens,
                            "raw_audit_output": raw_audit, "prompt_prime": state["prime"],
                            "patches": state["patches"], "error": repr(exc),
                        })
                        if attempt < args.retries:
                            retry.append({"plan": plan, "feedback": repr(exc)})
            active = retry

    completed = successes(output, plan_by_id)
    payload = {
        "shard": args.shard_id, "planned": len(selected), "completed": len(completed),
        "remaining": len(selected) - len(completed), "output": str(output),
    }
    print(json.dumps(payload, ensure_ascii=False))
    if payload["remaining"] and not args.allow_partial:
        raise SystemExit(2)


def status(args: argparse.Namespace) -> None:
    plan = read_jsonl(PLAN)
    payload: dict[str, Any] = {"planned": len(plan), "num_shards": args.num_shards, "shards": []}
    total = 0
    for shard_id in range(args.num_shards):
        selected = shard_rows(plan, shard_id, args.num_shards)
        scoped = {str(row["sample_id"]): row for row in selected}
        path = shard_output(shard_id)
        complete = len(successes(path, scoped))
        total += complete
        payload["shards"].append({
            "shard_id": shard_id, "planned": len(selected), "completed": complete,
            "remaining": len(selected) - complete, "attempt_records": len(read_jsonl(path)),
        })
    payload["completed"] = total
    payload["remaining"] = len(plan) - total
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def merge(args: argparse.Namespace) -> None:
    plan = read_jsonl(PLAN)
    plan_by_id = {str(row["sample_id"]): row for row in plan}
    accepted: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    attempts: Counter[str] = Counter()
    for shard_id in range(args.num_shards):
        path = shard_output(shard_id)
        rows = read_jsonl(path)
        attempts.update(str(row.get("status")) for row in rows)
        inputs.append({"path": str(path), "rows": len(rows), "sha256": file_sha256(path) if path.exists() else None})
        accepted.update(successes(path, plan_by_id))
    missing = [row["sample_id"] for row in plan if row["sample_id"] not in accepted]
    if missing and not args.allow_partial:
        raise ValueError(f"cannot merge incomplete run: {len(accepted)}/{len(plan)}")
    pairs: list[dict[str, Any]] = []
    for row in sorted(plan, key=lambda item: int(item["execution_rank"])):
        if row["sample_id"] not in accepted:
            continue
        result = accepted[row["sample_id"]]
        pairs.append({
            "schema_version": SCHEMA_VERSION,
            "sample_id": row["sample_id"], "source_pair_id": row["parent_positive_id"],
            "language": row["language"], "target_band": row["target_band"],
            "pair_type": "P_prime_A", "route": "route3_prompt_side_negative",
            "original_prompt": row["original_prompt"], "prompt_prime": result["prompt_prime"],
            "source_audio_path": row["audio_path"], "source_audio_sha256": row["audio_sha256"],
            "direct_target_rubrics": row["direct_target_rubrics"],
            "affected_rubrics": row["affected_rubrics"], "retained_rubrics": row["retained_rubrics"],
            "patches": result["patches"], "semantic_audit": result["semantic_audit"],
            "outside_patch_exact": True, "audio_byte_identical_to_parent": True,
            "label_visibility": "none", "label_status": "pending",
            "generator": {"backend": BACKEND, "model": MODEL_NAME, "attempt": result["attempt"]},
        })
    output = WORK / "manifests/path3_prompt_side_pairs_500.jsonl"
    write_jsonl(output, pairs)
    summary = {
        "timestamp": now(), "planned": len(plan), "accepted": len(pairs), "missing": len(missing),
        "missing_sample_ids": missing, "attempt_status": dict(attempts),
        "distribution": dict(Counter(f"{r['language']}|{r['target_band']}" for r in pairs)),
        "output": str(output), "output_sha256": file_sha256(output),
        "plan_sha256": file_sha256(PLAN), "script_sha256": file_sha256(Path(__file__)),
        "inputs": inputs, "labels_read": False, "audio_model_used": False, "http_api_used": False,
        "legacy_output_included": False,
    }
    atomic_write(WORK / "manifests/summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-shard")
    run.add_argument("--shard-id", type=int, required=True)
    run.add_argument("--num-shards", type=int, default=2)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run.add_argument("--max-model-len", type=int, default=16384)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    run.add_argument("--max-tokens", type=int, default=2400)
    run.add_argument("--audit-max-tokens", type=int, default=2400)
    run.add_argument("--retries", type=int, default=5)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--allow-partial", action="store_true")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--num-shards", type=int, default=2)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--num-shards", type=int, default=2)
    merge_parser.add_argument("--allow-partial", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
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
