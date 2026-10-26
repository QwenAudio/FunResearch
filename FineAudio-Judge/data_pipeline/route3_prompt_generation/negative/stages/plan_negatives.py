#!/usr/bin/env python3
"""Prepare and finalize the 1k native Path3 negative construction run.

This stage deliberately records construction intent only.  It does not use
partial Omni labels and it does not filter candidates by realized flips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUTE1_CODE = Path(__file__).resolve().parents[3] / "route1_real_world/negative/stages"
if str(ROUTE1_CODE) not in sys.path:
    sys.path.insert(0, str(ROUTE1_CODE))
import counterfactual_pipeline as route1  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "inputs/positive_with_questions.jsonl"
RUBRIC_SOURCE: Path | None = None
OSCAR_SOURCE = ROOT / "inputs/positive_review.jsonl"
PLAN = ROOT / "plan/negative_plan.jsonl"
PILOT = ROOT / "plan/pilot.jsonl"
PROPOSALS = ROOT / "patches/negative_patch_proposals.jsonl"
AUDIO_PLAN = ROOT / "plan/audio_side_plan.jsonl"
PROMPT_PLAN = ROOT / "plan/prompt_side_plan.jsonl"
AUDIO_WARM_PLAN = ROOT / "plan/audio_side_warm_plan.jsonl"
AUDIO_REST_PLAN = ROOT / "plan/audio_side_remaining_plan.jsonl"
AUDIO_PROPOSALS = ROOT / "patches/audio_side_patch_proposals.jsonl"
AUDIO_REPLACEMENTS = ROOT / "plan/audio_side_plan_replacements.jsonl"
# Preserve the frozen deterministic-selection namespace byte-for-byte without
# exposing the former internal run label in public filenames or metadata.
RANK_NAMESPACE = bytes.fromhex(
    "70617468332d6e656761746976652d316b2d32303236303930343a"
).decode("ascii")
FROZEN_SAMPLE_PREFIX = bytes.fromhex("70617468336e6567316b5f").decode("ascii")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(value: str) -> int:
    frozen_value = value.replace("route3neg_", FROZEN_SAMPLE_PREFIX)
    return int(hashlib.sha256(f"{RANK_NAMESPACE}{frozen_value}".encode()).hexdigest()[:16], 16)


def _row_id(row: dict[str, Any], *, location: str) -> str:
    value = row.get("sample_id") or row.get("id")
    if value is None or not str(value).strip():
        raise ValueError(f"{location}: missing sample_id/id")
    return str(value)


def _join_key(row: dict[str, Any], *, location: str) -> tuple[str | None, str]:
    dataset = row.get("dataset")
    return (str(dataset) if dataset is not None else None, _row_id(row, location=location))


def adapt_public_inputs(
    positive_path: Path, rubric_path: Path,
) -> list[dict[str, Any]]:
    """Join Route 3 positives with independently generated, label-blind rubrics."""
    positives = read_jsonl(positive_path)
    rubrics = read_jsonl(rubric_path)
    positive_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    positive_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(positives, 1):
        key = _join_key(row, location=f"{positive_path}:{index}")
        if key in positive_by_key:
            raise ValueError(f"duplicate positive join key: {key}")
        positive_by_key[key] = row
        positive_by_id[key[1]].append(row)

    rubric_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    for index, row in enumerate(rubrics, 1):
        key = _join_key(row, location=f"{rubric_path}:{index}")
        if key in rubric_by_key:
            raise ValueError(f"duplicate rubric join key: {key}")
        rubric_by_key[key] = row

    joined: list[dict[str, Any]] = []
    matched: set[tuple[str | None, str]] = set()
    for key, positive in positive_by_key.items():
        rubric_key = key
        rubric = rubric_by_key.get(rubric_key)
        if rubric is None and key[0] is not None:
            rubric_key = (None, key[1])
            rubric = rubric_by_key.get(rubric_key)
            if rubric is not None and len(positive_by_id[key[1]]) > 1:
                raise ValueError(
                    f"ambiguous rubric ID {key[1]!r}; include dataset in the rubric manifest"
                )
        if rubric is None:
            continue
        matched.add(rubric_key)
        prompt = str(positive.get("prompt") or positive.get("caption") or "").strip()
        rubric_prompt = str(rubric.get("prompt") or rubric.get("caption") or "").strip()
        if not prompt:
            raise ValueError(f"positive row {key} has no prompt")
        if rubric_prompt and rubric_prompt != prompt:
            raise ValueError(f"prompt/rubric mismatch for {key}")
        questions = rubric.get("questions") or rubric.get("rubrics") or []
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"rubric row {rubric_key} has no questions/rubrics")
        sample_id = key[1] if key[0] is None else f"{key[0]}:{key[1]}"
        joined.append({
            **positive,
            "source_id": sample_id,
            "sample_id": sample_id,
            "caption": prompt,
            "prompt": prompt,
            "audio_path": positive.get("audio_path") or positive.get("final_audio_path"),
            "questions": questions,
        })
    unmatched = sorted(set(rubric_by_key) - matched, key=str)
    if unmatched:
        raise ValueError(f"rubric rows have no matching positive row: {unmatched[:5]}")
    if not joined:
        raise ValueError("no positive/rubric rows matched")
    return joined


def _sanitize_source(raw: dict[str, Any], *, location: str) -> dict[str, Any] | None:
    accepted_routes = {"path3_fully_generated_audio", "route3_prompt_generation"}
    if raw.get("route") not in accepted_routes:
        return None
    row = dict(raw)
    row.setdefault("source_id", row.get("sample_id") or row.get("id"))
    row.setdefault("caption", row.get("prompt"))
    row.setdefault("audio_path", row.get("final_audio_path"))
    if not row.get("source_id") or not str(row.get("caption") or "").strip():
        raise ValueError(f"{location}: missing source ID or prompt")
    audio_path = Path(str(row.get("audio_path") or ""))
    if not audio_path.is_file():
        raise FileNotFoundError(f"{location}: missing audio file: {audio_path}")
    questions = row.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{location}: missing questions")
    clean_questions = []
    for item in questions:
        candidate = dict(item)
        # The canonical decomposition model serializes its construction-time
        # invariant ``expected_answer=yes``. Strip that constant at the Route 3
        # trust boundary; reject every other answer/label field via Route 1's
        # label-blind sanitizer.
        if "expected_answer" in candidate:
            if str(candidate["expected_answer"]).lower() != "yes":
                raise ValueError(f"{location}: non-positive expected_answer is forbidden")
            candidate.pop("expected_answer")
        clean_questions.append(route1.sanitize_question(candidate))
    rubric_ids = [str(item["rubric_id"]) for item in clean_questions]
    if len(rubric_ids) != len(set(rubric_ids)):
        raise ValueError(f"{location}: duplicate rubric IDs")
    known = set(rubric_ids)
    for item in clean_questions:
        unknown = set(map(str, item.get("parent_ids") or [])) - known
        if unknown:
            raise ValueError(
                f"{location}: rubric {item['rubric_id']} has unknown parents {sorted(unknown)}"
            )
        rejection = route1.target_rejection(item, str(row["caption"]))
        if rejection == "empty_basis" or (
            str(item.get("basis") or "") not in str(row["caption"])
        ):
            raise ValueError(f"{location}: invalid rubric basis for {item['rubric_id']}")
    row["questions"] = clean_questions
    return row


def load_eligible_sources() -> list[dict[str, Any]]:
    """Normalize either the legacy combined input or split positive/rubric manifests."""
    sources: list[dict[str, Any]] = []
    rows = adapt_public_inputs(SOURCE, RUBRIC_SOURCE) if RUBRIC_SOURCE else read_jsonl(SOURCE)
    for index, raw in enumerate(rows, 1):
        row = _sanitize_source(raw, location=f"{SOURCE}:{index}")
        if row is not None:
            sources.append(row)
    ids = [str(row["source_id"]) for row in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Route 3 source IDs")
    return sources


def classify_rubric(item: dict[str, Any]) -> str:
    kind = route1.rubric_kind(item)
    return kind or "unsupported"


def apportion(total: int, sizes: dict[str, int]) -> dict[str, int]:
    available = sum(sizes.values())
    if total > available:
        raise ValueError(f"need {total} eligible rubrics but only {available} are available")
    raw = {kind: total * size / available for kind, size in sizes.items()}
    result = {kind: min(size, math.floor(raw[kind])) for kind, size in sizes.items()}
    remaining = total - sum(result.values())
    order = sorted(sizes, key=lambda kind: (raw[kind] - result[kind], sizes[kind], kind), reverse=True)
    while remaining:
        progress = False
        for kind in order:
            if result[kind] < sizes[kind]:
                result[kind] += 1
                remaining -= 1
                progress = True
                if not remaining:
                    break
        if not progress:
            raise AssertionError("rubric apportionment stalled")
    return result


def select_changed(sample_id: str, prompt: str, questions: list[dict[str, Any]], fraction: float) -> list[str]:
    target = max(1, round(len(questions) * fraction))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in questions:
        if route1.target_rejection(item, prompt):
            continue
        buckets[classify_rubric(item)].append(item)
    sizes = {kind: len(buckets[kind]) for kind in ("event", "attribute", "relation")}
    quotas = apportion(target, sizes)
    selected: list[str] = []
    for kind in ("event", "attribute", "relation"):
        ranked = sorted(
            buckets[kind],
            key=lambda item: (
                0 if str(item.get("basis") or "") and prompt.count(str(item.get("basis"))) == 1 else 1,
                stable_rank(f"{sample_id}:{kind}:{item['rubric_id']}"),
            ),
        )
        selected.extend(str(item["rubric_id"]) for item in ranked[:quotas[kind]])
    if len(selected) != target or len(selected) != len(set(selected)):
        raise AssertionError(f"{sample_id}: invalid changed selection {len(selected)}/{target}")
    return sorted(selected, key=lambda rid: stable_rank(f"order:{sample_id}:{rid}"))


def slots(language: str) -> list[tuple[str, str]]:
    values = [
        *(('mid', 'P_prime_A') for _ in range(125)),
        *(('low', 'P_prime_A') for _ in range(125)),
        *(('mid', 'P_A_prime') for _ in range(125)),
        *(('low', 'P_A_prime') for _ in range(125)),
    ]
    return sorted(values, key=lambda value: stable_rank(f"slot:{language}:{value[0]}:{value[1]}:{values.index(value)}"))


def cmd_prepare(_: argparse.Namespace) -> None:
    sources = load_eligible_sources()
    if len(sources) < 1000:
        raise ValueError(f"need at least 1000 eligible Route 3 positive sources, got {len(sources)}")
    oscar = {row["sample_id"]: row for row in read_jsonl(OSCAR_SOURCE)}
    plan: list[dict[str, Any]] = []
    for language in ("zh", "en"):
        pool = sorted(
            (row for row in sources if row.get("language") == language),
            key=lambda row: stable_rank(f"source:{row['source_id']}"),
        )
        selected = pool[:500]
        if len(selected) != 500:
            raise ValueError(f"language {language}: only {len(selected)} sources")
        language_slots = slots(language)
        if len(language_slots) != 500:
            raise AssertionError("slot count mismatch")
        for source, (band, pair_type) in zip(selected, language_slots):
            source_id = str(source["source_id"])
            prompt = str(source["caption"])
            questions = source["questions"]
            change_fraction = 0.25 if band == "mid" else 0.50
            changed_ids = select_changed(source_id, prompt, questions, change_fraction)
            changed_set = set(changed_ids)
            typed_questions = [{**item, "rubric_type": classify_rubric(item)} for item in questions]
            changed = [item for item in typed_questions if str(item["rubric_id"]) in changed_set]
            retained = [item for item in typed_questions if str(item["rubric_id"]) not in changed_set]
            suffix = "p" if pair_type == "P_prime_A" else "a"
            sample_id = f"route3neg_{source_id}__{suffix}_{band}"
            source_oscar = oscar.get(source_id, {})
            plan.append({
                "sample_id": sample_id,
                "source_pair_id": source_id,
                "source_route": "path3_fully_generated_audio",
                "language": language,
                "target_band": band,
                "pair_type": pair_type,
                "original_prompt": prompt,
                "natural_prompt": prompt,
                "source_generation_prompt": prompt,
                "source_generation_prompt_sha256": sha256_text(prompt),
                "rubric_source_prompt_sha256": sha256_text(prompt),
                "source_audio_path": source["audio_path"],
                "source_audio_sha256": source_oscar.get("final_audio_sha256") or source_oscar.get("audio_sha256"),
                "source_audio_duration_seconds": source.get("audio_duration_seconds"),
                "source_review_flags": source_oscar.get("review_flags") or [],
                "rubrics_total": len(questions),
                "construction_keep_fraction_target": 0.75 if band == "mid" else 0.50,
                "construction_change_fraction_target": change_fraction,
                "construction_change_fraction_actual": round(len(changed) / len(questions), 6),
                "changed_rubric_ids": [str(item["rubric_id"]) for item in changed],
                "retained_rubric_ids": [str(item["rubric_id"]) for item in retained],
                "changed_rubrics": changed,
                "retained_rubrics": retained,
                "source_questions": questions,
                "filtering_status": "deferred_until_post_generation_omni_labels",
                "dependency_closure_status": "deferred_by_user_for_filtering_stage",
            })
    plan.sort(key=lambda row: stable_rank(f"plan:{row['sample_id']}"))
    if len(plan) != 1000 or len({row["source_pair_id"] for row in plan}) != 1000:
        raise AssertionError("plan must contain 1000 unique source pairs")
    expected = Counter({
        (language, band, pair_type): 125
        for language in ("zh", "en")
        for band in ("mid", "low")
        for pair_type in ("P_prime_A", "P_A_prime")
    })
    actual = Counter((row["language"], row["target_band"], row["pair_type"]) for row in plan)
    if actual != expected:
        raise AssertionError(f"slot distribution mismatch: {actual}")
    write_jsonl(PLAN, plan)
    audio_plan = [row for row in plan if row["pair_type"] == "P_A_prime"]
    write_jsonl(AUDIO_PLAN, audio_plan)
    write_jsonl(PROMPT_PLAN, [row for row in plan if row["pair_type"] == "P_prime_A"])
    warm_ids: set[str] = set()
    for language in ("zh", "en"):
        for band in ("mid", "low"):
            group = [
                row for row in audio_plan
                if row["language"] == language and row["target_band"] == band
            ]
            group.sort(key=lambda row: stable_rank(f"warm:{row['sample_id']}"))
            warm_ids.update(row["sample_id"] for row in group[:16])
    warm_plan = [row for row in audio_plan if row["sample_id"] in warm_ids]
    rest_plan = [row for row in audio_plan if row["sample_id"] not in warm_ids]
    if len(warm_plan) != 64 or len(rest_plan) != 436:
        raise AssertionError(f"invalid warm/rest split: {len(warm_plan)}/{len(rest_plan)}")
    write_jsonl(AUDIO_WARM_PLAN, warm_plan)
    write_jsonl(AUDIO_REST_PLAN, rest_plan)
    pilot = []
    for key in expected:
        pilot.append(next(row for row in plan if (row["language"], row["target_band"], row["pair_type"]) == key))
    write_jsonl(PILOT, pilot)
    summary = {
        "samples": len(plan),
        "unique_sources": len({row["source_pair_id"] for row in plan}),
        "distribution": {"|".join(key): value for key, value in sorted(actual.items())},
        "pair_types": dict(Counter(row["pair_type"] for row in plan)),
        "bands": dict(Counter(row["target_band"] for row in plan)),
        "languages": dict(Counter(row["language"] for row in plan)),
        "rubrics_total": sum(row["rubrics_total"] for row in plan),
        "changed_rubrics": sum(len(row["changed_rubric_ids"]) for row in plan),
        "source_audio_missing": sum(not Path(row["source_audio_path"]).is_file() for row in plan),
        "omni_filtering_applied": False,
        "dependency_closure_applied": False,
    }
    write_json(ROOT / "plan/negative_plan.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def latest_proposals(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in read_jsonl(path)}


def cmd_patch_status(args: argparse.Namespace) -> None:
    plan = read_jsonl(args.plan)
    latest = latest_proposals(args.proposals)
    good = sum(latest.get(row["sample_id"], {}).get("status") == "ok" for row in plan)
    payload = {"planned": len(plan), "patches_ok": good, "remaining": len(plan) - good}
    print(json.dumps(payload, ensure_ascii=False))
    if args.require_complete and good != len(plan):
        raise SystemExit(1)


def cmd_replace_failed_audio(args: argparse.Namespace) -> None:
    """Replace repeatedly invalid construction rows without blocking the other rows."""
    plan = read_jsonl(AUDIO_PLAN)
    if len(plan) != 500:
        raise ValueError(f"audio-side plan must contain 500 rows, got {len(plan)}")
    proposal_rows = read_jsonl(args.proposals)
    latest = latest_proposals(args.proposals)
    structural_error_counts = Counter(
        str(row.get("sample_id"))
        for row in proposal_rows
        if row.get("status") == "error" and str(row.get("error") or "").startswith("ValueError(")
    )
    replaceable = [
        row for row in plan
        if latest.get(row["sample_id"], {}).get("status") == "error"
        and str(latest[row["sample_id"]].get("error") or "").startswith("ValueError(")
        and structural_error_counts[row["sample_id"]] >= args.min_error_records
    ][: args.max_replacements]
    if not replaceable:
        print(json.dumps({"replaced": 0, "reason": "no repeatedly invalid structural patches"}, ensure_ascii=False))
        return

    all_sources = load_eligible_sources()
    original_full_plan = read_jsonl(PLAN)
    replacement_history = read_jsonl(AUDIO_REPLACEMENTS)
    used_sources = {str(row["source_pair_id"]) for row in original_full_plan}
    used_sources.update(str(row["source_pair_id"]) for row in plan)
    used_sources.update(str(row.get("new_source_pair_id")) for row in replacement_history)
    source_oscar = {row["sample_id"]: row for row in read_jsonl(OSCAR_SOURCE)}
    replacements: list[dict[str, Any]] = []
    retry_in_place: list[str] = []
    by_sample = {row["sample_id"]: row for row in plan}

    for old in replaceable:
        language = old["language"]
        candidates = [
            row for row in all_sources
            if row.get("language") == language
            and str(row["source_id"]) not in used_sources
            and Path(str(row["audio_path"])).is_file()
        ]
        candidates.sort(key=lambda row: stable_rank(f"replacement:{old['sample_id']}:{row['source_id']}"))
        if not candidates:
            # Exhausting a language-specific replacement pool must not block all
            # remaining work. Keep this row in the plan so the patch proposer can
            # repair its latest invalid output in place on the same round.
            retry_in_place.append(old["sample_id"])
            continue
        source = candidates[0]
        source_id = str(source["source_id"])
        used_sources.add(source_id)
        prompt = str(source["caption"])
        questions = source["questions"]
        fraction = float(old["construction_change_fraction_target"])
        changed_ids = select_changed(source_id, prompt, questions, fraction)
        changed_set = set(changed_ids)
        typed = [{**item, "rubric_type": classify_rubric(item)} for item in questions]
        changed = [item for item in typed if str(item["rubric_id"]) in changed_set]
        retained = [item for item in typed if str(item["rubric_id"]) not in changed_set]
        new_sample_id = f"route3neg_{source_id}__a_{old['target_band']}"
        oscar_row = source_oscar.get(source_id, {})
        new_row = {
            **old,
            "sample_id": new_sample_id,
            "source_pair_id": source_id,
            "original_prompt": prompt,
            "natural_prompt": prompt,
            "source_generation_prompt": prompt,
            "source_generation_prompt_sha256": sha256_text(prompt),
            "rubric_source_prompt_sha256": sha256_text(prompt),
            "source_audio_path": source["audio_path"],
            "source_audio_sha256": oscar_row.get("final_audio_sha256"),
            "source_audio_duration_seconds": source.get("audio_duration_seconds"),
            "source_review_flags": oscar_row.get("review_flags") or [],
            "rubrics_total": len(questions),
            "construction_change_fraction_actual": round(len(changed) / len(questions), 6),
            "changed_rubric_ids": [str(item["rubric_id"]) for item in changed],
            "retained_rubric_ids": [str(item["rubric_id"]) for item in retained],
            "changed_rubrics": changed,
            "retained_rubrics": retained,
            "source_questions": questions,
            "replacement_for_sample_id": old["sample_id"],
        }
        by_sample.pop(old["sample_id"])
        by_sample[new_sample_id] = new_row
        replacements.append({
            "replaced_at": datetime.now(timezone.utc).isoformat(),
            "old_sample_id": old["sample_id"],
            "old_source_pair_id": old["source_pair_id"],
            "new_sample_id": new_sample_id,
            "new_source_pair_id": source_id,
            "language": language,
            "target_band": old["target_band"],
            "structural_error_records": structural_error_counts[old["sample_id"]],
            "last_error": latest[old["sample_id"]].get("error"),
        })

    new_plan = [by_sample.get(row["sample_id"], row) for row in plan if row["sample_id"] in by_sample]
    present = {row["sample_id"] for row in new_plan}
    new_plan.extend(row for sample_id, row in by_sample.items() if sample_id not in present)
    if len(new_plan) != 500 or len({row["sample_id"] for row in new_plan}) != 500:
        raise AssertionError("replacement must preserve 500 unique audio plan rows")
    if Counter((row["language"], row["target_band"]) for row in new_plan) != Counter(
        (row["language"], row["target_band"]) for row in plan
    ):
        raise AssertionError("replacement changed language/band distribution")
    write_jsonl(AUDIO_PLAN, new_plan)
    write_jsonl(AUDIO_REPLACEMENTS, replacement_history + replacements)
    print(json.dumps({
        "replaced": len(replacements),
        "remaining_plan_rows": len(new_plan),
        "replacement_sample_ids": [row["new_sample_id"] for row in replacements],
        "retry_in_place_sample_ids": retry_in_place,
    }, ensure_ascii=False))


def finalize(plan: list[dict[str, Any]], latest: dict[str, dict[str, Any]], include_prompt: bool) -> None:
    missing = [row["sample_id"] for row in plan if latest.get(row["sample_id"], {}).get("status") != "ok"]
    if missing:
        raise ValueError(f"cannot finalize: {len(missing)} patch proposals are not ready")
    prompt_pairs: list[dict[str, Any]] = []
    audio_pairs: list[dict[str, Any]] = []
    oscar_rows: list[dict[str, Any]] = []
    for source in plan:
        proposal = latest[source["sample_id"]]
        patched = str(proposal["patched_prompt"])
        common = {
            "sample_id": source["sample_id"],
            "source_pair_id": source["source_pair_id"],
            "language": source["language"],
            "target_band": source["target_band"],
            "pair_type": source["pair_type"],
            "original_prompt": source["original_prompt"],
            "counterfactual_prompt": patched,
            "original_prompt_sha256": source["rubric_source_prompt_sha256"],
            "counterfactual_prompt_sha256": sha256_text(patched),
            "source_audio_path": source["source_audio_path"],
            "source_audio_duration_seconds": source["source_audio_duration_seconds"],
            "changed_rubric_ids": source["changed_rubric_ids"],
            "retained_rubric_ids": source["retained_rubric_ids"],
            "prompt_patches": proposal["patches"],
            "outside_patch_exact": proposal["outside_patch_exact"],
            "post_generation_filtering": "pending",
            "questions": source["source_questions"],
        }
        if source["pair_type"] == "P_prime_A":
            prompt_pairs.append({
                **common,
                "route": "path3_prompt_side_negative",
                "evaluation_prompt": patched,
                "evaluation_audio_path": source["source_audio_path"],
                "evaluation_rubrics_status": "pending_regeneration_from_P_prime",
            })
        else:
            audio_pairs.append({
                **common,
                "route": "path3_audio_side_negative",
                "evaluation_prompt": source["original_prompt"],
                "evaluation_audio_path": None,
                "evaluation_rubrics_status": "reuse_original_questions_after_audio_generation",
            })
            oscar_rows.append({
                "id": source["sample_id"],
                "prompt": patched,
                "prompt_sha256": sha256_text(patched),
                "source_sample_id": source["source_pair_id"],
                "pair_type": "P_A_prime",
                "target_band": source["target_band"],
                "language": source["language"],
                "evaluation_prompt": source["original_prompt"],
                "source_audio_duration_seconds": source["source_audio_duration_seconds"],
            })
    if include_prompt:
        write_jsonl(ROOT / "manifests/prompt_side_pairs.jsonl", prompt_pairs)
    write_jsonl(ROOT / "manifests/audio_side_pairs_pending.jsonl", audio_pairs)
    write_jsonl(ROOT / "manifests/audio_generation_prompts.jsonl", oscar_rows)
    for index in range(4):
        bucket = [row for row in oscar_rows if stable_rank(f"shard:{row['id']}") % 4 == index]
        write_jsonl(ROOT / f"manifests/oscar_audio_prime_shards/shard_{index:02d}.jsonl", bucket)
    summary = {
        "prompt_side_pairs": len(prompt_pairs),
        "audio_side_pairs": len(audio_pairs),
        "oscar_audio_to_generate": len(oscar_rows),
        "oscar_shards": [
            len(read_jsonl(ROOT / f"manifests/oscar_audio_prime_shards/shard_{index:02d}.jsonl"))
            for index in range(4)
        ],
        "duration_contract": "none; allow source-like natural duration distribution",
        "filtering": "deferred",
    }
    write_json(ROOT / "manifests/finalize.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_finalize(_: argparse.Namespace) -> None:
    finalize(read_jsonl(PLAN), latest_proposals(PROPOSALS), include_prompt=True)


def cmd_finalize_audio(args: argparse.Namespace) -> None:
    plan = read_jsonl(AUDIO_PLAN)
    if len(plan) != 500 or any(row["pair_type"] != "P_A_prime" for row in plan):
        raise ValueError(f"audio-side plan must contain exactly 500 P_A_prime rows, got {len(plan)}")
    finalize(plan, latest_proposals(args.proposals), include_prompt=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_assemble_audio(args: argparse.Namespace) -> None:
    pending = read_jsonl(ROOT / "manifests/audio_side_pairs_pending.jsonl")
    if len(pending) != 500:
        raise ValueError(f"expected 500 pending audio-side pairs, got {len(pending)}")
    final: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in pending:
        index = stable_rank(f"shard:{row['sample_id']}") % 4
        shared_root = ROOT / f"audio_side/oscar_outputs/samples/{row['sample_id']}"
        sharded_root = ROOT / f"audio_side/oscar_outputs/shard_{index:02d}/samples/{row['sample_id']}"
        sample_root = shared_root if shared_root.is_dir() else sharded_root
        marker = sample_root / "COMPLETE"
        summary_path = sample_root / "run_summary.json"
        if not marker.is_file() or not summary_path.is_file():
            missing.append(row["sample_id"])
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audio = Path(str(summary["final_audio"]))
        if not audio.is_file() or audio.stat().st_size == 0:
            missing.append(row["sample_id"])
            continue
        final.append({
            **row,
            "evaluation_audio_path": str(audio),
            "evaluation_audio_sha256": file_sha256(audio),
            "evaluation_audio_duration_seconds": summary.get("final_duration_seconds"),
            "generation_status": "complete",
            "oscar_run_summary_path": str(summary_path),
        })
    if missing and not args.allow_partial:
        raise ValueError(f"cannot assemble: {len(missing)} audio-side samples are incomplete")
    write_jsonl(ROOT / "manifests/audio_side_pairs.jsonl", final)
    report = {
        "expected": 500,
        "complete": len(final),
        "missing": len(missing),
        "missing_sample_ids": missing,
        "rubrics": "reuse original R(P)",
        "omni_filtering": "pending",
    }
    write_json(ROOT / "manifests/audio_side_assembly.summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=Path, default=ROOT)
    result.add_argument("--source", type=Path, default=SOURCE)
    result.add_argument(
        "--rubric-source", type=Path,
        help="Independent rubric manifest; when set, --source is the positive manifest.",
    )
    result.add_argument("--positive-review", type=Path, default=OSCAR_SOURCE)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    status = sub.add_parser("patch-status")
    status.add_argument("--plan", type=Path, default=PLAN)
    status.add_argument("--proposals", type=Path, default=PROPOSALS)
    status.add_argument("--require-complete", action="store_true")
    sub.add_parser("finalize")
    finalize_audio = sub.add_parser("finalize-audio")
    finalize_audio.add_argument("--proposals", type=Path, default=AUDIO_PROPOSALS)
    replace_audio = sub.add_parser("replace-failed-audio")
    replace_audio.add_argument("--proposals", type=Path, default=AUDIO_PROPOSALS)
    replace_audio.add_argument("--min-error-records", type=int, default=3)
    replace_audio.add_argument("--max-replacements", type=int, default=64)
    assemble_audio = sub.add_parser("assemble-audio")
    assemble_audio.add_argument("--allow-partial", action="store_true")
    return result


def main() -> None:
    global ROOT, SOURCE, RUBRIC_SOURCE, OSCAR_SOURCE, PLAN, PILOT, PROPOSALS
    global AUDIO_PLAN, PROMPT_PLAN, AUDIO_WARM_PLAN, AUDIO_REST_PLAN
    global AUDIO_PROPOSALS, AUDIO_REPLACEMENTS
    args = parser().parse_args()
    ROOT = args.root
    SOURCE = args.source
    RUBRIC_SOURCE = args.rubric_source
    OSCAR_SOURCE = args.positive_review
    PLAN = ROOT / "plan/negative_plan.jsonl"
    PILOT = ROOT / "plan/pilot.jsonl"
    PROPOSALS = ROOT / "patches/negative_patch_proposals.jsonl"
    AUDIO_PLAN = ROOT / "plan/audio_side_plan.jsonl"
    PROMPT_PLAN = ROOT / "plan/prompt_side_plan.jsonl"
    AUDIO_WARM_PLAN = ROOT / "plan/audio_side_warm_plan.jsonl"
    AUDIO_REST_PLAN = ROOT / "plan/audio_side_remaining_plan.jsonl"
    AUDIO_PROPOSALS = ROOT / "patches/audio_side_patch_proposals.jsonl"
    AUDIO_REPLACEMENTS = ROOT / "plan/audio_side_plan_replacements.jsonl"
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "patch-status":
        cmd_patch_status(args)
    elif args.command == "finalize":
        cmd_finalize(args)
    elif args.command == "finalize-audio":
        cmd_finalize_audio(args)
    elif args.command == "replace-failed-audio":
        cmd_replace_failed_audio(args)
    elif args.command == "assemble-audio":
        cmd_assemble_audio(args)


if __name__ == "__main__":
    main()
