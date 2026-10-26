"""Apply configured hard gates and rank Route 1 production candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_jsonl, write_jsonl
from content_safety import profanity_categories
from data_policy import source_group_id
from pipeline_meta import load_config, sha256_file
def source_asr_text(source: dict[str, Any]) -> str:
    """Flatten available source-ASR fields for deterministic safety checks."""
    value = source.get("source_asr")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for segment in value:
            if not isinstance(segment, dict):
                continue
            if segment.get("text"):
                texts.append(str(segment["text"]))
            if segment.get("full_text"):
                texts.append(str(segment["full_text"]))
            for item in segment.get("scene_text") or []:
                if isinstance(item, dict) and item.get("text"):
                    texts.append(str(item["text"]))
        return " ".join(dict.fromkeys(texts))
    return ""


def is_native_clip(source: dict[str, Any]) -> bool:
    """Return whether a source row represents one continuous native clip."""
    if source.get("is_composite") is True:
        return False
    locator = source.get("source_locator") or {}
    for key in ("scene_ids", "members", "segments"):
        value = locator.get(key)
        if isinstance(value, list) and len(value) > 1:
            return False
    return locator.get("composition", "native_single_clip") in {
        "native_single_clip",
        "native_row",
        "native_scene",
        "continuous_source_crop",
    }



def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _speech_intervals(source: dict[str, Any]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for item in source.get("trusted_time_annotations") or []:
        start, end = _float(item.get("start")), _float(item.get("end"))
        if start is not None and end is not None and end > start:
            intervals.append((max(0.0, start), end))
    if not intervals:
        for item in source.get("source_asr") or []:
            if not isinstance(item, dict):
                continue
            nested = item.get("scene_text")
            if isinstance(nested, list):
                for segment in nested:
                    span = segment.get("span") if isinstance(segment, dict) else None
                    if isinstance(span, list) and len(span) >= 2:
                        start, end = _float(span[0]), _float(span[1])
                        if start is not None and end is not None and end > start:
                            intervals.append((max(0.0, start), end))
            else:
                start = _float(item.get("clip_start_seconds"))
                end = _float(item.get("clip_end_seconds"))
                if start is not None and end is not None and end > start:
                    intervals.append((max(0.0, start), end))
    duration = float(source.get("duration") or 0)
    normalized = sorted(
        (min(start, duration), min(end, duration))
        for start, end in intervals
        if start < duration and end > 0
    )
    return [(start, end) for start, end in normalized if end > start]


def deterministic_speech_stats(source: dict[str, Any]) -> dict[str, float]:
    intervals = _speech_intervals(source)
    if not intervals:
        return {"coverage_ratio": 0.0, "longest_continuous_seconds": 0.0}
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.35:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    duration = max(float(source.get("duration") or 0), 1e-6)
    return {
        "coverage_ratio": min(1.0, sum(end - start for start, end in merged) / duration),
        "longest_continuous_seconds": max(end - start for start, end in merged),
    }


def clear_verified_events(
    verification: dict[str, Any], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    minimum_confidence = float(rules.get("verified_event_min_confidence", 0.85))
    allowed_prominence = set(
        rules.get("verified_event_allowed_prominence")
        or ("clear_foreground", "clear_background")
    )
    return [
        item
        for item in verification.get("verified_non_speech_events") or []
        if (
            isinstance(item, dict)
            and float(item.get("confidence") or 0) >= minimum_confidence
            and item.get("prominence") in allowed_prominence
            and str(item.get("event_type") or "").strip()
        )
    ]


def hard_gate(row: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    source = row.get("source") or {}
    assessment = row.get("assessment") or {}
    verification = row.get("independent_sound_verification") or {}
    reasons: list[str] = []
    duration = float(source.get("duration") or 0)
    bounds = rules["duration_seconds"]
    if not float(bounds["min"]) <= duration <= float(bounds["max"]):
        reasons.append(
            f"duration_outside_configured_range:{bounds['min']}..{bounds['max']}"
        )
    if rules.get("native_clip_only") and not is_native_clip(source):
        reasons.append("composite_or_merged_clip")

    screening_language = (assessment.get("language") or {}).get(
        "primary", "uncertain"
    )
    language = verification.get("speech_language") or screening_language
    if language == "no_speech":
        if not rules.get("allow_no_speech", False):
            reasons.append("no_speech_not_allowed")
    elif language not in set(rules.get("allowed_speech_languages") or []):
        reasons.append(f"language_not_allowed:{language}")

    quality = assessment.get("quality") or {}
    if rules.get("reject_corrupted_audio") and quality.get("audio_corrupted") is True:
        reasons.append("audio_corrupted")
    if (
        rules.get("reject_corrupted_or_repetitive_asr")
        and quality.get("asr_obviously_corrupted_or_repetitive") is True
    ):
        reasons.append("asr_corrupted_or_repetitive")

    if rules.get("reject_profanity"):
        categories = profanity_categories(source_asr_text(source))
        reasons.extend(f"profanity_in_source_asr:{item}" for item in categories)
        if bool((assessment.get("content_safety") or {}).get("contains_profanity")):
            reasons.append("profanity_detected_from_audio")

    if rules.get("independent_sound_verification", True):
        verified_events = clear_verified_events(verification, rules)
        non_speech_count = len({
            str(item["event_type"]).strip().lower() for item in verified_events
        })
        minimum = int(
            rules[
                "no_speech_min_non_speech_event_types"
                if language == "no_speech"
                else "speech_min_non_speech_event_types"
            ]
        )
        if non_speech_count < minimum:
            reasons.append(
                f"insufficient_independently_verified_non_speech_events:"
                f"{non_speech_count}<{minimum}"
            )
        if (
            rules.get("reject_pure_or_near_pure_speech", True)
            and verification.get("pure_or_near_pure_speech") is True
        ):
            reasons.append("pure_or_near_pure_speech")
        if rules.get("reject_indistinguishable_static_noise", True):
            if verification.get("background_too_noisy_or_masked") is True:
                reasons.append("background_too_noisy_or_masked")
            if verification.get("has_distinguishable_temporal_development") is not True:
                reasons.append("no_distinguishable_temporal_development")
        if (
            rules.get("reject_profanity")
            and (verification.get("content_safety") or {}).get("contains_profanity")
            is True
        ):
            reasons.append("profanity_detected_by_independent_relisten")
        if verification.get("decision") != "pass":
            reasons.append("independent_sound_verification_rejected")
    if rules.get("require_screening_pass", True) and assessment.get("decision") != "pass":
        reasons.append("screening_model_rejected")
    return list(dict.fromkeys(reasons))


def rank_key(row: dict[str, Any], rules: dict[str, Any]) -> tuple[Any, ...]:
    assessment = row.get("assessment") or {}
    quality = assessment.get("quality") or {}
    tier = {"priority": 3, "standard": 2, "low_priority": 1}.get(
        assessment.get("rank_tier"), 0
    )
    verification = row.get("independent_sound_verification") or {}
    verified_events = clear_verified_events(verification, rules)
    return (
        tier,
        len({str(item["event_type"]).lower() for item in verified_events}),
        int(verification.get("has_distinguishable_temporal_development") is True),
        float(quality.get("overall_clarity_score") or 0),
        float(quality.get("narrative_event_coherence_score") or 0),
        float(quality.get("event_boundary_clarity_score") or 0),
        float(quality.get("asr_reliability_score") or 0),
        float(assessment.get("complexity_score") or 0),
        len(assessment.get("non_speech_event_types") or []),
        len(assessment.get("temporal_stages") or []),
        -float((assessment.get("speech") or {}).get("ratio") or 0),
        str(row.get("id")),
    )


def select_with_progressive_cap(
    rows: list[dict[str, Any]], target: int, caps: list[int], rules: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    ranked = sorted(rows, key=lambda row: rank_key(row, rules), reverse=True)
    for cap in caps:
        selected: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in ranked:
            group = source_group_id(row["source"])
            if counts[group] >= cap:
                continue
            selected.append(row)
            counts[group] += 1
            if len(selected) >= target:
                return selected, cap
    return selected, caps[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()

    rules = load_config(args.config)["selection"]
    screening_path = args.root / "outputs" / f"{args.dataset}_screening.jsonl"
    rows = read_jsonl(screening_path)
    verifications = {
        str(row["id"]): row
        for row in read_jsonl(
            args.root / "outputs" / f"{args.dataset}_sound_verification.jsonl"
        )
    }
    eligible: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        verification_row = verifications.get(str(row["id"])) or {}
        row = {
            **row,
            "independent_sound_verification": verification_row.get(
                "verification"
            ),
            "independent_sound_verification_provenance": verification_row.get(
                "provenance"
            ),
            "deterministic_speech_stats": deterministic_speech_stats(
                row.get("source") or {}
            ),
        }
        reasons = hard_gate(row, rules)
        enriched = {
            **row,
            "production_selection": {
                "hard_gate_pass": not reasons,
                "rejection_reasons": reasons,
                "rank_key": list(rank_key(row, rules)),
                "config_path": str(args.config),
                "config_sha256": sha256_file(args.config),
            },
        }
        audit.append(enriched)
        if not reasons:
            eligible.append(enriched)

    caps = [int(value) for value in rules.get("source_caps_in_order") or [1]]
    selected, applied_cap = select_with_progressive_cap(
        eligible, args.target, caps, rules
    )
    selected_ids = {str(row["id"]) for row in selected}
    for row in audit:
        row["production_selection"]["selected_after_ranking"] = (
            str(row["id"]) in selected_ids
        )
        row["production_selection"]["applied_source_cap"] = applied_cap

    selected_sources = []
    for row in selected:
        assessment = row.get("assessment") or {}
        quality = assessment.get("quality") or {}
        selected_sources.append({
            **row["source"],
            "source_group_id": source_group_id(row["source"]),
            "independent_sound_verification": row.get(
                "independent_sound_verification"
            ),
            "deterministic_speech_stats": row.get(
                "deterministic_speech_stats"
            ),
            "selection_metadata": {
                "screening_language": (
                    row.get("independent_sound_verification") or {}
                ).get("speech_language")
                or (assessment.get("language") or {}).get("primary"),
                "speech_ratio": (assessment.get("speech") or {}).get("ratio"),
                "deterministic_speech_coverage_ratio": (
                    row.get("deterministic_speech_stats") or {}
                ).get("coverage_ratio"),
                "deterministic_longest_continuous_speech_seconds": (
                    row.get("deterministic_speech_stats") or {}
                ).get("longest_continuous_seconds"),
                "rank_tier": assessment.get("rank_tier"),
                "overall_clarity_score": quality.get("overall_clarity_score"),
                "narrative_event_coherence_score": quality.get(
                    "narrative_event_coherence_score"
                ),
                "event_boundary_clarity_score": quality.get("event_boundary_clarity_score"),
                "asr_reliability_score": quality.get("asr_reliability_score"),
                "complexity_score": assessment.get("complexity_score"),
                "non_speech_event_types": [
                    item.get("event_type")
                    for item in clear_verified_events(
                        row.get("independent_sound_verification") or {}, rules
                    )
                ],
                "temporal_stages": assessment.get("temporal_stages") or [],
                "applied_source_cap": applied_cap,
            },
        })

    write_jsonl(args.root / "outputs" / f"{args.dataset}_selection_audit.jsonl", audit)
    write_jsonl(args.root / "data" / args.dataset / "selected_manifest.jsonl", selected_sources)
    write_jsonl(args.root / args.dataset / "selected_manifest.jsonl", selected_sources)
    summary = {
        "dataset": args.dataset,
        "screened": len(rows),
        "hard_gate_pass": len(eligible),
        "selected": len(selected_sources),
        "unique_sources": len({row["source_group_id"] for row in selected_sources}),
        "applied_source_cap": applied_cap,
        "target_reached": len(selected_sources) >= args.target,
    }
    (args.root / "outputs" / f"{args.dataset}_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
