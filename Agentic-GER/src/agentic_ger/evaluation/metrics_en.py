#!/usr/bin/env python3
"""Repository-local English WER and biased-word error rate (B-WER)."""

from __future__ import annotations

from typing import Any, Iterable

import kaldialign

from agentic_ger.evaluation.metrics_zh import ErrorCounts, alignment_counts, segment_text
from vendor.gsb_english_normalizer import normalize


def normalize_text(text: str) -> str:
    return normalize(text) or ""


def word_tokens(text: str) -> list[str]:
    return text.strip().upper().split()


def entity_tokens(entities: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for entity in entities:
        result.update(word_tokens(str(entity)))
    return result


def bwer_counts(
    reference: list[str], hypothesis: list[str], entities: Iterable[str]
) -> ErrorCounts:
    """Count errors on reference tokens annotated as English entities."""
    counts = ErrorCounts()
    present_entity_tokens = entity_tokens(entities) & set(reference)
    for ref_token, hyp_token in kaldialign.align(reference, hypothesis, "*"):
        if ref_token == "*":
            if hyp_token in present_entity_tokens:
                counts.insertion += 1
        elif ref_token in present_entity_tokens:
            counts.reference += 1
            if hyp_token == "*":
                counts.deletion += 1
            elif ref_token != hyp_token:
                counts.substitution += 1
            else:
                counts.correct += 1
    return counts


def _segments_by_id(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = record.get("segments") or []
    result = {int(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate segment id")
    return result


def evaluate_record(
    reference_record: dict[str, Any],
    hypothesis_record: dict[str, Any],
    *,
    hypothesis_is_final: bool,
    min_duration: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one English recording with unchanged segment boundaries."""
    reference_segments = _segments_by_id(reference_record)
    hypothesis_segments = _segments_by_id(hypothesis_record)
    if reference_segments.keys() != hypothesis_segments.keys():
        raise ValueError("reference and hypothesis segment ids differ")

    wer = ErrorCounts()
    biased = ErrorCounts()
    scored_segments = 0
    for segment_id, ref in reference_segments.items():
        hyp = hypothesis_segments[segment_id]
        ref_start, ref_end = float(ref["start"]), float(ref["end"])
        hyp_start, hyp_end = float(hyp["start"]), float(hyp["end"])
        if abs(ref_start - hyp_start) > 0.01 or abs(ref_end - hyp_end) > 0.01:
            raise ValueError(f"segment {segment_id} boundaries differ")
        if ref_end - ref_start <= min_duration:
            continue

        reference = word_tokens(normalize_text(segment_text(ref, final=True)))
        hypothesis = word_tokens(
            normalize_text(segment_text(hyp, final=hypothesis_is_final))
        )
        if not reference:
            continue
        wer.add(alignment_counts(reference, hypothesis))
        biased.add(bwer_counts(reference, hypothesis, ref.get("entities") or []))
        scored_segments += 1

    return {"wer": wer, "bwer": biased, "segments": scored_segments}


def empty_result() -> dict[str, Any]:
    return {"wer": ErrorCounts(), "bwer": ErrorCounts(), "segments": 0}


def add_result(total: dict[str, Any], result: dict[str, Any]) -> None:
    total["wer"].add(result["wer"])
    total["bwer"].add(result["bwer"])
    total["segments"] += int(result["segments"])


def result_json(result: dict[str, Any]) -> dict[str, Any]:
    wer = result["wer"].json()
    bwer = result["bwer"].json()
    return {
        "I": wer["insertion"],
        "D": wer["deletion"],
        "S": wer["substitution"],
        "C": wer["correct"],
        "N": wer["reference"],
        "segments": result["segments"],
        "errors": wer["errors"],
        "rate": wer["rate"],
        "hotword": {
            "b_wer": bwer["rate"],
            "errors": bwer["errors"],
            "reference": bwer["reference"],
            "I": bwer["insertion"],
            "D": bwer["deletion"],
            "S": bwer["substitution"],
            "C": bwer["correct"],
        },
    }
