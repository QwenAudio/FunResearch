#!/usr/bin/env python3
"""Repository-local Chinese CER and biased-word error rate (B-WER)."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import kaldialign

from vendor.gsb_chinese_normalizer import normalize


ERROR = "*"
PUNCTUATION = set("!,?、。！，；？：「」︰『』《》")
SPACES = set(" \t\r\n")


@dataclass
class ErrorCounts:
    insertion: int = 0
    deletion: int = 0
    substitution: int = 0
    correct: int = 0
    reference: int = 0

    @property
    def errors(self) -> int:
        return self.insertion + self.deletion + self.substitution

    @property
    def rate(self) -> float:
        return 100.0 * self.errors / self.reference if self.reference else 0.0

    def add(self, other: "ErrorCounts") -> None:
        for field in (
            "insertion",
            "deletion",
            "substitution",
            "correct",
            "reference",
        ):
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def json(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(errors=self.errors, rate=self.rate)
        return value


def segment_text(segment: dict[str, Any], *, final: bool) -> str:
    if final:
        keys = ("text_final", "text_working", "text_original", "text")
    else:
        keys = ("text_original", "text_working", "text_final", "text")
    return next((str(segment[key]) for key in keys if segment.get(key)), "")


def normalize_text(text: str) -> str:
    return normalize(text)


def cer_tokens(text: str) -> list[str]:
    return list("".join(text.split()))


def characterize(text: str) -> list[str]:
    """Tokenization used by GigaSpeechBench's Chinese B-WER."""
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in PUNCTUATION:
            index += 1
            continue
        category = unicodedata.category(char)
        if category in {"Zs", "Cn"} or char in SPACES:
            index += 1
            continue
        if category == "Lo":
            result.append(char)
            index += 1
            continue
        separator = ">" if char == "<" else " "
        end = index + 1
        while end < len(text):
            candidate = text[end]
            if ord(candidate) >= 128 or candidate in SPACES or candidate == separator:
                break
            end += 1
        if end < len(text) and text[end] == ">":
            end += 1
        result.append(text[index:end])
        index = end
    return result


def bwer_tokens(text: str) -> list[str]:
    return [token.upper() for token in characterize(text)]


def entity_tokens(entities: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for entity in entities:
        result.update(bwer_tokens(str(entity)))
    return result


def alignment_counts(reference: list[str], hypothesis: list[str]) -> ErrorCounts:
    counts = ErrorCounts()
    for ref_token, hyp_token in kaldialign.align(reference, hypothesis, ERROR):
        if ref_token == ERROR:
            counts.insertion += 1
        elif hyp_token == ERROR:
            counts.deletion += 1
            counts.reference += 1
        elif ref_token != hyp_token:
            counts.substitution += 1
            counts.reference += 1
        else:
            counts.correct += 1
            counts.reference += 1
    return counts


def bwer_counts(
    reference: list[str], hypothesis: list[str], entities: Iterable[str]
) -> ErrorCounts:
    """Count errors only on reference tokens belonging to annotated entities."""
    counts = ErrorCounts()
    present_entity_tokens = entity_tokens(entities) & set(reference)
    for ref_token, hyp_token in kaldialign.align(reference, hypothesis, ERROR):
        if ref_token == ERROR:
            if hyp_token in present_entity_tokens:
                counts.insertion += 1
        elif ref_token in present_entity_tokens:
            counts.reference += 1
            if hyp_token == ERROR:
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
    """Evaluate one recording, requiring its segment boundaries to be unchanged."""
    reference_segments = _segments_by_id(reference_record)
    hypothesis_segments = _segments_by_id(hypothesis_record)
    if reference_segments.keys() != hypothesis_segments.keys():
        raise ValueError("reference and hypothesis segment ids differ")

    cer = ErrorCounts()
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

        ref_normalized = normalize_text(segment_text(ref, final=True))
        hyp_normalized = normalize_text(
            segment_text(hyp, final=hypothesis_is_final)
        )
        # This matches the official CER evaluator: an empty normalized reference
        # is not scored and insertions on it are ignored.
        if not ref_normalized.strip():
            continue
        cer.add(alignment_counts(cer_tokens(ref_normalized), cer_tokens(hyp_normalized)))
        biased.add(
            bwer_counts(
                bwer_tokens(ref_normalized),
                bwer_tokens(hyp_normalized),
                ref.get("entities") or [],
            )
        )
        scored_segments += 1

    return {"cer": cer, "bwer": biased, "segments": scored_segments}


def empty_result() -> dict[str, Any]:
    return {"cer": ErrorCounts(), "bwer": ErrorCounts(), "segments": 0}


def add_result(total: dict[str, Any], result: dict[str, Any]) -> None:
    total["cer"].add(result["cer"])
    total["bwer"].add(result["bwer"])
    total["segments"] += int(result["segments"])


def result_json(result: dict[str, Any]) -> dict[str, Any]:
    cer = result["cer"].json()
    bwer = result["bwer"].json()
    # Keep the older field names too, so reports remain easy to compare.
    return {
        "I": cer["insertion"],
        "D": cer["deletion"],
        "S": cer["substitution"],
        "C": cer["correct"],
        "N": cer["reference"],
        "segments": result["segments"],
        "errors": cer["errors"],
        "rate": cer["rate"],
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
