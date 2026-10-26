#!/usr/bin/env python3
"""Coverage gate and source-language caption finalization."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CORE_CODE = HERE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def reject(path: Path, row: dict[str, Any], stage: str, reasons: list[str], **extra: Any) -> None:
    append_jsonl(path, {"id": row.get("id"), "dataset": row.get("dataset"), "stage": stage, "rejection_reasons": reasons, **extra})


def caption_prompt(
    caption: str,
    language: str,
    speech_reference: str | None = None,
    previous: Any = None,
) -> str:
    target = "English" if language == "en" else "Chinese"
    repair = "" if previous is None else (
        "\nPrevious result and deterministic errors to repair:\n"
        + json.dumps(previous, ensure_ascii=False)[:16000]
    )
    return f"""
Convert only the narration of the verified audio-only caption below into {target}
and return one production caption. Preserve every audible event, spoken item,
background sound, temporal relation, level of detail, event order, overlap, and
ending. Do not summarize, shorten, add, remove, or merge audible facts.

Verified caption:
{caption}

ASR reference for audible speech (may be empty when there is no reliable speech):
{speech_reference or "[no reliable speech]"}
{repair}

Return JSON only:
{{
  "audio_caption": "the sole production caption in {language}",
  "caption_language": "{language}",
  "checks": {{
    "same_audible_facts": true,
    "production_language_matches_target": true,
    "spoken_content_preserved_verbatim": true,
    "production_has_no_inline_translation": true,
    "contains_machine_timecode": false,
    "contains_profanity": false
  }}
}}

Rules:
1. audio_caption is the only training and published caption. Its narration must
   be in {target}. Do not append translations in parentheses or create a bilingual
   caption.
2. Every directly quoted utterance must remain in its original language and keep
   the wording supported by the verified audio caption. The ASR reference is
   supporting evidence, not infallible ground truth. Never translate speech inside
   quotation marks or invent a quotation when no speech evidence exists.
3. Preserve the original level of detail and all audible temporal relationships.
   Do not simplify or shorten the caption for fluency.
4. Replace machine timecodes with natural relative timing. Stay strictly audio-only.
5. Report the checks truthfully. Do not hide profanity by masking or rewriting it.
6. Do not output markdown, explanations, or text outside the JSON object.
""".strip()


def direct_quotes(text: str) -> list[str]:
    """Extract paired direct-quote spans while ignoring apostrophes in words."""
    quotes: list[str] = []
    for opening, closing in (("\"", "\""), ("“", "”"), ("'", "'"), ("‘", "’")):
        start: int | None = None
        for index, char in enumerate(text):
            if char not in {opening, closing}:
                continue
            if opening == closing and char == "'":
                previous = text[index - 1] if index else ""
                following = text[index + 1] if index + 1 < len(text) else ""
                if previous.isalnum() and following.isalnum():
                    continue
            if start is None and char == opening:
                start = index + 1
            elif start is not None and char == closing:
                span = text[start:index].strip()
                if span:
                    quotes.append(span)
                start = None
    return quotes


def normalized_words(text: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", text.casefold()).strip()


def caption_errors(
    value: dict[str, Any],
    language: str,
    speech_reference: str | None = None,
    check_audit: bool = True,
) -> list[str]:
    errors: list[str] = []
    caption = value.get("audio_caption")
    audit = value.get("audit_caption_zh")
    if not isinstance(caption, str) or len(caption.strip()) < 80:
        errors.append("missing_or_short_audio_caption")
    if check_audit and (not isinstance(audit, str) or len(audit.strip()) < 40):
        errors.append("missing_or_short_audit_caption_zh")
    if value.get("caption_language") != language:
        errors.append("caption_language_mismatch")
    if isinstance(caption, str) and re.search(r"\b\d{1,2}:\d{2}\b", caption):
        errors.append("production_contains_machine_timecode")
    if check_audit and isinstance(audit, str) and re.search(r"\b\d{1,2}:\d{2}\b", audit):
        errors.append("audit_contains_machine_timecode")
    if language == "en" and isinstance(caption, str) and re.search(r"[\u3400-\u9fff]", caption):
        errors.append("production_contains_chinese_text")
    if check_audit and isinstance(caption, str) and isinstance(audit, str):
        production_quotes = sorted(
            normalized_words(quote) for quote in direct_quotes(caption) if normalized_words(quote)
        )
        audit_quotes = sorted(
            normalized_words(quote) for quote in direct_quotes(audit) if normalized_words(quote)
        )
        if production_quotes != audit_quotes:
            errors.append(
                "audit_quotes_must_match_production_verbatim:"
                + json.dumps(
                    {"production": production_quotes, "audit": audit_quotes},
                    ensure_ascii=False,
                )[:1200]
            )
    checks = value.get("checks") or {}
    for name in (
        "same_audible_facts",
        "production_language_matches_target",
        "spoken_content_preserved_verbatim",
        "production_has_no_inline_translation",
    ):
        if checks.get(name) is not True:
            errors.append("failed_check:" + name)
    if checks.get("contains_machine_timecode") is not False:
        errors.append("model_reported_machine_timecode")
    if checks.get("contains_profanity") is not False:
        errors.append("model_reported_profanity")
    return errors


def mask_direct_quotes(text: str) -> tuple[str, list[str]]:
    masked = text
    quotes = direct_quotes(text)
    for index, quote in enumerate(quotes):
        token = f"__VERBATIM_QUOTE_{index}__"
        for wrapped in (f'"{quote}"', f'“{quote}”', f"'{quote}'", f"‘{quote}’"):
            if wrapped in masked:
                masked = masked.replace(wrapped, token, 1)
                break
    return masked, quotes


def audit_prompt(masked_caption: str, tokens: list[str], previous: Any = None) -> str:
    repair = "" if previous is None else (
        "\nPrevious result and deterministic errors to repair:\n"
        + json.dumps(previous, ensure_ascii=False)[:12000]
    )
    return f"""
Translate only the narration of this audio caption into natural Chinese.
Preserve every audible fact, level of detail, event order, overlap, temporal
relation, and ending. Do not summarize, shorten, add, remove, or merge audible
facts.

The tokens below replace original-language speech and must each appear exactly
once, unchanged, and in the same order. Never translate, remove, duplicate, or
put quotation marks around a token.

Required tokens:
{json.dumps(tokens, ensure_ascii=False)}

Masked caption:
{masked_caption}
{repair}

Return JSON only:
{{"audit_caption_zh": "Chinese audit narration containing every required token"}}
""".strip()


def audit_errors(value: dict[str, Any], tokens: list[str]) -> list[str]:
    text = value.get("audit_caption_zh")
    if not isinstance(text, str) or len(text.strip()) < 40:
        return ["missing_or_short_audit_caption_zh"]
    errors = []
    narration = text
    for token in tokens:
        narration = narration.replace(token, "")
    if len(re.findall(r"[\u3400-\u9fff]", narration)) < 8:
        errors.append("audit_narration_is_not_chinese")
    for token in tokens:
        if text.count(token) != 1:
            errors.append(f"token_count:{token}={text.count(token)}")
    return errors


def restore_audit_quotes(text: str, quotes: list[str]) -> str:
    for index, quote in enumerate(quotes):
        text = text.replace(f"__VERBATIM_QUOTE_{index}__", f'"{quote}"')
    return text


MACHINE_TIMECODE_RE = re.compile(
    r"(?<!\d)(?:(?:\d{1,2}):)?(?:[0-5]?\d):(?:[0-5]\d)(?:\.\d+)?(?!\d)"
)
QUOTED_TEXT_RE = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\'[^\'\n]*\'(?!\w)|‘[^’\n]*’'
)


def timecode_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def relative_time_phrase(seconds: float, duration: float, language: str) -> str:
    ratio = seconds / max(duration, 1.0)
    if language == "zh":
        if ratio <= 0.15:
            return "音频开头"
        if ratio <= 0.35:
            return "音频前段"
        if ratio <= 0.65:
            return "音频中段"
        if ratio <= 0.85:
            return "音频后段"
        return "接近音频结尾"
    if ratio <= 0.15:
        return "at the beginning of the audio"
    if ratio <= 0.35:
        return "early in the audio"
    if ratio <= 0.65:
        return "in the middle of the audio"
    if ratio <= 0.85:
        return "later in the audio"
    return "near the end of the audio"


def naturalize_machine_timecodes(
    text: str, duration: float, language: str
) -> tuple[str, list[dict[str, Any]]]:
    """Replace narration timecodes with relative timing without touching quotes."""
    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        token = f"__QUOTED_TEXT_{len(protected)}__"
        protected.append(match.group(0))
        return token

    value = QUOTED_TEXT_RE.sub(protect, text)
    changes: list[dict[str, Any]] = []

    def phrase(match: re.Match[str]) -> str:
        token = match.group("timecode")
        replacement = relative_time_phrase(timecode_seconds(token), duration, language)
        if language != "zh" and match.start() == 0 and match.group(0)[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        changes.append({"timecode": token, "relative_timing": replacement})
        return replacement

    if language == "zh":
        value = re.sub(
            r"(?:持续至|一直持续到|持续到)\s*(?:约|大约)?\s*(?P<timecode>(?:(?:\d{1,2}):)?(?:[0-5]?\d):(?:[0-5]\d)(?:\.\d+)?)(?:处|时)?",
            lambda match: "持续至" + phrase(match),
            value,
        )
        value = re.sub(
            r"(?:约在|大约在|在约|在大约|在)\s*(?P<timecode>(?:(?:\d{1,2}):)?(?:[0-5]?\d):(?:[0-5]\d)(?:\.\d+)?)(?:处|时)?",
            phrase,
            value,
        )
        value = re.sub(
            r"(?P<timecode>(?:(?:\d{1,2}):)?(?:[0-5]?\d):(?:[0-5]\d)(?:\.\d+)?)",
            phrase,
            value,
        )
    else:
        value = re.sub(
            r"(?:at\s+approximately|at\s+around|approximately|around|at)\s+(?P<timecode>(?:(?:\d{1,2}):)?(?:[0-5]?\d):(?:[0-5]\d)(?:\.\d+)?)",
            phrase,
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"(?P<timecode>(?:(?:\d{1,2}):)?(?:[0-5]?\d):(?:[0-5]\d)(?:\.\d+)?)",
            phrase,
            value,
        )
    for index, quoted in enumerate(protected):
        value = value.replace(f"__QUOTED_TEXT_{index}__", quoted)
    return value, changes


def reliable_speech_reference(language_row: dict[str, Any]) -> str | None:
    segments = language_row.get("whisper_segments") or []
    texts = [
        str(segment.get("text") or "").strip()
        for segment in segments
        if segment.get("reliable") is True and str(segment.get("text") or "").strip()
    ]
    return " ".join(texts) or None


def relaxed_gate_errors(errors: list[str], policy: dict[str, Any]) -> list[str]:
    """Keep safety/validity gates while making semantic coverage auditable."""
    kept = []
    for error in errors:
        if not policy.get("require_every_coverage_check", True) and error.startswith("invalid_coverage_checks:"):
            continue
        if not policy.get("reject_reported_omissions", True) and error == "omitted_audible_evidence_reported":
            continue
        if not policy.get("require_all_dialogue_spans_in_both_languages", True) and error.startswith("verbatim_dialogue_missing_from_"):
            continue
        kept.append(error)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(CORE_CODE))
    from content_safety import profanity_categories
    from data_policy import machine_timecodes, source_group_id
    from prompts import caption_coverage_prompt
    from text_backend import VllmTextBatchRunner, caption_gate_errors

    config = json.loads(args.config.read_text(encoding="utf-8"))
    generation = config["generation"]
    caption_gates = config.get("caption_gates") or {}
    output = args.run_root / args.dataset / "caption_candidates.jsonl"
    rejected = args.run_root / args.dataset / "rejected.jsonl"
    if args.overwrite:
        output.unlink(missing_ok=True)
        rejected.unlink(missing_ok=True)
    done = {str(row["id"]) for row in read_jsonl(output)} | {str(row["id"]) for row in read_jsonl(rejected)}
    rows = [
        row for row in read_jsonl(args.run_root / "outputs" / f"{args.dataset}_audio_analysis.jsonl")
        if str(row["id"]) not in done
    ]
    safe = []
    for row in rows:
        categories = profanity_categories({"source_asr": row["source"].get("source_asr"), "audio_caption": row.get("audio_caption")})
        if categories:
            reject(rejected, row, "pre_caption_content_safety", ["profanity_detected:" + ",".join(categories)])
        else:
            safe.append(row)
    rows = safe
    runner = VllmTextBatchRunner(
        generation["text_model"], int(generation["text_tensor_parallel"]),
        max_num_seqs=32, max_model_len=16384, submit_batch_size=64,
    )
    prompts = [
        caption_coverage_prompt(
            row["audio_caption"], row.get("investigation_rounds"),
            row["source"].get("source_asr"), row["source"].get("trusted_time_annotations"),
            (row["source"].get("independent_sound_verification") or {}).get("verified_non_speech_events"),
        ) for row in rows
    ]
    gates = runner.generate_json(prompts, 3200)
    accepted: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for row, (gate, raw) in zip(rows, gates, strict=True):
        errors = relaxed_gate_errors(
            caption_gate_errors(gate, detective=True), caption_gates
        )
        if errors:
            reject(rejected, row, "audio_coverage_gate", errors, gate=gate, raw=raw)
        else:
            accepted.append((row, gate, raw))
    language_rows = {
        str(value["id"]): value
        for value in read_jsonl(args.run_root / "outputs" / f"{args.dataset}_language_gate.jsonl")
    }
    no_speech_default = str((config.get("caption_language") or {}).get("no_speech_default", "en"))
    targets = []
    speech_references = []
    prompts = []
    for row, gate, _ in accepted:
        language_row = language_rows.get(str(row["id"])) or {}
        speech_reference = reliable_speech_reference(language_row)
        has_reliable_speech = speech_reference is not None
        detected = language_row.get("whisper_overall_language")
        target = detected if has_reliable_speech and detected in {"zh", "en"} else no_speech_default
        targets.append(target)
        speech_references.append(speech_reference)
        prompts.append(caption_prompt(gate["audio_caption"], target, speech_reference))
    caption_results = runner.generate_json(prompts, 4200)
    for _ in range(3):
        bad = [
            index for index, (value, _) in enumerate(caption_results)
            if caption_errors(
                value, targets[index], speech_references[index], check_audit=False
            )
        ]
        if not bad:
            break
        repairs = runner.generate_json([
            caption_prompt(
                accepted[index][1]["audio_caption"], targets[index], speech_references[index],
                 {"previous_result": caption_results[index][0],
                 "machine_errors": caption_errors(
                     caption_results[index][0], targets[index], speech_references[index]
                     , check_audit=False
                 )},
            ) for index in bad
        ], 4200)
        for index, repaired in zip(bad, repairs, strict=True):
            caption_results[index] = repaired
    timecode_normalizations: list[list[dict[str, Any]]] = []
    for index, (result, raw) in enumerate(caption_results):
        caption = result.get("audio_caption")
        changes: list[dict[str, Any]] = []
        if isinstance(caption, str):
            normalized, changes = naturalize_machine_timecodes(
                caption, float(accepted[index][0]["source"].get("duration") or 0),
                targets[index],
            )
            result["audio_caption"] = normalized
            if not MACHINE_TIMECODE_RE.search(normalized):
                (result.setdefault("checks", {}))["contains_machine_timecode"] = False
        caption_results[index] = (result, raw)
        timecode_normalizations.append(changes)
    masked_captions = []
    quote_lists = []
    token_lists = []
    for result, _ in caption_results:
        masked, quotes = mask_direct_quotes(str(result.get("audio_caption") or ""))
        masked_captions.append(masked)
        quote_lists.append(quotes)
        token_lists.append([f"__VERBATIM_QUOTE_{index}__" for index in range(len(quotes))])
    audit_results = runner.generate_json([
        audit_prompt(masked, tokens)
        for masked, tokens in zip(masked_captions, token_lists, strict=True)
    ], 4200)
    for _ in range(3):
        bad_audits = [
            index for index, (value, _) in enumerate(audit_results)
            if audit_errors(value, token_lists[index])
        ]
        if not bad_audits:
            break
        repairs = runner.generate_json([
            audit_prompt(
                masked_captions[index], token_lists[index],
                {"previous_result": audit_results[index][0],
                 "machine_errors": audit_errors(audit_results[index][0], token_lists[index])},
            ) for index in bad_audits
        ], 4200)
        for index, repaired in zip(bad_audits, repairs, strict=True):
            audit_results[index] = repaired
    for index, ((result, raw), (audit, audit_raw)) in enumerate(zip(
        caption_results, audit_results, strict=True
    )):
        if not audit_errors(audit, token_lists[index]):
            result["audit_caption_zh"] = restore_audit_quotes(
                audit["audit_caption_zh"], quote_lists[index]
            )
            caption_results[index] = (result, raw + "\nAUDIT_RAW:\n" + audit_raw)
    for (row, gate, gate_raw), target, speech_reference, (result, result_raw), timecode_changes in zip(
        accepted, targets, speech_references, caption_results, timecode_normalizations, strict=True
    ):
        errors = caption_errors(result, target, speech_reference)
        if errors:
            reject(rejected, row, "source_language_caption_gate", errors, result=result)
            continue
        source = row["source"]
        language_row = language_rows.get(str(row["id"])) or {}
        has_reliable_speech = speech_reference is not None
        append_jsonl(output, {
            "id": row["id"], "dataset": source.get("dataset"),
            "pipeline": "route1_audio_grounded",
            "audio_duration": source.get("duration"), "audio_path": source["audio_path"],
            "video_path": source.get("video_path"), "source_locator": source.get("source_locator"),
            "source_group_id": source_group_id(source),
            "speech_language": target if has_reliable_speech else None,
            "caption_language": target,
            "language_basis": "detected_speech" if has_reliable_speech else "no_reliable_speech_default",
            "audio_caption": result["audio_caption"],
            "audit_caption_zh": result["audit_caption_zh"],
            "audio_caption_source_timed": gate.get("audio_caption") if machine_timecodes(gate.get("audio_caption")) else None,
            "caption_language_policy": "match_detected_speech_language_preserve_spoken_content_verbatim",
            "independent_sound_verification": source.get("independent_sound_verification"),
            "selection_metadata": source.get("selection_metadata") or {},
            "provenance": {
                "route": "audio_grounded_source_language_caption",
                "protocol": config["protocol"],
                "caption_model": row.get("model"), "text_gate_model": generation["text_model"],
                "pre_gate_audio_caption": row.get("audio_caption"), "coverage_gate": gate,
                "coverage_gate_raw_output": gate_raw, "caption_raw_output": result_raw,
                "caption_timecode_normalization": timecode_changes,
                "speech_reference": speech_reference if has_reliable_speech else None,
                "caption_stage_trace": row.get("provenance"),
            },
        })
        print(json.dumps({"id": row["id"], "stage": "finalize_route_a"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
