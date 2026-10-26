#!/usr/bin/env python3
"""Generate natural captions while locking source labels and timeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from typing import Any

import requests


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("model did not return JSON")
    return json.loads(match.group())


def label(event: dict[str, Any], language: str) -> str:
    return event["caption_label_zh"] if language == "zh" else event["caption_label_en"]


def compact_relations(row: dict[str, Any]) -> list[dict[str, Any]]:
    events = sorted(row["events"], key=lambda value: (value["start"], value["end"], value["event_id"]))
    result = []
    seen = set()
    for left, right in zip(events, events[1:]):
        relation = "overlap" if right["start"] < left["end"] - 0.08 else "before"
        key = (left["event_id"], relation, right["event_id"])
        if key not in seen:
            result.append({"relation_id": f"R{len(result)+1}", "source": key[0], "relation": key[1], "target": key[2]})
            seen.add(key)
    foreground = [event for event in events if event["role"] != "ambience"]
    for index, left in enumerate(foreground):
        for right in foreground[index + 1:]:
            if min(left["end"], right["end"]) - max(left["start"], right["start"]) <= 0.08:
                continue
            key = (left["event_id"], "overlap", right["event_id"])
            reverse = (right["event_id"], "overlap", left["event_id"])
            if key in seen or reverse in seen:
                continue
            result.append({"relation_id": f"R{len(result)+1}", "source": key[0], "relation": key[1], "target": key[2]})
            seen.add(key)
    return result


def payload(row: dict[str, Any], language: str) -> dict[str, Any]:
    return {
        "target_language": "Chinese" if language == "zh" else "English",
        "events": [
            {
                "event_id": event["event_id"], "role": event["role"],
                "locked_label": label(event, language), "start": event["start"], "end": event["end"],
                "speaker": event.get("speaker"), "speech_text": event.get("text"),
                "speaker_attributes": event.get("primary_attrs") if event["role"] == "speech" else None,
            }
            for event in row["events"]
        ],
        "relations_to_express": compact_relations(row),
        "background_mode": row["background_mode"],
    }


def prompt(row: dict[str, Any], language: str, errors: list[str]) -> str:
    if language == "zh":
        instructions = """
依据给定的源标签和确定性混音时间线，写一段自然、连贯的中文声音流描述。

硬规则：
- locked_label 是源数据集真值；每个非 Speech 标签必须逐字出现，不能改名、删减或凭听感替换。
- start/end 只用于理解顺序、持续和局部重叠；正文不得写秒数或时间戳。
- Speech 必须以自然的说话人作为主语，并逐字保留 speech_text；不要写成“带有某风格的说话声出现”。
- 准确表达 relations_to_express，但把关系融入自然叙述，避免逐事件报幕、连续分号和固定句式。
- 不必构造统一场景主题；随机声音可以直接按声音流组织。
- 不虚构地点、画面、动作、因果、音量、距离或新声源，不写“这段音频/录音”，不写 Gain。
- 篇幅由事件内容自然决定，不设字数限制。

只返回 JSON：
{"caption":"...","covered_event_ids":["E1"],"relation_evidence":[{"relation_id":"R1","quote":"caption中的原文片段"}]}
"""
    else:
        instructions = """
Using the locked source labels and deterministic mix timeline, write one natural, coherent English description of the evolving sound stream.

Hard rules:
- locked_label is source-dataset ground truth. Every non-speech label must appear verbatim; never rename, omit, or replace it with a perceptual guess.
- Use start/end only to understand order, duration, and partial overlap. Do not mention seconds or timestamps in the caption.
- Make a natural speaker the grammatical subject of Speech and preserve speech_text verbatim. Never write a mechanical phrase such as “a voice with ... style appears.”
- Express every relations_to_express item naturally. Avoid event-by-event announcements, semicolon chains, and repetitive templates.
- No common scene theme is required; unrelated sounds may simply be organized as a sound stream.
- Do not invent a place, visuals, actions, causes, loudness, distance, or a new source. Do not say “this audio/recording” and do not discuss Gain.
- Let detail follow the actual events; there is no word-count target.

Return JSON only:
{"caption":"...","covered_event_ids":["E1"],"relation_evidence":[{"relation_id":"R1","quote":"exact substring from caption"}]}
"""
    retry = ""
    if errors:
        retry = ("\n上一版错误，请整体重写：" if language == "zh" else "\nPrevious errors; rewrite the whole caption: ") + json.dumps(errors, ensure_ascii=False)
    return instructions + retry + "\nINPUT:\n" + json.dumps(payload(row, language), ensure_ascii=False)


def validate(row: dict[str, Any], language: str, value: dict[str, Any]) -> list[str]:
    caption = str(value.get("caption") or "").strip()
    errors = []
    if not caption:
        return ["empty_caption"]
    caption_for_match = caption if language == "zh" else caption.casefold()
    missing_labels = [
        label(event, language) for event in row["events"]
        if event["role"] != "speech" and (
            label(event, language) if language == "zh" else label(event, language).casefold()
        ) not in caption_for_match
    ]
    if missing_labels:
        errors.append("missing_labels:" + "|".join(missing_labels))
    def speech_match_text(text: str) -> str:
        return "".join(re.findall(r"[\w\u4e00-\u9fff]+", text.casefold(), re.UNICODE))
    caption_speech_match = speech_match_text(caption)
    missing_speech = [
        event["event_id"] for event in row["events"]
        if event["role"] == "speech" and speech_match_text(str(event.get("text") or "")) not in caption_speech_match
    ]
    if missing_speech:
        errors.append("missing_speech:" + "|".join(missing_speech))
    expected_ids = {event["event_id"] for event in row["events"]}
    if set(value.get("covered_event_ids") or []) != expected_ids:
        errors.append("covered_event_ids")
    def normalized(text: str) -> str:
        text = text.replace("“", '"').replace("”", '"').replace("’", "'")
        text = re.sub(r"\s+", " ", text).strip()
        return text if language == "zh" else text.casefold()
    normalized_caption = normalized(caption)
    evidence = {item.get("relation_id"): str(item.get("quote") or "") for item in value.get("relation_evidence") or []}
    for relation in compact_relations(row):
        quote = evidence.get(relation["relation_id"], "")
        if not quote or normalized(quote) not in normalized_caption:
            errors.append("relation_evidence:" + relation["relation_id"])
    if re.search(r"\d+(?:\.\d+)?\s*(?:秒|seconds?|s\b)", caption, re.I):
        errors.append("timestamp_in_natural_caption")
    lower = caption.lower()
    if (language == "zh" and ("这段音频" in caption or "录音中" in caption)) or (
        language == "en" and ("this audio" in lower or "this recording" in lower)
    ):
        errors.append("meta_narration")
    if re.search(r"\bgain\b", lower) or "混音增益" in caption:
        errors.append("gain")
    return errors


def repair_relation_evidence_quotes(
    row: dict[str, Any], language: str, value: dict[str, Any]
) -> dict[str, Any]:
    """Replace non-verbatim relation quotes with exact locked-label spans.

    This does not alter the caption. A repair is possible only when both locked
    labels occur in the caption and every `before` relation has the expected
    textual order; otherwise the original evidence is retained and validation
    still fails closed.
    """
    caption = str(value.get("caption") or "")
    search_caption = caption if language == "zh" else caption.casefold()
    events = {event["event_id"]: event for event in row["events"]}
    evidence = {
        item.get("relation_id"): dict(item)
        for item in value.get("relation_evidence") or []
        if item.get("relation_id")
    }
    for relation in compact_relations(row):
        source_label = label(events[relation["source"]], language)
        target_label = label(events[relation["target"]], language)
        source_search = source_label if language == "zh" else source_label.casefold()
        target_search = target_label if language == "zh" else target_label.casefold()
        source_start = search_caption.find(source_search)
        target_start = search_caption.find(target_search)
        if source_start < 0 or target_start < 0:
            continue
        if relation["relation"] == "before" and source_start >= target_start:
            continue
        start = min(source_start, target_start)
        end = max(source_start + len(source_label), target_start + len(target_label))
        evidence[relation["relation_id"]] = {
            "relation_id": relation["relation_id"],
            "quote": caption[start:end],
        }
    repaired = dict(value)
    repaired["relation_evidence"] = [
        evidence.get(relation["relation_id"], {"relation_id": relation["relation_id"], "quote": ""})
        for relation in compact_relations(row)
    ]
    return repaired


def repair_missing_speech_text(
    row: dict[str, Any], language: str, value: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Insert a locked transcript only when the model omitted that exact fact.

    The insertion is anchored immediately after the locked speaker phrase, so
    it does not invent wording or move the speech event to a different point in
    the described sequence. Relation evidence is then rebuilt from exact
    caption substrings and the full validator remains authoritative.
    """
    repaired = dict(value)
    caption = str(repaired.get("caption") or "")
    normalized_caption = "".join(
        re.findall(r"[\w\u4e00-\u9fff]+", caption.casefold(), re.UNICODE)
    )
    repairs = []
    for event in row["events"]:
        if event["role"] != "speech":
            continue
        transcript = str(event.get("text") or "")
        normalized_transcript = "".join(
            re.findall(r"[\w\u4e00-\u9fff]+", transcript.casefold(), re.UNICODE)
        )
        if not transcript or normalized_transcript in normalized_caption:
            continue
        speaker_phrase = str(event.get("speaker_description") or label(event, language))
        search_caption = caption if language == "zh" else caption.casefold()
        search_phrase = speaker_phrase if language == "zh" else speaker_phrase.casefold()
        start = search_caption.find(search_phrase)
        if start < 0:
            continue
        end = start + len(speaker_phrase)
        insertion = (
            f"，说出“{transcript}”"
            if language == "zh"
            else f', saying "{transcript}"'
        )
        suffix = caption[end:]
        if language == "en" and transcript.rstrip().endswith((".", "!", "?")) and suffix.startswith("."):
            suffix = suffix[1:]
        caption = caption[:end] + insertion + suffix
        # Relation-focused model retries sometimes repeat the full locked
        # speaker phrase in consecutive clauses. Keep the first exact phrase
        # and turn later mentions into a neutral anaphor without changing facts.
        if language == "en":
            first_end = caption.casefold().find(search_phrase) + len(speaker_phrase)
            while True:
                repeated = caption.casefold().find(search_phrase, first_end)
                if repeated < 0:
                    break
                caption = caption[:repeated] + "This speech" + caption[repeated + len(speaker_phrase):]
        normalized_caption = "".join(
            re.findall(r"[\w\u4e00-\u9fff]+", caption.casefold(), re.UNICODE)
        )
        repairs.append({
            "type": "insert_locked_speech_text",
            "event_id": event["event_id"],
            "anchor": speaker_phrase,
            "inserted_text": transcript,
        })
    repaired["caption"] = caption
    if repairs:
        repaired = repair_relation_evidence_quotes(row, language, repaired)
    return repaired, repairs


def repair_adjacent_speech_text(
    row: dict[str, Any], language: str, value: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Add quotation structure when a model glued a transcript to its speaker."""
    repaired = dict(value)
    caption = str(repaired.get("caption") or "")
    repairs = []
    for event in row["events"]:
        if event["role"] != "speech":
            continue
        speaker_phrase = str(event.get("speaker_description") or label(event, language))
        transcript = str(event.get("text") or "")
        glued = speaker_phrase + transcript
        if language == "zh" and transcript and f"“{transcript}”" not in caption:
            replacement_type = None
            for prefix in ("内容为", "说出"):
                plain = prefix + transcript
                if plain in caption:
                    caption = caption.replace(plain, f"{prefix}“{transcript}”", 1)
                    replacement_type = "quote_locked_speech_text"
                    break
            if replacement_type is None:
                for separator in ("，", ","):
                    plain = speaker_phrase + separator + transcript
                    if plain in caption:
                        caption = caption.replace(
                            plain, f"{speaker_phrase}，说出“{transcript}”", 1
                        )
                        replacement_type = "separate_locked_speaker_and_text"
                        break
            if replacement_type is None and glued in caption:
                caption = caption.replace(
                    glued, f"{speaker_phrase}，说出“{transcript}”", 1
                )
                replacement_type = "separate_locked_speaker_and_text"
            if replacement_type:
                repairs.append({
                    "type": replacement_type,
                    "event_id": event["event_id"],
                    "anchor": speaker_phrase,
                    "inserted_text": transcript,
                })
        if language == "en":
            quoted = f'"{transcript}"'
            plain = f"saying {transcript}"
            if transcript and quoted not in caption and plain in caption:
                caption = caption.replace(plain, f'saying "{transcript}"', 1)
                repairs.append({
                    "type": "quote_locked_speech_text",
                    "event_id": event["event_id"],
                    "anchor": speaker_phrase,
                    "inserted_text": transcript,
                })
            awkward_connector = f"followed by {speaker_phrase}"
            connector_start = caption.casefold().find(awkward_connector.casefold())
            if connector_start >= 0:
                speaker_start = connector_start + len("followed by ")
                matched_speaker = caption[speaker_start:speaker_start + len(speaker_phrase)]
                caption = (
                    caption[:connector_start]
                    + f"and then {matched_speaker}"
                    + caption[connector_start + len(awkward_connector):]
                )
                repairs.append({
                    "type": "normalize_speech_clause_connector",
                    "event_id": event["event_id"],
                    "anchor": speaker_phrase,
                    "inserted_text": "and then",
                })
            search_phrase = speaker_phrase.casefold()
            first = caption.casefold().find(search_phrase)
            if first >= 0:
                cursor = first + len(speaker_phrase)
                while True:
                    repeated = caption.casefold().find(search_phrase, cursor)
                    if repeated < 0:
                        break
                    caption = caption[:repeated] + "This speech" + caption[repeated + len(speaker_phrase):]
                    cursor = repeated + len("This speech")
                    repairs.append({
                        "type": "deduplicate_locked_speaker_phrase",
                        "event_id": event["event_id"],
                        "anchor": speaker_phrase,
                        "inserted_text": "This speech",
                    })
            if transcript.rstrip().endswith((".", "!", "?")) and '".' in caption:
                updated = caption.replace('".', '"', 1)
                if updated != caption:
                    caption = updated
                    repairs.append({
                        "type": "normalize_quoted_transcript_punctuation",
                        "event_id": event["event_id"],
                        "anchor": speaker_phrase,
                        "inserted_text": transcript,
                    })
    repaired["caption"] = caption
    if repairs:
        repaired = repair_relation_evidence_quotes(row, language, repaired)
    return repaired, repairs


def call(row: dict[str, Any], language: str, base_url: str, model: str, attempts: int) -> dict[str, Any]:
    errors: list[str] = []
    value: dict[str, Any] = {}
    raw = ""
    deterministic_repairs: list[dict[str, str]] = []
    for attempt in range(1, attempts + 1):
        user_prompt = prompt(row, language, errors)
        if errors and value:
            correction = (
                "\n上一版无效 JSON（只修正报告的问题，仍须满足全部规则）：\n"
                if language == "zh"
                else "\nPrevious invalid JSON (fix the reported issues while preserving every rule):\n"
            )
            user_prompt += correction + json.dumps(value, ensure_ascii=False)
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Follow locked facts exactly and write natural audio-only captions."},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.35, "top_p": 0.9, "max_tokens": 1400,
                "response_format": {"type": "json_object"},
            }, timeout=240,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        try:
            value = extract_json(raw)
            value, formatting_repairs = repair_adjacent_speech_text(row, language, value)
            deterministic_repairs.extend(formatting_repairs)
            errors = validate(row, language, value)
            if errors and all(error.startswith("relation_evidence:") for error in errors):
                value = repair_relation_evidence_quotes(row, language, value)
                errors = validate(row, language, value)
            if errors and all(error.startswith("missing_speech:") for error in errors):
                value, speech_repairs = repair_missing_speech_text(row, language, value)
                deterministic_repairs.extend(speech_repairs)
                errors = validate(row, language, value)
        except Exception as exc:
            errors = [f"parse:{type(exc).__name__}"]
        if not errors:
            break
    return {
        "job_id": row["id"] + "::" + language, "sample_id": row["id"], "language": language,
        "caption": value.get("caption", ""),
        "covered_event_ids": value.get("covered_event_ids", []),
        "relation_evidence": value.get("relation_evidence", []),
        "deterministic_repairs": deterministic_repairs,
        "validation_errors": errors, "attempts": attempt, "model": model, "raw": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--jobs-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    by_id = {row["id"]: row for row in rows}
    existing = read_jsonl(args.jobs_output) if args.resume and args.jobs_output.exists() else []
    for item in existing:
        row = by_id[item["sample_id"]]
        value = {
            "caption": item.get("caption", ""),
            "covered_event_ids": item.get("covered_event_ids", []),
            "relation_evidence": item.get("relation_evidence", []),
        }
        value, formatting_repairs = repair_adjacent_speech_text(row, item["language"], value)
        if formatting_repairs:
            item["caption"] = value["caption"]
            item["relation_evidence"] = value["relation_evidence"]
            item["deterministic_repairs"] = [
                *(item.get("deterministic_repairs") or []), *formatting_repairs,
            ]
            item["validation_errors"] = validate(row, item["language"], value)
        if item.get("validation_errors") and all(
            error.startswith("relation_evidence:") for error in item["validation_errors"]
        ):
            try:
                value = extract_json(str(item.get("raw") or ""))
            except Exception:
                value = dict(item)
            value = repair_relation_evidence_quotes(row, item["language"], value)
            item["caption"] = value.get("caption", item.get("caption", ""))
            item["covered_event_ids"] = value.get("covered_event_ids", [])
            item["relation_evidence"] = value["relation_evidence"]
            item["validation_errors"] = validate(row, item["language"], value)
        if item.get("validation_errors") and all(
            error.startswith("missing_speech:") for error in item["validation_errors"]
        ):
            value = {
                "caption": item.get("caption", ""),
                "covered_event_ids": item.get("covered_event_ids", []),
                "relation_evidence": item.get("relation_evidence", []),
            }
            value, repairs = repair_missing_speech_text(row, item["language"], value)
            item["caption"] = value["caption"]
            item["relation_evidence"] = value["relation_evidence"]
            item["deterministic_repairs"] = [
                *(item.get("deterministic_repairs") or []), *repairs,
            ]
            item["validation_errors"] = validate(row, item["language"], value)
    completed = {item["job_id"] for item in existing if not item.get("validation_errors")}
    jobs = [(row, language) for row in rows for language in row["caption_languages"] if row["id"] + "::" + language not in completed]
    results = {item["job_id"]: item for item in existing if not item.get("validation_errors")}
    args.jobs_output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call, row, language, args.base_url, args.model, args.attempts): (row["id"], language) for row, language in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            results[item["job_id"]] = item
            with args.jobs_output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(json.dumps({"generated": index, "pending_at_start": len(jobs)}, ensure_ascii=False), flush=True)
    ordered_jobs = [results[row["id"] + "::" + language] for row in rows for language in row["caption_languages"]]
    args.jobs_output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered_jobs), encoding="utf-8")
    failed = [item for item in ordered_jobs if item["validation_errors"]]
    final_rows = []
    for row in rows:
        value = dict(row)
        value["caption_natural_by_language"] = {
            language: results[row["id"] + "::" + language]["caption"] for language in row["caption_languages"]
        }
        value["caption_rewrite_audit_by_language"] = {
            language: {key: results[row["id"] + "::" + language].get(key) for key in ("validation_errors", "attempts", "relation_evidence", "deterministic_repairs", "model")}
            for language in row["caption_languages"]
        }
        final_rows.append(value)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows), encoding="utf-8")
    print(json.dumps({"samples": len(final_rows), "caption_jobs": len(ordered_jobs), "failed": len(failed), "output": str(args.output)}, ensure_ascii=False))
    if failed:
        raise SystemExit("one or more caption rewrite jobs failed")


if __name__ == "__main__":
    main()
