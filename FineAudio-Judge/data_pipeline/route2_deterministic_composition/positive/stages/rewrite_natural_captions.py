#!/usr/bin/env python3
"""Rewrite timeline drafts while keeping selected speech attributes locked."""

from __future__ import annotations

import json
import hashlib
import re

import natural_caption_backend as base

legacy_validate = base.validate
legacy_call = base.call


def payload(row, language):
    return {
        "target_language": "Chinese" if language == "zh" else "English",
        "events": [{
            "event_id": event["event_id"], "role": event["role"],
            "locked_label": base.label(event, language),
            "start": event["start"], "end": event["end"],
            "speech_text": event.get("text"),
            "locked_speaker_phrase": event.get("speaker_description") if event["role"] == "speech" else None,
            "selected_speech_attributes": event.get("selected_caption_attrs") if event["role"] == "speech" else None,
        } for event in row["events"]],
        "relations_to_express": base.compact_relations(row),
        "background_mode": row["background_mode"],
    }


def prompt(row, language, errors):
    if language == "zh":
        rules = """
依据锁定标签和确定性混音时间线，将确定性草稿改写成自然、连贯的中文声音流描述。
硬规则：
- 每个非 Speech 的 locked_label 必须逐字出现，不得改名或遗漏。
- Speech 必须逐字包含 locked_speaker_phrase 和 speech_text；只写 selected_speech_attributes，不得从其它元数据补充年龄、情绪、口音、音色等属性。
- start/end 只用于表达先后、持续和重叠，正文不得出现秒数或时间戳。
- 自然表达全部 relations_to_express；不得虚构地点、画面、动作、因果、距离、音量或新声源。
- relation_evidence 必须逐一包含 relations_to_express 中的每个 relation_id；quote 必须是 caption 中逐字连续出现的原文片段。
- 不写“这段音频/录音”、Gain、核验过程，也不机械逐项报幕。
只返回 JSON：
{"caption":"...","covered_event_ids":["E1"],"relation_evidence":[{"relation_id":"R1","quote":"caption中的原文片段"}]}
"""
    else:
        rules = """
Rewrite the deterministic draft as one natural, coherent English description using only the locked labels and deterministic mix timeline.
Hard rules:
- Every non-speech locked_label must appear verbatim; never rename or omit it.
- Speech must contain locked_speaker_phrase and speech_text verbatim. Use only selected_speech_attributes; do not add age, emotion, accent, timbre, or any attribute from omitted metadata.
- Use start/end only for order, duration, and overlap. Do not write seconds or timestamps.
- Express all relations_to_express naturally. Invent no location, visuals, action, cause, distance, loudness, or new source.
- relation_evidence must include every relation_id in relations_to_express exactly once; each quote must be a verbatim contiguous substring of caption.
- Do not mention "this audio/recording", Gain, verification, or narrate events as a mechanical list.
Return JSON only:
{"caption":"...","covered_event_ids":["E1"],"relation_evidence":[{"relation_id":"R1","quote":"exact substring from caption"}]}
"""
    retry = ""
    if errors:
        retry = ("\n上一版错误，请整体重写：" if language == "zh" else "\nPrevious errors; rewrite fully: ") + json.dumps(errors, ensure_ascii=False)
    return rules + retry + "\nINPUT:\n" + json.dumps(payload(row, language), ensure_ascii=False)


def normalized(text):
    return "".join(re.findall(r"[\w\u4e00-\u9fff]+", str(text).casefold(), re.UNICODE))


def validate(row, language, value):
    errors = legacy_validate(row, language, value)
    caption = normalized(value.get("caption", ""))
    for event in row["events"]:
        if event["role"] == "speech" and normalized(event["speaker_description"]) not in caption:
            errors.append("missing_locked_speaker_phrase:" + event["event_id"])
    return errors


def distributed_call(row, language, base_urls, model, attempts):
    urls = [value.strip() for value in base_urls.split(",") if value.strip()]
    key = int(hashlib.sha256(f"{row['id']}::{language}".encode()).hexdigest()[:8], 16)
    return legacy_call(row, language, urls[key % len(urls)], model, attempts)


base.payload = payload
base.prompt = prompt
base.validate = validate
base.call = distributed_call

if __name__ == "__main__":
    base.main()
