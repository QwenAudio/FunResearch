"""Frozen inference protocols used by the FineAudio-Judge paper.

This module is deliberately model-free.  Prompt construction, deterministic
question ordering, and response parsing can therefore be audited and tested on
a CPU-only machine.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Iterable


SINGLE_RUBRIC_SYSTEM_PROMPT = (
    "你是一位专业的音频感知评估专家。请完整聆听音频，根据音频中实际可感知的内容判断题目描述是否正确。"
    "重点判断题目提到的声音或事件、语音内容、声音属性以及时间或共现关系是否与音频一致。"
    "不依赖文件名、构造元数据、背景知识或常识进行推断。描述正确时回答 yes；"
    "描述不符、相关内容不可辨识或证据不足时回答 no。最终只输出小写的 yes 或 no，"
    "不得输出解释、标点或其他内容。"
)

HOLISTIC_SYSTEM_PROMPT = (
    "You are an audio-text consistency evaluator. Listen to the entire candidate audio and judge how well "
    "the audio as a whole fulfills all requirements explicitly stated in the reference prompt. Assign one "
    "holistic consistency score from 0 to 100, where 0 means completely inconsistent and 100 means fully "
    "consistent. Return an integer and output exactly: {\"score\": <integer>}."
)


def rubric_id(row: dict[str, Any]) -> str:
    value = row.get("rubric_id", row.get("id"))
    if value is None or not str(value).strip():
        raise ValueError("every rubric requires rubric_id or id")
    return str(value).strip()


def validate_rubrics(rubrics: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rubrics)
    if not rows:
        raise ValueError("at least one rubric is required")
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("question", "")).strip():
            raise ValueError("every rubric must be an object with a non-empty question")
        identifiers.append(rubric_id(row))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("rubric IDs must be unique within one sample")
    return rows


def build_single_rubric_messages(
    question: str,
    criterion_type: str = "",
) -> list[dict[str, str]]:
    proposition = f"【{criterion_type.strip()}】{question.strip()}" if criterion_type.strip() else question.strip()
    return [
        {"role": "system", "content": SINGLE_RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"<audio>\n\n判断题：\n{proposition}"},
    ]


def parse_yes_no(raw: str) -> tuple[str | None, str]:
    stripped = raw.strip()
    if stripped in {"yes", "no"}:
        return stripped, "strict"
    lowered = stripped.lower()
    if lowered in {"yes", "no"}:
        return lowered, "case_normalized"
    normalized = lowered.strip(" \t\r\n.,;:!?，。；：！？'\"")
    if normalized in {"yes", "no"}:
        return normalized, "punctuation_normalized"
    matches = re.findall(r"(?<![a-z])(yes|no)(?![a-z])", lowered)
    if len(matches) == 1:
        return matches[0], "single_token_extracted"
    return None, "unparsed"


def _batch_prompt_prefix(question_count: int) -> list[str]:
    return [
        "你是一位专业的音频感知评估专家。请仔细聆听所附完整音频，并对以下判断题逐一作答（yes 或 no）。",
        "",
        "## 核心规则",
        "1. 只听不推断：仅根据音频中实际可感知的内容作答，不依赖文件名、构造元数据、背景知识或常识推断",
        "2. 方向已统一：所有题目均以 yes 表示音频符合题目所述特征",
        "3. 模糊从严：命题不符、目标不可辨识、被遮蔽或证据不足时回答 no",
        "",
        "## 输出格式",
        "严格输出 JSON 数组，数组长度必须与题目数量完全一致，不含任何其他内容。",
        '每项格式为：{"id":"原样返回题目ID","answer":"yes|no"}。',
        "answer 字段只能填写字面值 yes 或 no；不得用题目中的属性词（如 male、female、woman、fast、slow）替代。",
        "不得遗漏、增加、重复或修改任何 ID。",
        "",
        f"## 判断题（共 {question_count} 道）",
    ]


def build_batch_rubric_prompt(rubrics: Iterable[dict[str, Any]]) -> str:
    rows = validate_rubrics(rubrics)
    propositions = [
        {
            "id": rubric_id(row),
            "criterion_type": row.get("subtype"),
            "proposition": str(row["question"]),
        }
        for row in rows
    ]
    return "\n".join(
        _batch_prompt_prefix(len(rows))
        + [json.dumps(propositions, ensure_ascii=False, separators=(",", ":"))]
    )


def order_and_chunk_rubrics(
    sample_id: str,
    rubrics: Iterable[dict[str, Any]],
    *,
    chunk_size: int = 80,
    ordering_backend: str = "qwen",
    pass_index: int = 1,
) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rows = validate_rubrics(rubrics)
    seed_text = f"{sample_id}|structured_json|{ordering_backend}|{pass_index}|eval-majority-v1"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(rows)
    return [rows[index:index + chunk_size] for index in range(0, len(rows), chunk_size)]


def parse_batch_labels(
    raw: str,
    rubrics: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    rows = validate_rubrics(rubrics)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    candidate: Any = None
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            candidate = value
            break
        if isinstance(value, dict) and isinstance(value.get("labels"), list):
            candidate = value["labels"]
            break
    if not isinstance(candidate, list):
        raise ValueError("response contains no JSON label list")

    expected = [rubric_id(row) for row in rows]
    expected_set = set(expected)
    answers: dict[str, str] = {}
    for item in candidate:
        if not isinstance(item, dict):
            raise ValueError("non-object label item")
        identifier = str(item.get("id", "")).strip()
        if identifier not in expected_set and identifier.startswith("[") and "]" in identifier:
            bracketed = identifier[1:identifier.index("]")].strip()
            if bracketed in expected_set:
                identifier = bracketed
        answer = str(item.get("answer", "")).strip().lower()
        if identifier in answers:
            raise ValueError(f"duplicate ID: {identifier}")
        if answer not in {"yes", "no"}:
            raise ValueError(f"invalid answer for {identifier}: {answer}")
        answers[identifier] = answer
    missing = [identifier for identifier in expected if identifier not in answers]
    extra = sorted(set(answers) - expected_set)
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={missing}, extra={extra}")
    return [{"rubric_id": identifier, "answer": answers[identifier]} for identifier in expected]


def batch_guided_json_regex(rubrics: Iterable[dict[str, Any]]) -> str:
    rows = validate_rubrics(rubrics)
    whitespace = r"[ \t\r\n]{0,32}"
    items = [
        r'\{"id"'
        + whitespace + r":" + whitespace + r'"' + re.escape(rubric_id(row)) + r'"'
        + whitespace + r"," + whitespace + r'"answer"' + whitespace + r":" + whitespace
        + r'"(?:yes|no)"\}'
        for row in rows
    ]
    return r"\[" + whitespace + (whitespace + r"," + whitespace).join(items) + whitespace + r"\]"


def build_holistic_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": HOLISTIC_SYSTEM_PROMPT},
        {"role": "user", "content": "Candidate audio:\n<audio>\n\nReference prompt:\n" + prompt.strip()},
    ]


def parse_holistic_score(raw: str) -> tuple[int | None, str]:
    stripped = raw.strip()

    def valid(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and 0 <= value <= 100:
            return value
        if isinstance(value, float) and value.is_integer() and 0 <= value <= 100:
            return int(value)
        return None

    try:
        value = json.loads(stripped)
        if isinstance(value, dict) and (score := valid(value.get("score"))) is not None:
            return score, "strict_json"
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict) and (score := valid(value.get("score"))) is not None:
            return score, "fenced_json"
    except json.JSONDecodeError:
        pass
    match = re.search(r'["\']?score["\']?\s*:\s*(-?\d+(?:\.0+)?)', cleaned, flags=re.I)
    if match and (score := valid(float(match.group(1)))) is not None:
        return score, "score_field_extracted"
    if re.fullmatch(r"\d{1,3}", cleaned) and (score := valid(int(cleaned))) is not None:
        return score, "bare_integer"
    return None, "unparsed"
