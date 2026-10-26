"""Language helpers used by controlled counterfactual validators."""

from __future__ import annotations

import re


HAN = re.compile(r"[\u3400-\u9fff]")
ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
QUOTED_PATTERNS = (
    re.compile(r'“([^”]+)”'),
    re.compile(r'「([^」]+)」'),
    re.compile(r'『([^』]+)』'),
    re.compile(r'"([^"]+)"'),
    re.compile(r"(?<![A-Za-z])'(.+?)'(?![A-Za-z])"),
)


def text_language(text: str | None) -> str:
    value = text or ""
    han = len(HAN.findall(value))
    english = len(ENGLISH_WORD.findall(value))
    if han == 0 and english == 0:
        return "other"
    if han and not english:
        return "zh"
    if english and not han:
        return "en"
    ratio = han / (han + english)
    return "zh" if ratio >= 0.7 else "en" if ratio <= 0.3 else "mixed"


def quoted_spans(text: str | None) -> list[str]:
    value = text or ""
    found: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern in QUOTED_PATTERNS[:-1]:
        for match in pattern.finditer(value):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            found.append((match.start(), match.group(1).strip()))
            occupied.append((match.start(), match.end()))
    positions = []
    for index, char in enumerate(value):
        if char != "'":
            continue
        previous = value[index - 1] if index else " "
        following = value[index + 1] if index + 1 < len(value) else " "
        if previous.isalnum() and following.isalnum():
            continue
        if not any(start <= index < end for start, end in occupied):
            positions.append(index)
    for start, end in zip(positions[0::2], positions[1::2]):
        if end > start + 1:
            found.append((start, value[start + 1:end].strip()))
    return [
        item for _, item in sorted(found)
        if len(item) >= 2 and text_language(item) != "other"
    ]


def strip_quoted_spans(text: str | None) -> str:
    value = text or ""
    for pattern in QUOTED_PATTERNS:
        value = pattern.sub(" ", value)
    return value


def question_frame_language(question: str) -> str:
    frame = text_language(strip_quoted_spans(question))
    return text_language(question) if frame == "other" else frame


def normalized_text(text: str | None) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", (text or "").lower())
