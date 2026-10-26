from __future__ import annotations

import re
from typing import Any


_SPEECH_CUE = re.compile(
    r"(?:说|说道|说着|道|继续说|开口|低声道|喊道|喊|问道|问|回答|回应|补充|接话|"
    r"报告|播报|朗读|读道|齐声高喊|大喊|喝道|声音(?:响起|传来)|"
    r"\b(?:say|says|said|read|reads|replies|replied|responds|responded|adds?|added|"
    r"asks?|asked|shouts?|shouted|yells?|yelled|mutters?|muttered|concludes?|"
    r"concluded|continues?|continued|announces?|announced|calls? out|voice)\b)"
    r"[^。！？.!?]{0,72}$",
    re.IGNORECASE,
)
_METALINGUISTIC_CUE = re.compile(
    r"(?:提到|提及|说到|说出|念到|在说|(?:这个|该)?词(?:语)?|"
    r"\b(?:on|at)\s+the\s+(?:word|phrase)|\bas\s+(?:he|she|they|the speaker)\s+says?|"
    r"\bthe\s+(?:word|phrase|term))[^。！？.!?]{0,32}$",
    re.IGNORECASE,
)
_SOUND_CUE = re.compile(
    r"(?:音效|警示音|提示音|一声|拟声|"
    r"\b(?:sound(?: effect)?|noise|notification|chime|beep|ping|ding|click|clink)\b)"
    r"[^。！？.!?]{0,36}$",
    re.IGNORECASE,
)
_METALINGUISTIC_SUFFIX = re.compile(
    r"^\s*(?:一词|这个词|该词|\b(?:word|phrase|term)\b)",
    re.IGNORECASE,
)
_SOUND_SUFFIX = re.compile(
    r"^\s*(?:响|声(?:响|尖啸|传来|出现|响起)|音效|警示音|提示音|"
    r"\b(?:sound effect|sound|noise)\b)",
    re.IGNORECASE,
)
_STRONG_SOUND_CUE = re.compile(
    r"(?:模仿|猫叫|狗叫|动物叫|拟声|imitat(?:e|es|ed|ing)|mimic(?:s|ked|king)?)"
    r"[^。！？.!?]{0,48}$",
    re.IGNORECASE,
)


def normalize_quoted_content(value: str) -> str:
    """Normalize harmless typography for coverage comparison, not generation."""
    value = value.strip().casefold()
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n,，.。!?！？;；:：")


def _quotation_spans(prompt: str) -> list[dict[str, Any]]:
    patterns = (
        re.compile(r"“([^”\n]+)”"),
        re.compile(r"‘([^’\n]+)’"),
        re.compile(r'"([^"\n]+)"'),
    )
    matches: list[tuple[int, int, str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(prompt):
            matches.append((match.start(), match.end(), match.group(0), match.group(1).strip()))
    output = []
    occupied: list[tuple[int, int]] = []
    for start, end, raw, value in sorted(matches):
        if any(start >= left and end <= right for left, right in occupied):
            continue
        occupied.append((start, end))
        output.append({"start": start, "end": end, "raw": raw, "text": value})
    return output


def _classify_quotation(prompt: str, item: dict[str, Any]) -> str:
    left = prompt[max(0, item["start"] - 120):item["start"]]
    right = prompt[item["end"]:min(len(prompt), item["end"] + 48)]
    boundary = max(left.rfind(mark) for mark in "。！？.!?")
    clause_left = left[boundary + 1:]
    speech = bool(_SPEECH_CUE.search(clause_left))
    metalinguistic = bool(
        _METALINGUISTIC_CUE.search(clause_left) or _METALINGUISTIC_SUFFIX.search(right)
    )
    strong_sound = bool(_STRONG_SOUND_CUE.search(clause_left) or _SOUND_SUFFIX.search(right))
    sound = bool(_SOUND_CUE.search(clause_left))
    if strong_sound:
        return "sound_imitation"
    if metalinguistic:
        return "lexical_reference"

    # A colon after a nearby human voice/reporting context is a direct-speech
    # construction even when the reporting verb is farther away than one clause.
    colon = max(clause_left.rfind(":"), clause_left.rfind("："))
    if colon >= 0 and re.search(
        r"(?:人|记者|主播|指挥官|老人|女人|男人|他|她|声音|语气|"
        r"speaker|voice|narrator|barista|reporter|\bhe\b|\bshe\b)"
        r"[^。！？.!?]{0,80}$",
        clause_left[:colon],
        re.IGNORECASE,
    ):
        return "spoken_utterance"
    if sound:
        return "sound_imitation"
    if speech:
        return "spoken_utterance"
    if item["text"].lstrip().startswith(("...", "…")):
        return "spoken_utterance"
    return "unclassified_quotation"


def _silence_spans(prompt: str) -> list[dict[str, Any]]:
    patterns = (
        re.compile(r"(?:短暂|片刻|短促|短短|一阵|一片|完全|空旷)?(?:的)?(?:停顿|沉默|静默|寂静)"),
        re.compile(
            r"\b(?:(?:brief|short|momentary|prolonged|complete|ensuing)\s+)?"
            r"(?:pause|silence)\b",
            re.IGNORECASE,
        ),
    )
    matches: list[tuple[int, int, str]] = []
    for pattern in patterns:
        matches.extend((match.start(), match.end(), match.group(0)) for match in pattern.finditer(prompt))
    output = []
    last_end = -1
    for start, end, text in sorted(matches):
        if start < last_end:
            continue
        output.append({"start": start, "end": end, "text": text})
        last_end = end
    return output


def extract_prompt_obligations(prompt: str) -> dict[str, Any]:
    quotations = []
    for index, item in enumerate(_quotation_spans(prompt), 1):
        kind = _classify_quotation(prompt, item)
        quotations.append({
            "id": f"QUOTE_{index}",
            **item,
            "kind": kind,
            "requires_speech_event": kind == "spoken_utterance",
        })
    silences = [
        {"id": f"SILENCE_{index}", **item, "requires_silence_event": True}
        for index, item in enumerate(_silence_spans(prompt), 1)
    ]
    return {"quotations": quotations, "silences": silences}


def generation_obligations(prompt: str) -> dict[str, Any]:
    """Compact public contract supplied to the scene-graph generator."""
    extracted = extract_prompt_obligations(prompt)
    return {
        "spoken_utterances": [
            {"id": item["id"], "text": item["text"], "basis": item["raw"]}
            for item in extracted["quotations"] if item["requires_speech_event"]
        ],
        "non_speech_quotations": [
            {"id": item["id"], "text": item["text"], "classification": item["kind"]}
            for item in extracted["quotations"] if not item["requires_speech_event"]
        ],
        "explicit_silences": [
            {"id": item["id"], "basis": item["text"]}
            for item in extracted["silences"]
        ],
    }
