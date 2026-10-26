"""Deterministic dataset policies shared by selection and validation stages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


MACHINE_TIMECODE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\d{1,2}):)?(?:[0-5]?\d):(?:[0-5]\d)(?:\.\d+)?(?!\d)"
)
FINEVIDEO_CLIP_SUFFIX = re.compile(r"_s\d{9}_e\d{9}$")
QUOTED_SPAN_PATTERNS = (
    re.compile(r'“([^”\n]{1,500})”'),
    re.compile(r'"([^"\n]{1,500})"'),
)


def machine_timecodes(value: Any) -> list[str]:
    """Return explicit MM:SS/HH:MM:SS markers that must not enter captions."""
    return sorted(set(MACHINE_TIMECODE_PATTERN.findall(str(value or ""))))


def quoted_spans(value: Any) -> list[str]:
    """Extract quoted audible text that derived captions must preserve."""
    text = str(value or "")
    return list(dict.fromkeys(
        match.group(1).strip()
        for pattern in QUOTED_SPAN_PATTERNS
        for match in pattern.finditer(text)
        if match.group(1).strip()
    ))


def source_group_id(source: dict[str, Any]) -> str:
    """Return a stable upstream-video group for diversity caps and split safety."""
    dataset = str(source.get("dataset") or "unknown").lower()
    locator = source.get("source_locator") or {}
    sample_id = str(source.get("id") or locator.get("clip_id") or "unknown")

    if dataset == "vript":
        meta = source.get("source_meta") or {}
        explicit = source.get("source_group_id")
        if str(explicit).startswith("vript:"):
            return str(explicit)
        video_id = (
            explicit
            or meta.get("video_id")
            or locator.get("video_id")
            or sample_id.rsplit("-Scene-", 1)[0]
        )
        return f"vript:{video_id}"

    if dataset == "finevideo":
        explicit = source.get("source_group_id")
        if str(explicit).startswith("finevideo:"):
            explicit = str(explicit).removeprefix("finevideo:")
        original = (
            explicit
            or locator.get("original_video_filename")
            or (source.get("finevideo_original_json") or {}).get(
                "original_video_filename"
            )
        )
        raw_video_id = Path(str(original)).stem if original else sample_id
        video_id = FINEVIDEO_CLIP_SUFFIX.sub("", raw_video_id)
        return f"finevideo:{video_id}"

    return f"{dataset}:{sample_id}"
