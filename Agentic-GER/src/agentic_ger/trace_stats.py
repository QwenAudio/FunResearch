#!/usr/bin/env python3
"""Read-only joins and funnel statistics for agent trace files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceStats:
    scan_passes: int
    suspects: int
    zero_suspects: bool
    decisions: list[dict[str, Any]]


def read_trace_stats(path: Path) -> TraceStats:
    """Join check evidence to commit decisions and count the scan funnel."""
    if not path.is_file():
        return TraceStats(0, 0, True, [])
    pending: dict[int, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    scan_passes = 0
    suspects = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event_name = event.get("event")
        if event_name == "scan.result":
            scan_passes += 1
            suspects += len(event.get("suspects") or [])
            continue
        if event_name == "check.result":
            pair = event.get("pair") or {}
            suspect = pair.get("suspect") or {}
            raw_segment_id = suspect.get("segment_id")
            segment_id = int(raw_segment_id) if raw_segment_id is not None else -1
            pending.setdefault(segment_id, []).append(event)
            continue
        if event_name != "patch.decision":
            continue
        patch = event.get("patch") or {}
        raw_segment_id = event.get("segment_id")
        if raw_segment_id is None:
            raw_segment_id = patch.get("segment_id")
        segment_id = int(raw_segment_id) if raw_segment_id is not None else -1
        waiting = pending.get(segment_id) or []
        check_event = waiting.pop(0) if waiting else {}
        pair = check_event.get("pair") or {}
        suspect = pair.get("suspect") or {}
        check = event.get("check") or check_event.get("decision") or {}
        decisions.append(
            {
                "decision": str(event.get("decision") or "unknown"),
                "segment_id": segment_id,
                "focus": str(patch.get("focus") or suspect.get("focus") or ""),
                "before": str(
                    patch.get("before_segment")
                    or pair.get("before_segment")
                    or ""
                ),
                "after": str(
                    patch.get("after_segment")
                    or check.get("edited_segment")
                    or pair.get("before_segment")
                    or ""
                ),
                "asr_text": str(patch.get("asr_text") or pair.get("asr_text") or ""),
                "reason": str(
                    check.get("reason")
                    or patch.get("reason")
                    or ""
                ),
                "evidence_source": check.get("evidence_source"),
                "baseline_spoken_form_valid": check.get(
                    "baseline_spoken_form_valid"
                ),
            }
        )
    return TraceStats(scan_passes, suspects, suspects == 0, decisions)
