#!/usr/bin/env python3
"""Traceable, label-blind Route 1 prompt-negative generation pipeline.

This pipeline deliberately never reads perceptual label files.  Every valid
Route 1 source row is treated as positive-by-construction. It edits only the
prompt, keeps the source audio byte-identical, rebuilds only affected rubrics,
and emits unlabeled P'-A pairs for a later, separate labeling stage.

Stages:
    prepare -> generate -> materialize -> rebuild -> audit

Model-calling stages require both an explicit local OpenAI-compatible endpoint
and the literal execution token ``ROUTE1_NEGATIVE_GENERATION``.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
try:
    from .language_repair_common import normalized_text, question_frame_language, quoted_spans
except ImportError:  # Direct script execution.
    from language_repair_common import normalized_text, question_frame_language, quoted_spans

DEFAULT_INPUT = None
DEFAULT_MODEL = "reproducer_selected_text_model"
EXECUTION_TOKEN = "ROUTE1_NEGATIVE_GENERATION"
SCHEMA_VERSION = "route1-negative-unlabeled"

CATEGORIES = ("attribute", "relation", "event", "all_three")
CATEGORY_STRENGTH = {
    "attribute": "medium",
    "relation": "medium",
    "event": "strong",
    "all_three": "strong",
}
TARGET_CHANGED_FRACTION = {"medium": 0.25, "strong": 0.50}
STRENGTH_RANGE = {"medium": (0.10, 0.40), "strong": (0.35, 0.85)}

RELATION_SUBTYPES = {"before", "after", "overlap", "background_of", "foreground_of"}
RELATION_FLIPS = {
    "before": {"after", "overlap"},
    "after": {"before", "overlap"},
    "overlap": {"before", "after"},
    "background_of": {"foreground_of"},
    "foreground_of": {"background_of"},
}
PATCH_RELATION_FLIPS = {
    "before": {"overlap"},
    "after": {"overlap"},
    "overlap": {"before", "after"},
    "background_of": {"foreground_of"},
    "foreground_of": {"background_of"},
}
ATTRIBUTE_DENYLIST = {
    "exists", "occurrence_count", "count", "frequency", "absolute_time",
    "start_time", "end_time", "timestamp", "time_range", "specific_time",
}
ALLOWED_QUESTION_KEYS = {
    "rubric_id", "tuple_id", "axis", "scope", "domain", "subtype",
    "question", "basis", "basis_start", "basis_end", "target_ids", "parent_ids",
    "evidence_type",
}
FORBIDDEN_LABEL_KEYS = {
    "answer", "answers", "expected_answer", "label", "labels", "loss_mask",
    "agree_yes", "agree_no", "uncertain", "baseline_consensus_yes_ratio",
}
VISUAL_RE = re.compile(
    r"\b(?:screen|on[- ]?screen|camera|visual|video|frame|footage|logo|subtitle|"
    r"caption box|text overlay|colour|color|visible|visually|shown|wearing|"
    r"looks? like|appears? on|see a|watch the|sparks? (?:fly|flying|shower(?:ing)?)|"
    r"vibrat(?:es|ing)\s+(?:the|a)\s+\w+|stored kinetic energy|rotating shaft|"
    r"friction of (?:metal )?gears?|vibration of (?:the|a)\s+\w+)\b|"
    r"(?:画面|镜头|屏幕|字幕|标志|颜色|可见|看到|身穿|火花飞溅)",
    re.IGNORECASE,
)
EXPLICIT_ABSENCE_RE = re.compile(
    r"\b(?:no|not|without|absence|omit(?:s|ted)?|silence)\b|"
    r"(?:没有|不再|未出现|缺少|静音)",
    re.IGNORECASE,
)
STALE_SPEECH_CONTENT_RE = re.compile(
    r"\b(?:explain(?:s|ed|ing)?|describe(?:s|d|ing)?|discuss(?:es|ed|ing)?|"
    r"mention(?:s|ed|ing)?|narrat(?:es|ed|ing|ion)|monologue|dialogue|spoken words?|"
    r"cover(?:s|ing) the (?:use|operation|functionality|principles?|topic|experiment))\b|"
    r"(?:讲解|解释|描述|提到|叙述|独白|对白|台词内容|讲话内容)",
    re.IGNORECASE,
)
MUSIC_GENRE_RE = re.compile(
    r"\b(?:rock|pop|jazz|classical|orchestral|electronic|acoustic|hip[- ]?hop|"
    r"rap|heavy[- ]?metal|metal music|folk|country|blues|dubstep|chiptune|bossa nova)\b",
    re.IGNORECASE,
)
ACCENT_RE = re.compile(r"\b(?:accent|drawl|rhotic|british|american|australian)\b", re.IGNORECASE)
PROPER_NAME_RE = re.compile(r"(?<![.!?]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
EVENT_FAMILY_ALIASES = {
    "speech": "speech",
    "voice": "speech",
    "spoken": "speech",
    "dialogue": "speech",
    "music": "music",
    "musical": "music",
    "song": "music",
    "sound_effect": "sound_effect",
    "sound effect": "sound_effect",
    "sfx": "sound_effect",
    "vocalization": "vocalization",
    "animal": "animal",
    "animal_sound": "animal",
    "ambient": "ambience",
    "ambience": "ambience",
    "environment": "ambience",
    "mechanical": "mechanical",
    "machinery": "mechanical",
    "vehicle": "vehicle",
    "impact": "impact",
    "footstep": "footsteps",
    "footsteps": "footsteps",
    "other": "other",
}
EVENT_FAMILY_PATTERNS = {
    "speech": re.compile(r"\b(?:speech|voice|speaker|narrator|dialogue|monologue|says?|saying|speaks?|speaking|shouts?|whispers?)\b|(?:说话|旁白|对白|讲话|喊|低语)", re.I),
    "music": re.compile(r"\b(?:music|musical|song|melody|instrumental|piano|guitar|orchestral|electronic track)\b|(?:音乐|歌曲|旋律|钢琴|吉他)", re.I),
    "mechanical": re.compile(r"\b(?:machine|machinery|mechanical|engine|motor|industrial|grinding|clanking)\b|(?:机器|机械|引擎|马达|工业)", re.I),
    "animal": re.compile(r"\b(?:animal|dog|bark|cat|meow|bird|chirp|roar|insect)\b|(?:动物|狗叫|猫叫|鸟叫|鸣叫|咆哮)", re.I),
    "vehicle": re.compile(r"\b(?:vehicle|car|truck|train|aircraft|motorcycle|horn)\b|(?:车辆|汽车|卡车|火车|飞机|摩托|喇叭)", re.I),
    "footsteps": re.compile(r"\b(?:footstep|footsteps|walking|running steps)\b|(?:脚步|走路声|跑步声)", re.I),
    "impact": re.compile(r"\b(?:impact|crash|bang|knock|clack|collision|explosion)\b|(?:撞击|碰撞|爆炸|敲击)", re.I),
    "ambience": re.compile(r"\b(?:ambience|ambient|rain|wind|thunder|water|crowd noise|room tone)\b|(?:环境声|雨声|风声|雷声|水声|人群声)", re.I),
    "vocalization": re.compile(r"\b(?:cheer|laughter|laughing|crying|scream|cough)\b|(?:欢呼|笑声|哭声|尖叫|咳嗽)", re.I),
}
EVENT_FAMILY_REFERENCE_TERMS = {
    "speech": r"speech|voice|narration|narrator|speaker|dialogue|monologue|说话|语音|旁白|对白|讲话",
    "music": r"music|song|melody|instrumental|track|音乐|歌曲|旋律",
    "sound_effect": r"sound effect|effect|sound|音效|声音",
    "vocalization": r"vocalization|cheer|laughter|crying|scream|cough|欢呼|笑声|哭声|尖叫|咳嗽",
    "animal": r"animal sound|bark|meow|chirp|roar|动物声|狗叫|猫叫|鸟叫|咆哮",
    "ambience": r"ambience|ambient sound|rain|wind|room tone|环境声|雨声|风声",
    "mechanical": r"mechanical sound|machine|machinery|engine|motor|机械声|机器声|引擎声|马达声",
    "vehicle": r"vehicle sound|car|truck|train|aircraft|motorcycle|horn|车辆声|汽车声|卡车声|火车声|飞机声|摩托声|喇叭声",
    "impact": r"impact|crash|bang|knock|collision|explosion|撞击声|碰撞声|爆炸声|敲击声",
    "footsteps": r"footstep|footsteps|walking|running steps|脚步声",
}
QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "during",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "there", "this", "to", "until", "while", "with",
}
DANGLING_BASIS_END_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: str) -> str:
    return text_sha256("://".join(parts))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def output_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "created_at": now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "script": str(script),
        "script_sha256": file_sha256(script),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_tracked_status": git_value("status", "--porcelain", "--untracked-files=no"),
    }


def snapshot_code_and_config(run: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    script = Path(__file__).resolve()
    script_sha = file_sha256(script)
    script_snapshot = run / f"provenance/code/{script_sha}.py"
    if not script_snapshot.exists():
        script_snapshot.parent.mkdir(parents=True, exist_ok=True)
        script_snapshot.write_bytes(script.read_bytes())
    config_text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    config_sha = text_sha256(config_text)
    config_snapshot = run / f"provenance/config_snapshots/{config_sha}.json"
    if not config_snapshot.exists():
        config_snapshot.parent.mkdir(parents=True, exist_ok=True)
        config_snapshot.write_text(config_text, encoding="utf-8")
    return script_snapshot, config_snapshot


@contextmanager
def run_lock(run: Path) -> Iterator[None]:
    run.mkdir(parents=True, exist_ok=True)
    lock_path = run / "state/pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another pipeline process owns {lock_path}") from exc
        handle.seek(0)
        handle.truncate(0)
        handle.write(f"pid={os.getpid()} started_at={now()}\n")
        handle.flush()
        yield


def record_stage(run: Path, stage: str, status: str, details: dict[str, Any]) -> None:
    script = Path(__file__).resolve()
    script_sha = file_sha256(script)
    script_snapshot = run / f"provenance/code/{script_sha}.py"
    if not script_snapshot.exists():
        script_snapshot.parent.mkdir(parents=True, exist_ok=True)
        script_snapshot.write_bytes(script.read_bytes())
    event = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": now(),
        "stage": stage,
        "status": status,
        "pid": os.getpid(),
        "argv": sys.argv,
        "script_sha256": script_sha,
        "script_snapshot": str(script_snapshot),
        **details,
    }
    append_jsonl(run / "provenance/stage_events.jsonl", event)
    atomic_write_json(run / f"state/{stage}.json", event)


@contextmanager
def stage_event(run: Path, stage: str, details: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    payload = dict(details or {})
    record_stage(run, stage, "started", payload)
    try:
        yield payload
    except Exception as exc:
        record_stage(run, stage, "failed", {**payload, "error": repr(exc)})
        raise
    else:
        record_stage(run, stage, "completed", payload)


def assert_label_blind_input(path: Path) -> None:
    if "labels" in {part.lower() for part in path.parts}:
        raise ValueError(f"label-blind pipeline refuses input under a labels directory: {path}")


def row_id(row: dict[str, Any], *, location: str) -> str:
    value = row.get("sample_id") or row.get("id")
    if value is None or not str(value).strip():
        raise ValueError(f"{location}: missing sample_id/id")
    return str(value)


def public_join_key(row: dict[str, Any], *, location: str) -> tuple[str | None, str]:
    dataset = row.get("dataset")
    return (str(dataset) if dataset is not None else None, row_id(row, location=location))


def adapt_public_inputs(positive_path: Path, rubric_path: Path) -> list[dict[str, Any]]:
    """Join the released positive manifest with an independently produced rubric manifest."""
    positive_rows = read_jsonl(positive_path)
    rubric_rows = read_jsonl(rubric_path)
    positives_by_full: dict[tuple[str | None, str], dict[str, Any]] = {}
    positives_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(positive_rows, 1):
        key = public_join_key(row, location=f"{positive_path}:{index}")
        if key in positives_by_full:
            raise ValueError(f"duplicate positive join key: {key}")
        positives_by_full[key] = row
        positives_by_id[key[1]].append(row)

    rubrics_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    for index, row in enumerate(rubric_rows, 1):
        key = public_join_key(row, location=f"{rubric_path}:{index}")
        if key in rubrics_by_key:
            raise ValueError(f"duplicate rubric join key: {key}")
        rubrics_by_key[key] = row

    adapted: list[dict[str, Any]] = []
    matched_rubric_keys: set[tuple[str | None, str]] = set()
    for key, positive in positives_by_full.items():
        rubric_key = key
        rubric = rubrics_by_key.get(rubric_key)
        if rubric is None and key[0] is not None:
            rubric_key = (None, key[1])
            rubric = rubrics_by_key.get(rubric_key)
            if rubric is not None and len(positives_by_id[key[1]]) > 1:
                raise ValueError(
                    f"ambiguous rubric ID {key[1]!r}; include dataset in the rubric manifest"
                )
        if rubric is None:
            continue
        if key[0] is None and len(positives_by_id[key[1]]) > 1:
            raise ValueError(f"ambiguous positive ID without dataset: {key[1]}")
        matched_rubric_keys.add(rubric_key)
        caption = str(positive.get("audio_caption") or positive.get("caption") or "").strip()
        if not caption:
            raise ValueError(f"positive row {key} has no audio_caption")
        rubric_prompt = str(rubric.get("prompt") or rubric.get("audio_caption") or "").strip()
        if rubric_prompt and rubric_prompt != caption:
            raise ValueError(f"caption/rubric prompt mismatch for {key}")
        questions = rubric.get("questions") or rubric.get("rubrics") or []
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"rubric row {rubric_key} has no questions/rubrics")
        sample_id = key[1] if key[0] is None else f"{key[0]}:{key[1]}"
        adapted.append({
            "route": "path1_whole_real_audio",
            "sample_id": sample_id,
            "source_id": key[1],
            "dataset": key[0],
            "caption": caption,
            "audio_path": positive.get("audio_path"),
            "audio_duration_seconds": positive.get("audio_duration_seconds", positive.get("audio_duration")),
            "questions": questions,
        })
    unmatched = sorted(set(rubrics_by_key) - matched_rubric_keys, key=str)
    if unmatched:
        raise ValueError(f"rubric rows have no matching positive row: {unmatched[:5]}")
    if not adapted:
        raise ValueError("no positive/rubric rows matched")
    return adapted


def sanitize_question(question: dict[str, Any]) -> dict[str, Any]:
    leaked = FORBIDDEN_LABEL_KEYS & set(question)
    if leaked:
        raise ValueError(f"rubric contains forbidden label keys: {sorted(leaked)}")
    clean = {key: deepcopy(value) for key, value in question.items() if key in ALLOWED_QUESTION_KEYS}
    required = {"rubric_id", "scope", "subtype", "question", "basis"}
    missing = sorted(required - set(clean))
    if missing:
        raise ValueError(f"rubric missing required keys: {missing}")
    return clean


def assert_no_label_keys(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        bad = FORBIDDEN_LABEL_KEYS & set(value)
        if bad:
            raise ValueError(f"label keys leaked into frozen input at {location}: {sorted(bad)}")
        for key, child in value.items():
            assert_no_label_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_label_keys(child, f"{location}[{index}]")


def language(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "zh" if cjk >= max(8, latin // 4) else "en"


def freeze_source_row(row: dict[str, Any], hash_audio: bool = True) -> dict[str, Any]:
    if row.get("route") != "path1_whole_real_audio":
        raise ValueError("not a Route 1 source row")
    prompt = str(row.get("caption") or "")
    if not prompt.strip():
        raise ValueError(f"empty caption: {row.get('sample_id')}")
    audio_path = Path(str(row.get("audio_path") or ""))
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    questions = [sanitize_question(item) for item in row.get("questions") or []]
    if not questions:
        raise ValueError(f"no rubrics: {row.get('sample_id')}")
    ids = [str(item["rubric_id"]) for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate rubric ID: {row.get('sample_id')}")
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(row["sample_id"]),
        "source_id": str(row.get("source_id") or row["sample_id"]),
        "source_group_id": str(row["sample_id"]),
        "route": "path1_whole_real_audio",
        "source_assumption": "positive_by_construction",
        "label_visibility": "none",
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt),
        "language": language(prompt),
        "audio_path": str(audio_path),
        "audio_duration_seconds": row.get("audio_duration_seconds"),
        "audio_sha256": file_sha256(audio_path) if hash_audio else None,
        "questions": questions,
    }
    assert_no_label_keys(frozen)
    return frozen


def rubric_kind(question: dict[str, Any]) -> str | None:
    if question.get("subtype") == "exists" and not (question.get("parent_ids") or []):
        return "event"
    if question.get("scope") == "relational" and question.get("subtype") in RELATION_SUBTYPES:
        return "relation"
    if question.get("scope") == "local" and question.get("subtype") not in ATTRIBUTE_DENYLIST:
        return "attribute"
    return None


def target_rejection(question: dict[str, Any], prompt: str) -> str | None:
    kind = rubric_kind(question)
    if kind is None:
        return "unsupported_rubric_kind"
    basis = str(question.get("basis") or "").strip()
    if not basis:
        return "empty_basis"
    if prompt.count(basis) != 1:
        return "basis_not_unique_verbatim"
    if VISUAL_RE.search(f"{basis}\n{question.get('question') or ''}"):
        return "visual_or_non_audio_target"
    if kind == "relation" and len(question.get("target_ids") or []) < 2:
        return "relation_missing_endpoints"
    return None


def descendants(root: str, children: dict[str, set[str]]) -> set[str]:
    found = {root}
    pending = [root]
    while pending:
        for child in children[pending.pop()]:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def build_choice(source: dict[str, Any], category: str) -> dict[str, Any] | None:
    questions = {item["rubric_id"]: item for item in source["questions"]}
    all_ids = set(questions)
    children: dict[str, set[str]] = defaultdict(set)
    for question in source["questions"]:
        for parent in question.get("parent_ids") or []:
            if parent in all_ids:
                children[parent].add(question["rubric_id"])
    groups: dict[str, list[tuple[str, set[str]]]] = {key: [] for key in ("event", "attribute", "relation")}
    for rubric_id, question in questions.items():
        kind = rubric_kind(question)
        if kind is None or target_rejection(question, source["prompt"]):
            continue
        groups[kind].append((rubric_id, descendants(rubric_id, children)))
    for kind in groups:
        groups[kind].sort(key=lambda item: stable_key(source["sample_id"], category, kind, item[0]))

    required: list[tuple[str, set[str]]] = []
    if category == "all_three":
        if any(not groups[kind] for kind in ("event", "attribute", "relation")):
            return None
        required = [groups[kind][0] for kind in ("event", "attribute", "relation")]
        pool = groups["event"] + groups["attribute"] + groups["relation"]
    else:
        if not groups[category]:
            return None
        pool = list(groups[category])
        required = [pool[0]]

    strength = CATEGORY_STRENGTH[category]
    target_count = max(1, math.ceil(len(all_ids) * TARGET_CHANGED_FRACTION[strength]))
    direct: set[str] = set()
    affected: set[str] = set()
    for rubric_id, closure in required:
        direct.add(rubric_id)
        affected.update(closure)
    pool.sort(key=lambda item: (
        abs(target_count - len(item[1])),
        stable_key(source["sample_id"], category, "pool", item[0]),
    ))
    for rubric_id, closure in pool:
        if len(affected) >= target_count:
            break
        if closure <= affected:
            continue
        direct.add(rubric_id)
        affected.update(closure)
    if not direct:
        return None
    if category == "all_three":
        kinds = {rubric_kind(questions[rubric_id]) for rubric_id in direct}
        if not {"event", "attribute", "relation"}.issubset(kinds):
            return None
    changed_fraction = len(affected) / len(all_ids)
    low, high = STRENGTH_RANGE[strength]
    if not low <= changed_fraction <= high:
        return None
    return {
        "direct_target_rubric_ids": sorted(direct),
        "hierarchy_affected_rubric_ids": sorted(affected),
        "retained_rubric_ids": sorted(all_ids - affected),
        "direct_target_count": len(direct),
        "affected_count": len(affected),
        "total_rubrics": len(all_ids),
        "construction_changed_fraction": round(changed_fraction, 6),
        "intended_perturbation_strength": strength,
    }


class MinCostFlow:
    def __init__(self, size: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(size)]

    def add(self, source: int, target: int, capacity: int, cost: int = 0) -> None:
        forward = [target, capacity, len(self.graph[target]), cost]
        reverse = [source, 0, len(self.graph[source]), -cost]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def flow(self, source: int, sink: int, requested: int) -> tuple[int, int]:
        size = len(self.graph)
        potential = [0] * size
        sent = 0
        total_cost = 0
        infinity = 10**30
        while sent < requested:
            distance = [infinity] * size
            previous_node = [-1] * size
            previous_edge = [-1] * size
            distance[source] = 0
            heap: list[tuple[int, int]] = [(0, source)]
            while heap:
                current_distance, node = heapq.heappop(heap)
                if current_distance != distance[node]:
                    continue
                for edge_index, edge in enumerate(self.graph[node]):
                    target, capacity, _, cost = edge
                    if not capacity:
                        continue
                    candidate = current_distance + cost + potential[node] - potential[target]
                    if candidate < distance[target]:
                        distance[target] = candidate
                        previous_node[target] = node
                        previous_edge[target] = edge_index
                        heapq.heappush(heap, (candidate, target))
            if distance[sink] == infinity:
                break
            for node in range(size):
                if distance[node] < infinity:
                    potential[node] += distance[node]
            node = sink
            while node != source:
                parent = previous_node[node]
                edge_index = previous_edge[node]
                edge = self.graph[parent][edge_index]
                edge[1] -= 1
                self.graph[node][edge[2]][1] += 1
                total_cost += edge[3]
                node = parent
            sent += 1
        return sent, total_cost


def balanced_quotas(count: int) -> dict[str, int]:
    base, remainder = divmod(count, len(CATEGORIES))
    return {category: base + (index < remainder) for index, category in enumerate(CATEGORIES)}


def assign_balanced(
    options: dict[str, dict[str, dict[str, Any]]], requested: int,
) -> tuple[dict[str, str], dict[str, int]]:
    sample_ids = sorted(options, key=lambda sid: (len(options[sid]), stable_key("sample", sid)))
    target_count = len(sample_ids) if requested <= 0 else min(requested, len(sample_ids))
    quotas = balanced_quotas(target_count)
    source_node = 0
    sample_offset = 1
    category_offset = sample_offset + len(sample_ids)
    sink = category_offset + len(CATEGORIES)
    graph = MinCostFlow(sink + 1)
    sample_index = {sid: sample_offset + index for index, sid in enumerate(sample_ids)}
    category_index = {category: category_offset + index for index, category in enumerate(CATEGORIES)}
    for sid in sample_ids:
        graph.add(source_node, sample_index[sid], 1)
        for category, choice in options[sid].items():
            tie_break = int(stable_key(sid, category)[:4], 16) % 100
            cost = choice["direct_target_count"] * 100000 + choice["affected_count"] * 100 + tie_break
            graph.add(sample_index[sid], category_index[category], 1, cost)
    for category, quota in quotas.items():
        graph.add(category_index[category], sink, quota)
    achieved, _ = graph.flow(source_node, sink, target_count)
    if achieved != target_count:
        raise ValueError(
            f"balanced assignment infeasible: {achieved}/{target_count}; "
            "set --target-count lower or revise category policy"
        )
    reverse_category = {node: category for category, node in category_index.items()}
    assignment: dict[str, str] = {}
    for sid in sample_ids:
        for target, capacity, _, _ in graph.graph[sample_index[sid]]:
            if target in reverse_category and capacity == 0:
                assignment[sid] = reverse_category[target]
                break
    if len(assignment) != target_count:
        raise AssertionError(f"assignment extraction failed: {len(assignment)}/{target_count}")
    return assignment, quotas


def interleaved_execution_order(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for row in plan:
        buckets[row["disturbance_category"]].append(row)
    for category in buckets:
        buckets[category].sort(key=lambda row: stable_key("execution", row["sample_id"]))
    ordered: list[dict[str, Any]] = []
    index = 0
    while any(index < len(buckets[category]) for category in CATEGORIES):
        for category in CATEGORIES:
            if index < len(buckets[category]):
                ordered.append(buckets[category][index])
        index += 1
    for rank, row in enumerate(ordered, 1):
        row["execution_rank"] = rank
    return ordered


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    legacy_input = getattr(args, "input", None)
    positive_manifest = getattr(args, "positive_manifest", None)
    rubric_manifest = getattr(args, "rubric_manifest", None)
    if legacy_input and (positive_manifest or rubric_manifest):
        raise ValueError("use either --input or --positive-manifest with --rubric-manifest, not both")
    if legacy_input:
        input_path = legacy_input.resolve()
        assert_label_blind_input(input_path)
        source_rows = [
            row for row in read_jsonl(input_path)
            if row.get("route") == "path1_whole_real_audio"
        ]
        input_details = {"mode": "legacy_combined", "input": str(input_path)}
        input_records = {"legacy_input": output_record(input_path)}
    elif positive_manifest and rubric_manifest:
        positive_path = positive_manifest.resolve()
        rubric_path = rubric_manifest.resolve()
        assert_label_blind_input(positive_path)
        assert_label_blind_input(rubric_path)
        source_rows = adapt_public_inputs(positive_path, rubric_path)
        input_details = {
            "mode": "public_joined",
            "positive_manifest": str(positive_path),
            "rubric_manifest": str(rubric_path),
        }
        input_records = {
            "positive_manifest": output_record(positive_path),
            "rubric_manifest": output_record(rubric_path),
        }
    else:
        raise ValueError(
            "provide --input, or provide both --positive-manifest and --rubric-manifest"
        )
    with stage_event(run, "prepare", input_details) as event:
        if args.expected_sources and len(source_rows) != args.expected_sources:
            raise ValueError(f"expected {args.expected_sources} Route 1 rows, got {len(source_rows)}")
        frozen = [freeze_source_row(row, hash_audio=not args.skip_audio_hash) for row in source_rows]
        frozen.sort(key=lambda row: row["sample_id"])
        frozen_path = run / "inputs/path1_positive_frozen.jsonl"
        atomic_write_jsonl(frozen_path, frozen)

        options: dict[str, dict[str, dict[str, Any]]] = {}
        rejected_targets: list[dict[str, Any]] = []
        unsupported_sources: list[dict[str, Any]] = []
        for source in frozen:
            source_options: dict[str, dict[str, Any]] = {}
            for question in source["questions"]:
                reason = target_rejection(question, source["prompt"])
                if reason:
                    rejected_targets.append({
                        "sample_id": source["sample_id"],
                        "rubric_id": question["rubric_id"],
                        "reason": reason,
                    })
            for category in CATEGORIES:
                choice = build_choice(source, category)
                if choice is not None:
                    source_options[category] = choice
            if source_options:
                options[source["sample_id"]] = source_options
            else:
                unsupported_sources.append({
                    "sample_id": source["sample_id"],
                    "reason": "no_label_blind_safe_edit_option",
                })
        assignment, quotas = assign_balanced(options, args.target_count)
        by_source = {row["sample_id"]: row for row in frozen}
        plan: list[dict[str, Any]] = []
        for parent_id, category in assignment.items():
            source = by_source[parent_id]
            choice = options[parent_id][category]
            qmap = {item["rubric_id"]: item for item in source["questions"]}
            direct = set(choice["direct_target_rubric_ids"])
            affected = set(choice["hierarchy_affected_rubric_ids"])
            negative_id = f"{parent_id}::neg::prompt_side::{category}::unlabeled"
            plan.append({
                "schema_version": SCHEMA_VERSION,
                "sample_id": negative_id,
                "negative_id": negative_id,
                "parent_positive_id": parent_id,
                "source_group_id": source["source_group_id"],
                "route": "path1_prompt_side_negative",
                "pair_type": "P_prime_A",
                "positive_pair": False,
                "source_assumption": "positive_by_construction",
                "label_visibility": "none",
                "label_status": "pending",
                "selection_policy": "rubric_structure_only_no_perceptual_labels",
                "disturbance_category": category,
                "intended_perturbation_strength": choice["intended_perturbation_strength"],
                "construction_changed_fraction": choice["construction_changed_fraction"],
                "language": source["language"],
                "original_prompt": source["prompt"],
                "original_prompt_sha256": source["prompt_sha256"],
                "audio_path": source["audio_path"],
                "audio_sha256": source["audio_sha256"],
                "audio_duration_seconds": source["audio_duration_seconds"],
                "direct_target_rubric_ids": sorted(direct),
                "hierarchy_affected_rubric_ids": sorted(affected),
                "retained_rubric_ids": sorted(set(qmap) - affected),
                "direct_target_rubrics": [deepcopy(qmap[rid]) for rid in sorted(direct)],
                "affected_rubrics": [deepcopy(qmap[rid]) for rid in sorted(affected)],
                "retained_rubrics": [deepcopy(qmap[rid]) for rid in sorted(set(qmap) - affected)],
                "all_original_rubrics": [deepcopy(item) for item in source["questions"]],
            })
        plan = interleaved_execution_order(plan)
        assert_no_label_keys(plan)
        plan_path = run / "plan/counterfactual_plan.jsonl"
        rejected_path = run / "plan/rejected_target_rubrics.jsonl"
        unsupported_path = run / "plan/unsupported_sources.jsonl"
        atomic_write_jsonl(plan_path, plan)
        atomic_write_jsonl(rejected_path, rejected_targets)
        atomic_write_jsonl(unsupported_path, unsupported_sources)

        config = {
            "schema_version": SCHEMA_VERSION,
            "created_at": now(),
            "input_mode": input_details["mode"],
            "inputs": input_records,
            "frozen_input": output_record(frozen_path),
            "label_visibility": "none",
            "source_assumption": "positive_by_construction",
            "expected_sources": args.expected_sources,
            "target_count_requested": args.target_count,
            "model": DEFAULT_MODEL,
            "categories": list(CATEGORIES),
            "category_strength": CATEGORY_STRENGTH,
            "target_changed_fraction": TARGET_CHANGED_FRACTION,
            "strength_range": STRENGTH_RANGE,
            "audio_hashing_skipped": args.skip_audio_hash,
        }
        atomic_write_json(run / "provenance/config.json", config)
        script_snapshot, config_snapshot = snapshot_code_and_config(run, config)
        environment = environment_snapshot()
        environment["script_snapshot"] = output_record(script_snapshot)
        environment["config_snapshot"] = output_record(config_snapshot)
        atomic_write_json(run / "provenance/environment.json", environment)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "source_rows": len(frozen),
            "sources_with_options": len(options),
            "planned_negatives": len(plan),
            "unsupported_sources": len(unsupported_sources),
            "rejected_target_rubrics": len(rejected_targets),
            "categories": dict(Counter(row["disturbance_category"] for row in plan)),
            "intended_strength": dict(Counter(row["intended_perturbation_strength"] for row in plan)),
            "languages": dict(Counter(row["language"] for row in plan)),
            "quotas": quotas,
            "outputs": [
                output_record(frozen_path), output_record(plan_path),
                output_record(rejected_path), output_record(unsupported_path),
            ],
        }
        summary_path = run / "plan/summary.json"
        atomic_write_json(summary_path, summary)
        event["summary"] = output_record(summary_path)
        return summary


def compact_rubric(question: dict[str, Any]) -> dict[str, Any]:
    return {key: question.get(key) for key in (
        "rubric_id", "axis", "scope", "domain", "subtype", "question",
        "basis", "target_ids", "parent_ids",
    )}


def canonical_event_family(value: Any) -> str | None:
    values = value if isinstance(value, list) else [value]
    families = {
        EVENT_FAMILY_ALIASES.get(str(item).strip().lower().replace("-", "_"))
        for item in values if item is not None
    }
    families.discard(None)
    return next(iter(families)) if len(families) == 1 else None


def detected_event_families(text: str) -> set[str]:
    return {family for family, pattern in EVENT_FAMILY_PATTERNS.items() if pattern.search(text)}


def declared_family_matches_text(declared: str, detected: set[str]) -> bool:
    """Return whether a controlled family is compatible with lexical evidence.

    ``sound_effect`` is the public umbrella used by decomposition manifests;
    the deterministic detector reports several of its concrete audible
    subfamilies separately.
    """
    if declared in detected:
        return True
    return declared == "sound_effect" and bool(
        detected & {"impact", "mechanical", "vehicle", "footsteps"}
    )


def new_proper_names(source: str, replacement: str) -> set[str]:
    source_names = {match.group(0).casefold() for match in PROPER_NAME_RE.finditer(source)}
    return {
        match.group(0) for match in PROPER_NAME_RE.finditer(replacement)
        if match.group(0).casefold() not in source_names
    }


def stale_event_coreferences(
    plan: dict[str, Any], patches: list[dict[str, Any]], patch: dict[str, Any],
    original_family: str,
) -> list[str]:
    """Find unchanged temporal references to an event whose family was replaced."""
    family_terms = EVENT_FAMILY_REFERENCE_TERMS.get(original_family)
    if not family_terms:
        return []
    edited_target_ids = {
        str(target_id)
        for rubric in plan["direct_target_rubrics"]
        if str(rubric.get("rubric_id")) in set(map(str, patch["rubric_ids"]))
        for target_id in rubric.get("target_ids") or []
    }
    if not edited_target_ids:
        return []
    temporal_reference = re.compile(
        rf"\b(?:after|before|when|once|while|as soon as)\s+(?:the\s+|that\s+)?"
        rf"(?:{family_terms})\b|"
        rf"\b(?:{family_terms})\s+(?:ends?|starts?|begins?|stops?),?\s+"
        rf"(?:then|before|after)\b|"
        rf"(?:在|当|等到)?(?:{family_terms})(?:结束|开始|停止)(?:后|前|时)",
        re.I,
    )
    source = plan["original_prompt"]
    stale: list[str] = []
    for rubric in plan["affected_rubrics"]:
        if rubric_kind(rubric) != "relation":
            continue
        if not (edited_target_ids & set(map(str, rubric.get("target_ids") or []))):
            continue
        basis = str(rubric.get("basis") or "")
        if not basis:
            continue
        search_from = 0
        while True:
            basis_start = source.find(basis, search_from)
            if basis_start < 0:
                break
            for match in temporal_reference.finditer(basis):
                start = basis_start + match.start()
                end = basis_start + match.end()
                if not any(
                    start < candidate["end_char"] and end > candidate["start_char"]
                    for candidate in patches
                ):
                    stale.append(source[start:end])
            search_from = basis_start + 1
    return sorted(set(stale))


def target_specific_instructions(plan: dict[str, Any]) -> str:
    direct_event_target_ids = {
        str(target_id)
        for direct_target in plan["direct_target_rubrics"]
        if rubric_kind(direct_target) == "event"
        for target_id in direct_target.get("target_ids") or []
    }
    target_instructions: list[str] = []
    for target in plan["direct_target_rubrics"]:
        rubric_id = str(target.get("rubric_id"))
        kind = rubric_kind(target)
        family = canonical_event_family(target.get("domain"))
        subtype = str(target.get("subtype") or "unspecified")
        if kind == "event":
            detail = (
                f"- {rubric_id} is an EVENT target. Its original coarse family is "
                f"{family or 'given by the target text'}. Replace it with a different coarse "
                "audible family, not a different speaker gender, person, topic, wording, or "
                "other attribute of the same event."
            )
            if family == "speech":
                detail += (
                    " The replacement must be non-speech and must not describe a narrator, "
                    "speaker, voice, person, utterance, or speech topic. Replace the complete "
                    "speech-carried proposition; do not retain its explanation, measurements, "
                    "quotations, or topic clauses. Use one concise auditory description with no "
                    "unseen physical motion, mechanism, or energy state."
                )
            target_ids = set(map(str, target.get("target_ids") or []))
            direct_relation_ids = {
                str(item.get("rubric_id"))
                for item in plan["direct_target_rubrics"]
                if rubric_kind(item) == "relation"
            }
            preserved_relation_subtypes = sorted({
                str(item.get("subtype"))
                for item in plan.get("affected_rubrics") or []
                if rubric_kind(item) == "relation"
                and str(item.get("rubric_id")) not in direct_relation_ids
                and target_ids & set(map(str, item.get("target_ids") or []))
            })
            if preserved_relation_subtypes:
                detail += (
                    " Update adjacent references to the replacement event but preserve these "
                    "untargeted relation subtype(s) exactly: "
                    f"{preserved_relation_subtypes}."
                )
            target_instructions.append(detail)
        elif kind == "attribute":
            target_instructions.append(
                f"- {rubric_id} is an ATTRIBUTE target ({subtype}). Preserve the same coarse "
                f"audible family ({family or 'the family in the target text'}) and flip only "
                "that requested mutually exclusive audible property."
            )
        elif kind == "relation":
            relation_target_ids = set(map(str, target.get("target_ids") or []))
            changed_endpoint_ids = relation_target_ids & direct_event_target_ids
            unchanged_endpoint_ids = relation_target_ids - changed_endpoint_ids
            unchanged_endpoint_anchors: list[str] = []
            for endpoint_id in sorted(unchanged_endpoint_ids):
                candidates = [
                    str(rubric.get("basis") or "")
                    for rubric in plan.get("all_original_rubrics") or []
                    if list(map(str, rubric.get("target_ids") or [])) == [endpoint_id]
                    and str(rubric.get("basis") or "") in plan["original_prompt"]
                ]
                if candidates:
                    unchanged_endpoint_anchors.append(min(candidates, key=len))
            if changed_endpoint_ids:
                target_instructions.append(
                    f"- {rubric_id} is a RELATION target ({subtype}). Flip only the requested "
                    "relation subtype. Preserve every endpoint not changed by an EVENT target "
                    "verbatim. For EVENT-modified endpoint ID(s) "
                    f"{sorted(changed_endpoint_ids)}, use the replacement event anchor from "
                    "P PRIME in relation_endpoints. "
                    f"UNCHANGED ENDPOINT ANCHORS (copy exactly): {unchanged_endpoint_anchors}. "
                    "Prefer editing only the relation connective so these anchors remain "
                    "byte-identical."
                )
            else:
                target_instructions.append(
                    f"- {rubric_id} is a RELATION target ({subtype}). Preserve both endpoint "
                    "event identities verbatim and flip only the requested relation subtype."
                )
            if subtype in {"before", "after"}:
                target_instructions.append(
                    f"  For {rubric_id}, express the flip as unambiguous overlap using while, "
                    "at the same time, simultaneously, or an equivalent phrase. Do not use a "
                    "bare before/after/prior/subsequent connective whose endpoint orientation "
                    "can remain unchanged."
                )
    return "\n".join(target_instructions)


def patch_request(plan: dict[str, Any], previous_error: str | None) -> str:
    retry = ""
    if previous_error:
        guidance = "Repair only that defect."
        if "music_genre_near_neighbor" in previous_error:
            if plan["disturbance_category"] == "attribute":
                subtypes = sorted({str(row.get("subtype")) for row in plan["direct_target_rubrics"]})
                guidance = (
                    f"Do not change music genre or event identity. Change the requested attribute "
                    f"subtype {subtypes}; for placement, flip background and foreground."
                )
            else:
                guidance = (
                    "For an event target, replace the music or tonal effect with a different "
                    "non-music coarse audible family."
                )
        elif "edit_kind_target_mismatch" in previous_error:
            expected = sorted({
                str(rubric_kind(row)) for row in plan["direct_target_rubrics"]
                if rubric_kind(row) is not None
            })
            guidance = (
                f"Use edit_kind matching the direct target kind(s) {expected}. "
                "An event target changes coarse audible family; it is not a speaker-gender, topic, "
                "or other attribute edit. Prefer one non-overlapping patch per direct target kind. "
                "Do not attach a relation ID to an event patch when its connective can be edited "
                "separately. Only when one inseparable source span necessarily carries multiple "
                "direct target kinds, attach those IDs once and set edit_kind to mixed."
            )
        elif "original_family_text_mismatch" in previous_error:
            guidance = (
                "Set original_event_family to the coarse family directly supported by the "
                "original target text. sound_effect may be used as an umbrella for impact, "
                "mechanical, vehicle, or footsteps sounds."
            )
        elif "replacement_family_text_mismatch" in previous_error:
            guidance = (
                "Make replacement_event_family agree with the replacement text, or rewrite the "
                "replacement as the intended coarse family."
            )
        elif "stale_event_coreference" in previous_error:
            guidance = (
                "Expand the same exact contiguous event patch through each adjacent temporal "
                "reference to the replaced event, and rewrite that reference to the new event "
                "family. Do not leave phrases such as 'after the speech ends' after replacing "
                "speech with a non-speech event."
            )
        elif "relation_endpoint" in previous_error:
            guidance = (
                "Copy each unaffected relation endpoint character-for-character from FROZEN P "
                "and keep it unchanged in P PRIME. For an endpoint directly changed by an EVENT "
                "target, use an exact replacement-event substring that will exist in P PRIME."
            )
        elif "relation_patch_subtype" in previous_error:
            guidance = (
                "Use an explicit relation phrase whose meaning is a permitted flip of the source "
                "subtype. For before, use an unambiguous after or overlap construction; do not "
                "use prior to, subsequently, or another before construction."
            )
        elif "incidental_relation_change" in previous_error:
            guidance = (
                "When updating a temporal reference to an EVENT replacement, preserve the "
                "original relation word and subtype unless that relation rubric is itself a "
                "DIRECT TARGET. For example, rewrite 'after the speech ends' as 'after the "
                "mechanical sound ends', not 'before the mechanical sound ends'."
            )
        elif "visual_leakage" in previous_error:
            guidance = (
                "Remove visually inferred object motion or state. Describe only the sound itself "
                "and its audible qualities."
            )
        elif "stale_speech_content" in previous_error:
            guidance = (
                "When replacing speech with a non-speech event, remove the complete explanation, "
                "topic, quotation, measurements, and other propositions that were carried by the "
                "speech. Write one self-contained, purely audible non-speech event instead."
            )
        retry = f"\nPREVIOUS VALIDATOR ERROR: {previous_error}\n{guidance}"

    target_instruction_text = target_specific_instructions(plan)
    return f"""Create one controlled audio-description negative using exact-span patches only.

The source is assumed positive by construction. No perceptual labels are available or permitted.
The audio is frozen and must not be edited. Return JSON only.

Hard rules:
1. Each original_span is one unique, contiguous, character-for-character substring of FROZEN P.
2. Patches do not overlap. Never use ellipses to join separated spans.
3. The union of rubric_ids equals DIRECT TARGET IDs exactly, with each ID appearing once.
4. Text outside patches remains byte-identical and the original language is preserved.
5. Add no visual facts, timestamps, causality, or unverifiable context.
6. Do not create a negative merely by adding no/not/without/没有/未出现.
7. Event replacements use a clearly different coarse audible identity.
8. Attribute replacements keep event identity but use a mutually exclusive audible property.
9. Relation replacements keep each endpoint identity unless that same endpoint is also a direct
   event target, and flip before/after, overlap/separation, or foreground/background.
10. Avoid near-neighbour changes such as genre-to-genre, accent-to-accent, nearby counts,
    or small intensity changes.
11. Every patch includes edit_kind, semantic_distance=high, contradiction_explanation,
    and an audio-only audible_test. The explanation names every attached rubric ID.
12. Do not invent a person's name, organization, location, quoted utterance, or narrative context.
13. For event and attribute edits, report original_event_family and replacement_event_family.
    Event edits must change that coarse audible family. Attribute edits must preserve it.
14. For relation edits, report two relation_endpoints found exactly in P PRIME. An endpoint not
    directly changed by an event target must also remain an exact substring of P. If an endpoint
    is directly changed by an event target, report its replacement anchor from P PRIME instead.
15. Each replacement must remain grammatical with the byte-identical text immediately before
    and after it; include an adjacent article in original_span when the replacement requires it.
16. If an event-family change makes an adjacent temporal reference false, include that reference
    in the same contiguous patch and update it to the replacement event (for example, do not leave
    "after the speech ends" after replacing speech with a non-speech event). Preserve that
    reference's relation subtype unless the relation rubric is also a direct target.

LANGUAGE: {plan['language']}
CATEGORY: {plan['disturbance_category']}
INTENDED STRENGTH: {plan['intended_perturbation_strength']}
DIRECT TARGET IDS: {json.dumps(plan['direct_target_rubric_ids'], ensure_ascii=False)}
DIRECT TARGET RUBRICS:
{json.dumps([compact_rubric(row) for row in plan['direct_target_rubrics']], ensure_ascii=False, indent=2)}

TARGET-SPECIFIC INSTRUCTIONS:
{target_instruction_text}

HIERARCHY-AFFECTED RUBRICS:
{json.dumps([compact_rubric(row) for row in plan['affected_rubrics']], ensure_ascii=False, indent=2)}

RUBRICS CURRENTLY EXPECTED TO BE RETAINED:
{json.dumps([compact_rubric(row) for row in plan['retained_rubrics']], ensure_ascii=False, indent=2)}

Keep these rubrics unchanged when their basis lies outside every patch. If a
direct target necessarily shares text with one of these bases, keep the patch
as short as possible; the pipeline will promote the overlapping rubric to the
affected rebuild set deterministically.

FROZEN P:
{plan['original_prompt']}
{retry}

Return exactly:
{{"patches":[{{"rubric_ids":["R..."],"original_span":"exact source substring",
"replacement_span":"natural counterfactual text","edit_kind":"event|attribute|relation|mixed",
"original_event_family":"speech|music|sound_effect|vocalization|animal|ambience|mechanical|vehicle|impact|footsteps|other",
"replacement_event_family":"same controlled vocabulary value",
"relation_endpoints":["unchanged event anchor 1","unchanged event anchor 2"],
"semantic_distance":"high","contradiction_explanation":"...","audible_test":"..."}}],
"note":"short audit note"}}"""


def extract_json(raw: str, required_key: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        # Reasoning-capable servers may emit JSON-looking arrays or example
        # objects before the final answer. Only accept an object with the
        # schema's required top-level key, and prefer the last such object.
        if isinstance(value, dict) and required_key in value:
            candidates.append(value)
    if not candidates:
        raise ValueError(f"response contains no JSON object with key {required_key!r}")
    return candidates[-1]


def near_neighbor_reason(original: str, replacement: str) -> str | None:
    if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", original) and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", replacement):
        return "numeric_only_substitution"
    original_genres = {value.casefold() for value in MUSIC_GENRE_RE.findall(original)}
    replacement_genres = {value.casefold() for value in MUSIC_GENRE_RE.findall(replacement)}
    if original_genres and replacement_genres and original_genres != replacement_genres:
        return "music_genre_near_neighbor"
    original_accents = {value.casefold() for value in ACCENT_RE.findall(original)}
    replacement_accents = {value.casefold() for value in ACCENT_RE.findall(replacement)}
    if original_accents and replacement_accents and original_accents != replacement_accents:
        return "accent_near_neighbor"
    return None


def apply_exact_patches(
    source: str, patches: list[dict[str, Any]], expected_ids: set[str],
) -> tuple[str, list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: list[str] = []
    for index, patch in enumerate(patches):
        original = str(patch.get("original_span") or "")
        replacement = str(patch.get("replacement_span") or "")
        if not original or not replacement or original == replacement:
            raise ValueError(f"patch_{index}_empty_or_unchanged")
        if ("..." in original or "…" in original) and source.count(original) == 0:
            raise ValueError(
                f"patch_{index}_noncontiguous_ellipsis; copy one literal contiguous source span"
            )
        occurrence_count = source.count(original)
        if occurrence_count != 1:
            raise ValueError(
                f"patch_{index}_original_span_not_unique:count={occurrence_count}; "
                "extend the exact source quote with adjacent context until count=1"
            )
        raw_rubric_ids = [str(value) for value in patch.get("rubric_ids") or []]
        if not raw_rubric_ids or len(raw_rubric_ids) != len(set(raw_rubric_ids)):
            raise ValueError(f"patch_{index}_invalid_rubric_ids")
        # The model sometimes annotates a direct edit with descendant rubrics
        # from the separately supplied affected set. Those IDs are provenance
        # metadata, not extra edit targets, so remove them deterministically.
        rubric_ids = [value for value in raw_rubric_ids if value in expected_ids]
        if not rubric_ids:
            raise ValueError(f"patch_{index}_contains_no_direct_target_rubric")
        seen_ids.extend(rubric_ids)
        start = source.index(original)
        normalized.append({
            **patch,
            "rubric_ids": rubric_ids,
            "model_rubric_ids": raw_rubric_ids,
            "start_char": start,
            "end_char": start + len(original),
        })
    if len(seen_ids) != len(set(seen_ids)) or set(seen_ids) != expected_ids:
        raise ValueError(f"patch_rubric_union_mismatch:{sorted(set(seen_ids))}!={sorted(expected_ids)}")
    normalized.sort(key=lambda row: row["start_char"])
    for pair_index, (left, right) in enumerate(zip(normalized, normalized[1:])):
        if left["end_char"] > right["start_char"]:
            raise ValueError(
                f"overlapping_patches:{pair_index},{pair_index + 1}; merge them into one exact patch "
                "or shorten both source spans"
            )
    chunks: list[str] = []
    cursor = 0
    delta = 0
    for patch in normalized:
        chunks.append(source[cursor:patch["start_char"]])
        chunks.append(str(patch["replacement_span"]))
        patch["new_start_char"] = patch["start_char"] + delta
        patch["new_end_char"] = patch["new_start_char"] + len(str(patch["replacement_span"]))
        delta += len(str(patch["replacement_span"])) - (patch["end_char"] - patch["start_char"])
        cursor = patch["end_char"]
    chunks.append(source[cursor:])
    return "".join(chunks), normalized


def rubric_basis_overlaps_patches(
    rubric: dict[str, Any], source: str, patches: list[dict[str, Any]],
) -> bool:
    basis = str(rubric.get("basis") or "")
    if not basis:
        return False
    start = 0
    while True:
        found = source.find(basis, start)
        if found < 0:
            return False
        end = found + len(basis)
        if any(found < patch["end_char"] and end > patch["start_char"] for patch in patches):
            return True
        start = found + 1


def validate_patch_result(plan: dict[str, Any], prompt_prime: str, patches: list[dict[str, Any]]) -> None:
    if prompt_prime == plan["original_prompt"]:
        raise ValueError("prompt_prime_equals_source")
    allowed_kinds = {"event", "attribute", "relation", "mixed"}
    for index, patch in enumerate(patches):
        kind = str(patch.get("edit_kind") or "").lower()
        if kind not in allowed_kinds:
            raise ValueError(f"patch_{index}_invalid_edit_kind")
        if str(patch.get("semantic_distance") or "").lower() != "high":
            raise ValueError(f"patch_{index}_not_high_distance")
        replacement = str(patch["replacement_span"])
        original = str(patch["original_span"])
        if plan.get("language") == "en":
            prefix = prompt_prime[:int(patch["new_start_char"])]
            article_match = re.search(r"\b(a|an)\s*$", prefix, re.I)
            first_word_match = re.search(r"[A-Za-z]+", replacement)
            if article_match and first_word_match:
                article = article_match.group(1).lower()
                first_word = first_word_match.group(0).lower()
                vowel_sound = bool(re.match(r"[aeiou]", first_word)) and not re.match(
                    r"(?:uni(?:t|v|f)|use|user|usual|euro|one)", first_word,
                )
                vowel_sound = vowel_sound or bool(re.match(r"(?:honest|hour|heir)", first_word))
                if (article == "a" and vowel_sound) or (article == "an" and not vowel_sound):
                    raise ValueError(
                        f"patch_{index}_article_boundary_mismatch:{article} {first_word}; "
                        "include the article in original_span and repair the full phrase"
                    )
        audible_test = str(patch.get("audible_test") or "")
        explanation = str(patch.get("contradiction_explanation") or "")
        if not audible_test or not explanation:
            raise ValueError(f"patch_{index}_missing_audit_text")
        visual_match = VISUAL_RE.search(replacement) or VISUAL_RE.search(audible_test)
        if visual_match:
            raise ValueError(
                f"patch_{index}_visual_leakage:{visual_match.group(0)!r}; use only audible wording"
            )
        absence_match = EXPLICIT_ABSENCE_RE.search(replacement)
        if kind in {"event", "attribute", "mixed"} and absence_match:
            raise ValueError(
                f"patch_{index}_explicit_absence_shortcut:{absence_match.group(0)!r}; "
                "replace with a different audible event/property instead"
            )
        reason = near_neighbor_reason(original, replacement)
        declared_original_for_neighbor = canonical_event_family(
            patch.get("original_event_family")
        )
        declared_replacement_for_neighbor = canonical_event_family(
            patch.get("replacement_event_family")
        )
        if reason == "music_genre_near_neighbor" and {
            declared_original_for_neighbor, declared_replacement_for_neighbor,
        } != {"music"}:
            reason = None
        if reason == "accent_near_neighbor" and {
            declared_original_for_neighbor, declared_replacement_for_neighbor,
        } != {"speech"}:
            reason = None
        if reason:
            raise ValueError(
                f"patch_{index}_{reason}; do not replace one music genre/style/accent with another; "
                "choose a non-music coarse sound event or a non-genre audible property"
            )
        for rubric_id in patch["rubric_ids"]:
            if rubric_id not in explanation:
                raise ValueError(f"patch_{index}_explanation_missing_{rubric_id}")
        target_by_id = {
            str(row["rubric_id"]): row for row in plan["direct_target_rubrics"]
        }
        target_kinds = {rubric_kind(target_by_id[rubric_id]) for rubric_id in patch["rubric_ids"]}
        target_kinds.discard(None)
        if kind != "mixed" and target_kinds != {kind}:
            raise ValueError(
                f"patch_{index}_edit_kind_target_mismatch:{kind}!={sorted(target_kinds)}"
            )
        if kind == "mixed" and len(target_kinds) < 2:
            raise ValueError(f"patch_{index}_mixed_without_multiple_target_kinds")

        invented_names = new_proper_names(plan["original_prompt"], replacement)
        if invented_names:
            raise ValueError(f"patch_{index}_invented_proper_names:{sorted(invented_names)}")

        if target_kinds & {"event", "attribute"}:
            declared_original = canonical_event_family(patch.get("original_event_family"))
            declared_replacement = canonical_event_family(patch.get("replacement_event_family"))
            if not declared_original or not declared_replacement:
                raise ValueError(f"patch_{index}_missing_or_invalid_event_family")
            target_families = {
                canonical_event_family(target_by_id[rubric_id].get("domain"))
                for rubric_id in patch["rubric_ids"]
                if rubric_kind(target_by_id[rubric_id]) in {"event", "attribute"}
            }
            target_families.discard(None)
            if (
                len(target_families) == 1
                and "sound_effect" not in target_families
                and declared_original not in target_families
            ):
                raise ValueError(
                    f"patch_{index}_original_family_mismatch:{declared_original}!={sorted(target_families)}"
                )
            original_evidence = " ".join(
                [original] + [
                    str(target_by_id[rubric_id].get("basis") or "")
                    for rubric_id in patch["rubric_ids"]
                ]
            )
            detected_original = detected_event_families(original_evidence)
            if detected_original and not declared_family_matches_text(
                declared_original, detected_original,
            ):
                raise ValueError(
                    f"patch_{index}_original_family_text_mismatch:"
                    f"{declared_original} not in {sorted(detected_original)}"
                )
            detected_replacement = detected_event_families(replacement)
            if detected_replacement and not declared_family_matches_text(
                declared_replacement, detected_replacement,
            ):
                raise ValueError(
                    f"patch_{index}_replacement_family_text_mismatch:"
                    f"{declared_replacement} not in {sorted(detected_replacement)}"
                )
            if "attribute" in target_kinds and "event" not in target_kinds and declared_original != declared_replacement:
                raise ValueError(
                    f"patch_{index}_attribute_changes_event_identity:"
                    f"{declared_original}->{declared_replacement}"
                )
            if (
                "attribute" in target_kinds
                and "event" not in target_kinds
                and detected_original
                and detected_replacement
                and not (
                    detected_original & detected_replacement
                    or declared_family_matches_text(declared_original, detected_replacement)
                    or declared_family_matches_text(declared_replacement, detected_original)
                )
            ):
                raise ValueError(
                    f"patch_{index}_attribute_changes_detected_event_identity:"
                    f"{sorted(detected_original)}->{sorted(detected_replacement)}"
                )
            if "event" in target_kinds and declared_original == declared_replacement:
                raise ValueError(
                    f"patch_{index}_event_keeps_same_coarse_identity:{declared_original}"
                )
            if "event" in target_kinds:
                if (
                    declared_original == "speech"
                    and declared_replacement != "speech"
                    and STALE_SPEECH_CONTENT_RE.search(replacement)
                ):
                    raise ValueError(
                        f"patch_{index}_stale_speech_content; remove explanation, topic, "
                        "quotation, and other speech-carried propositions from the non-speech "
                        "replacement"
                    )
                stale_references = stale_event_coreferences(
                    plan, patches, patch, declared_original,
                )
                if stale_references:
                    raise ValueError(
                        f"patch_{index}_stale_event_coreference:{stale_references}; "
                        "include and update the dependent temporal reference in the same patch"
                    )
                direct_relation_ids = {
                    str(target.get("rubric_id"))
                    for target in plan["direct_target_rubrics"]
                    if rubric_kind(target) == "relation"
                }
                original_patch_relation = relation_subtype_from_question(original)
                replacement_patch_relation = relation_subtype_from_question(replacement)
                if original_patch_relation:
                    related_untargeted = [
                        rubric for rubric in plan["affected_rubrics"]
                        if rubric_kind(rubric) == "relation"
                        and str(rubric.get("rubric_id")) not in direct_relation_ids
                        and str(rubric.get("subtype") or "") == original_patch_relation
                        and rubric_basis_overlaps_patches(
                            rubric, plan["original_prompt"], [patch],
                        )
                    ]
                    if related_untargeted and replacement_patch_relation != original_patch_relation:
                        raise ValueError(
                            f"patch_{index}_incidental_relation_change:"
                            f"{original_patch_relation}->{replacement_patch_relation}; "
                            "preserve the relation when only the event is targeted"
                        )

        if "relation" in target_kinds:
            endpoints = [str(value).strip() for value in patch.get("relation_endpoints") or []]
            if len(endpoints) != 2 or len(set(endpoints)) != 2 or any(not value for value in endpoints):
                raise ValueError(f"patch_{index}_invalid_relation_endpoints")
            event_target_ids = {
                str(target_id)
                for target in plan["direct_target_rubrics"]
                if rubric_kind(target) == "event"
                for target_id in target.get("target_ids") or []
            }
            relation_target_ids = {
                str(target_id)
                for rubric_id in patch["rubric_ids"]
                for target_id in target_by_id[rubric_id].get("target_ids") or []
            }
            changed_endpoint_capacity = len(event_target_ids & relation_target_ids)
            changed_endpoints = 0
            for endpoint in endpoints:
                if endpoint not in prompt_prime:
                    raise ValueError(
                        f"patch_{index}_relation_endpoint_missing_from_prompt_prime:{endpoint!r}"
                    )
                if endpoint not in plan["original_prompt"]:
                    changed_endpoints += 1
            if changed_endpoints > changed_endpoint_capacity:
                raise ValueError(
                    f"patch_{index}_too_many_changed_relation_endpoints:"
                    f"{changed_endpoints}>{changed_endpoint_capacity}"
                )
            source_relation_subtypes = {
                str(target_by_id[rubric_id].get("subtype") or "")
                for rubric_id in patch["rubric_ids"]
                if rubric_kind(target_by_id[rubric_id]) == "relation"
            }
            if len(source_relation_subtypes) == 1:
                source_subtype = next(iter(source_relation_subtypes))
                replacement_subtype = relation_subtype_from_question(replacement)
                if replacement_subtype not in PATCH_RELATION_FLIPS.get(source_subtype, set()):
                    raise ValueError(
                        f"patch_{index}_relation_patch_subtype_not_flipped:"
                        f"{source_subtype}->{replacement_subtype}"
                    )
    changed_source_chars = sum(patch["end_char"] - patch["start_char"] for patch in patches)
    maximum = 0.45 if plan["intended_perturbation_strength"] == "medium" else 0.85
    changed_fraction = changed_source_chars / max(1, len(plan["original_prompt"]))
    if changed_fraction > maximum:
        raise ValueError(
            f"patched_source_fraction_too_large:{changed_fraction:.4f}>{maximum:.4f}; "
            "use shorter exact spans"
        )
    for rubric in plan["retained_rubrics"]:
        basis = str(rubric.get("basis") or "")
        if basis and basis in plan["original_prompt"] and basis not in prompt_prime:
            if not rubric_basis_overlaps_patches(rubric, plan["original_prompt"], patches):
                raise ValueError(f"retained_basis_lost_outside_patches:{rubric['rubric_id']}")


async def call_model_json(
    client: Any, model: str, prompt: str, max_tokens: int, required_key: str,
) -> tuple[dict[str, Any], str]:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = response.choices[0].message.content or ""
    return extract_json(raw, required_key), raw


def require_model_execution(args: argparse.Namespace) -> None:
    if args.execute_token != EXECUTION_TOKEN:
        raise ValueError(f"model execution requires --execute-token {EXECUTION_TOKEN}")
    if not args.endpoint:
        raise ValueError("model execution requires at least one explicit --endpoint")
    for endpoint in args.endpoint:
        parsed = urlparse(endpoint)
        host = (parsed.hostname or "").lower()
        local_host = (
            host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
            or host.startswith("10.")
            or host.startswith("192.168.")
            or bool(re.match(r"^172\.(?:1[6-9]|2\d|3[01])\.", host))
            or host.startswith("dlc-")
        )
        if parsed.scheme not in {"http", "https"} or not local_host:
            raise ValueError(f"only explicit local HTTP(S) endpoints are allowed: {endpoint}")


def latest_successes(path: Path, plan_by_id: dict[str, dict[str, Any]], validator: Any) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    successes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if row.get("status") != "ok" or sample_id not in plan_by_id:
            continue
        try:
            validator(plan_by_id[sample_id], row)
        except Exception:
            continue
        successes[sample_id] = row
    return successes


def latest_failure_errors(
    path: Path, plan_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Recover the most recent validator diagnosis for resumable generation."""
    if not path.exists():
        return {}
    failures: dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if row.get("status") == "error" and sample_id in plan_by_id and row.get("error"):
            failures[sample_id] = str(row["error"])
    return failures


def validate_stored_patch(plan: dict[str, Any], row: dict[str, Any]) -> None:
    rebuilt, patches = apply_exact_patches(
        plan["original_prompt"], row["patches"], set(plan["direct_target_rubric_ids"]),
    )
    if rebuilt != row.get("prompt_prime"):
        raise ValueError("stored_prompt_prime_mismatch")
    validate_patch_result(plan, rebuilt, patches)


async def generate_async(args: argparse.Namespace) -> dict[str, Any]:
    require_model_execution(args)
    from openai import AsyncOpenAI

    run = args.run.resolve()
    plan_rows = sorted(read_jsonl(run / "plan/counterfactual_plan.jsonl"), key=lambda row: row["execution_rank"])
    if args.limit:
        plan_rows = plan_rows[:args.limit]
    plan_by_id = {row["sample_id"]: row for row in plan_rows}
    output = run / "model/patch_attempts.jsonl"
    completed = latest_successes(output, plan_by_id, validate_stored_patch)
    pending = [row for row in plan_rows if row["sample_id"] not in completed]
    resumed_errors = latest_failure_errors(output, plan_by_id)
    endpoints = [value.rstrip("/") for value in args.endpoint]
    clients = [AsyncOpenAI(base_url=value, api_key="EMPTY", timeout=args.timeout) for value in endpoints]
    semaphores = [asyncio.Semaphore(args.concurrency_per_endpoint) for _ in endpoints]

    async def one(
        plan: dict[str, Any], endpoint_index: int, initial_error: str | None,
    ) -> dict[str, Any]:
        previous_error = initial_error
        for attempt in range(1, args.retries + 1):
            prompt = patch_request(plan, previous_error)
            started = now()
            raw = ""
            try:
                async with semaphores[endpoint_index]:
                    parsed, raw = await call_model_json(
                        clients[endpoint_index], args.model, prompt, args.max_tokens, "patches",
                    )
                patches = list(parsed.get("patches") or [])
                prompt_prime, normalized = apply_exact_patches(
                    plan["original_prompt"], patches, set(plan["direct_target_rubric_ids"]),
                )
                validate_patch_result(plan, prompt_prime, normalized)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "timestamp": now(), "started_at": started,
                    "stage": "generate", "sample_id": plan["sample_id"],
                    "parent_positive_id": plan["parent_positive_id"],
                    "status": "ok", "attempt": attempt,
                    "endpoint": endpoints[endpoint_index], "model": args.model,
                    "request_sha256": text_sha256(prompt),
                    "raw_model_output": raw,
                    "prompt_prime": prompt_prime,
                    "prompt_prime_sha256": text_sha256(prompt_prime),
                    "patches": normalized,
                    "outside_patch_exact": True,
                    "label_visibility": "none", "label_status": "pending",
                }
                append_jsonl(output, row)
                return row
            except Exception as exc:
                previous_error = repr(exc)
                append_jsonl(output, {
                    "schema_version": SCHEMA_VERSION,
                    "timestamp": now(), "started_at": started,
                    "stage": "generate", "sample_id": plan["sample_id"],
                    "parent_positive_id": plan["parent_positive_id"],
                    "status": "error", "attempt": attempt,
                    "endpoint": endpoints[endpoint_index], "model": args.model,
                    "request_sha256": text_sha256(prompt),
                    "raw_model_output": raw,
                    "error": previous_error,
                    "label_visibility": "none",
                })
        return {"sample_id": plan["sample_id"], "status": "error", "error": previous_error}

    tasks = [
        asyncio.create_task(one(
            plan, index % len(endpoints), resumed_errors.get(str(plan["sample_id"])),
        ))
        for index, plan in enumerate(pending)
    ]
    results = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)
    summary = {
        "scheduled": len(pending),
        "ok": sum(row.get("status") == "ok" for row in results),
        "error": sum(row.get("status") != "ok" for row in results),
        "already_completed": len(completed),
        "model": args.model,
        "endpoints": endpoints,
        "attempt_log": output_record(output) if output.exists() else None,
    }
    atomic_write_json(run / "model/generate_summary.json", summary)
    return summary


def generate(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    with stage_event(run, "generate", {"model": args.model, "endpoints": args.endpoint}) as event:
        summary = asyncio.run(generate_async(args))
        event["summary"] = summary
        return summary


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    with stage_event(run, "materialize") as event:
        plan_rows = read_jsonl(run / "plan/counterfactual_plan.jsonl")
        if getattr(args, "limit", 0):
            plan_rows = sorted(plan_rows, key=lambda row: row["execution_rank"])[:args.limit]
        plan_by_id = {row["sample_id"]: row for row in plan_rows}
        successes = latest_successes(run / "model/patch_attempts.jsonl", plan_by_id, validate_stored_patch)
        if not args.allow_partial and len(successes) != len(plan_rows):
            raise ValueError(f"generation incomplete: {len(successes)}/{len(plan_rows)}")
        pairs: list[dict[str, Any]] = []
        manifests: list[dict[str, Any]] = []
        for sample_id in sorted(successes, key=lambda sid: plan_by_id[sid]["execution_rank"]):
            plan = plan_by_id[sample_id]
            result = successes[sample_id]
            all_by_id = {row["rubric_id"]: row for row in plan["all_original_rubrics"]}
            affected_ids = set(plan["hierarchy_affected_rubric_ids"])
            promoted_ids = {
                row["rubric_id"] for row in plan["retained_rubrics"]
                if rubric_basis_overlaps_patches(row, plan["original_prompt"], result["patches"])
            }
            affected_ids.update(promoted_ids)
            retained_ids = set(all_by_id) - affected_ids
            affected_rubrics = [deepcopy(all_by_id[rid]) for rid in sorted(affected_ids)]
            retained_rubrics = [deepcopy(all_by_id[rid]) for rid in sorted(retained_ids)]
            audio_path = Path(plan["audio_path"])
            actual_audio_sha = file_sha256(audio_path) if plan["audio_sha256"] else None
            if plan["audio_sha256"] and actual_audio_sha != plan["audio_sha256"]:
                raise ValueError(f"audio changed: {sample_id}")
            pair = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "negative_id": sample_id,
                "parent_positive_id": plan["parent_positive_id"],
                "source_group_id": plan["source_group_id"],
                "route": "path1_prompt_side_negative",
                "pair_type": "P_prime_A",
                "positive_pair": False,
                "source_assumption": "positive_by_construction",
                "label_visibility": "none",
                "label_status": "pending",
                "audio_path": plan["audio_path"],
                "audio_sha256": actual_audio_sha,
                "audio_byte_identical_to_parent": True,
                "audio_duration_seconds": plan["audio_duration_seconds"],
                "positive_prompt": plan["original_prompt"],
                "positive_prompt_sha256": plan["original_prompt_sha256"],
                "evaluation_prompt": result["prompt_prime"],
                "evaluation_prompt_sha256": result["prompt_prime_sha256"],
                "disturbance_category": plan["disturbance_category"],
                "intended_perturbation_strength": plan["intended_perturbation_strength"],
                "construction_changed_fraction": plan["construction_changed_fraction"],
                "direct_target_original_rubric_ids": plan["direct_target_rubric_ids"],
                "hierarchy_affected_original_rubric_ids": plan["hierarchy_affected_rubric_ids"],
                "patch_overlap_promoted_original_rubric_ids": sorted(promoted_ids),
                "affected_original_rubric_ids": sorted(affected_ids),
                "retained_original_rubric_ids": sorted(retained_ids),
                "actual_affected_fraction": round(len(affected_ids) / len(all_by_id), 6),
                "patches": result["patches"],
                "outside_patch_exact": True,
                "generator": {
                    "backend": result.get("backend", "openai_compatible_http"),
                    "model": result["model"],
                    "endpoint": result.get("endpoint"),
                    "model_path": result.get("model_path"),
                    "shard_id": result.get("shard_id"),
                    "num_shards": result.get("num_shards"),
                    "attempt": result["attempt"],
                    "request_sha256": result["request_sha256"],
                },
            }
            assert_no_label_keys(pair)
            pairs.append(pair)
            manifests.append({
                "sample_id": sample_id,
                "parent_positive_id": plan["parent_positive_id"],
                "language": plan["language"],
                "disturbance_category": plan["disturbance_category"],
                "direct_target_rubric_ids": plan["direct_target_rubric_ids"],
                "original_prompt": plan["original_prompt"],
                "prompt_prime": result["prompt_prime"],
                "patches": result["patches"],
                "affected_rubrics": affected_rubrics,
                "retained_rubrics": retained_rubrics,
                "all_original_rubrics": plan["all_original_rubrics"],
                "label_visibility": "none",
            })
        pairs_path = run / "pairs/unlabeled_pairs.jsonl"
        manifest_path = run / "rubrics/rebuild_manifest.jsonl"
        atomic_write_jsonl(pairs_path, pairs)
        atomic_write_jsonl(manifest_path, manifests)
        summary = {
            "planned": len(plan_rows), "materialized": len(pairs),
            "missing": len(plan_rows) - len(pairs),
            "categories": dict(Counter(row["disturbance_category"] for row in pairs)),
            "patch_overlap_promotions": sum(
                len(row["patch_overlap_promoted_original_rubric_ids"]) for row in pairs
            ),
            "outputs": [output_record(pairs_path), output_record(manifest_path)],
        }
        atomic_write_json(run / "pairs/materialize_summary.json", summary)
        event["summary"] = summary
        return summary


def translate_retained_rubric(
    rubric: dict[str, Any], original_prompt: str, prompt_prime: str, patches: list[dict[str, Any]],
) -> dict[str, Any]:
    basis = str(rubric.get("basis") or "")
    if not basis:
        return {**rubric, "rebuild_policy": "retained_empty_basis_unchanged"}
    old_start = rubric.get("basis_start")
    old_end = rubric.get("basis_end")
    if not isinstance(old_start, int) or not isinstance(old_end, int) or original_prompt[old_start:old_end] != basis:
        old_start = original_prompt.find(basis)
        if old_start < 0:
            return {
                **rubric,
                "rebuild_policy": "retained_unlocatable_basis_unchanged",
            }
        old_end = old_start + len(basis)
    delta = 0
    for patch in patches:
        if old_start < patch["end_char"] and old_end > patch["start_char"]:
            raise ValueError(f"retained rubric overlaps patch: {rubric['rubric_id']}")
        if patch["end_char"] <= old_start:
            delta += len(str(patch["replacement_span"])) - (patch["end_char"] - patch["start_char"])
    new_start, new_end = old_start + delta, old_end + delta
    if prompt_prime[new_start:new_end] != basis:
        raise ValueError(f"retained basis projection mismatch: {rubric['rubric_id']}")
    return {
        **rubric,
        "basis": basis,
        "basis_start": new_start,
        "basis_end": new_end,
        "rebuild_policy": "retained_deterministic_offset_projection",
    }


def rebuild_request(
    row: dict[str, Any], affected_chunk: list[dict[str, Any]], previous_error: str | None,
) -> str:
    retry = ""
    if previous_error:
        specialized = "Repair only that defect."
        if "quoted_span" in previous_error:
            specialized = (
                "Rewrite the failing question without quotation-mark delimiters. If a spoken "
                "quote is essential, copy one exact contiguous span from P PRIME character-for-character."
            )
        elif "visual_rebuilt_rubric" in previous_error:
            specialized = (
                "Rewrite the failing proposition using audible wording only; remove references to "
                "video, screen, visuals, frames, showing, seeing, or appearance."
            )
        elif "invalid_basis_offsets" in previous_error or "basis_cuts_word" in previous_error:
            specialized = (
                "Copy a non-empty exact contiguous basis string from P PRIME. The caller derives "
                "its offsets deterministically; do not paraphrase or join separated spans."
            )
        elif "relation_subtype" in previous_error:
            specialized = (
                "Return the relation subtype actually stated by the rewritten question and P PRIME. "
                "A directly edited relation must use a valid flipped subtype."
            )
        elif "evidence_question_mismatch" in previous_error:
            specialized = (
                "Choose a longer exact contiguous evidence span from P PRIME that directly "
                "supports the complete question, including its main event and claimed property."
            )
        elif "property_missing_from_basis" in previous_error:
            specialized = (
                "Expand the exact evidence span to include the audible property named by the "
                "question, such as foreground/background, language, or tone."
            )
        elif "relation_evidence_families_missing" in previous_error:
            specialized = (
                "Return one exact contiguous basis spanning the explicit relation phrase and "
                "both audible endpoint events named by the question."
            )
        elif "incomplete_basis_boundary" in previous_error:
            specialized = (
                "Extend the exact basis through the end of the phrase or sentence; never end an "
                "evidence span on a dangling article, preposition, or conjunction."
            )
        retry = f"\nPREVIOUS VALIDATOR ERROR: {previous_error}\n{specialized}"
    affected = [{key: rubric.get(key) for key in (
        "rubric_id", "axis", "scope", "domain", "subtype", "question",
        "basis", "target_ids", "parent_ids",
    )} for rubric in affected_chunk]
    required_ids = [rubric["rubric_id"] for rubric in affected_chunk]
    return f"""Locally rebuild only the affected audio rubrics after exact prompt patches.

No audio and no perceptual labels are available. Preserve each rubric ID, axis, scope, domain,
target_ids, and parent_ids. Preserve subtype except for a directly edited relation rubric; for
that rubric, return the relation subtype actually expressed in P PRIME. Rewrite only the question
and its evidence span so it states the corresponding claim in P PRIME. Evidence must be audio-only.

Write the complete question in the same language as P PRIME. If the evidence contains spoken
words, copy those spoken words character-for-character from P PRIME inside the question; never
translate, paraphrase, or otherwise alter a quoted transcript.

Return JSON only as {{"rubrics":[...]}}. Return exactly one item for every affected rubric ID.
Each item contains rubric_id, subtype, question, and basis. basis must be one exact contiguous
substring copied character-for-character from P PRIME. basis_start and basis_end are optional;
the caller derives authoritative zero-based half-open offsets from the exact basis string.

REQUIRED RUBRIC IDS (return this exact set, no omissions or additions):
{json.dumps(required_ids, ensure_ascii=False)}

PATCHES:
{json.dumps(row['patches'], ensure_ascii=False, indent=2)}

AFFECTED ORIGINAL RUBRICS:
{json.dumps(affected, ensure_ascii=False, indent=2)}

P PRIME:
{row['prompt_prime']}
{retry}"""


def validate_graph(rubrics: list[dict[str, Any]]) -> None:
    ids = {row["rubric_id"] for row in rubrics}
    children: dict[str, list[str]] = defaultdict(list)
    for rubric in rubrics:
        for parent in rubric.get("parent_ids") or []:
            if parent not in ids:
                raise ValueError(f"unknown parent {parent} for {rubric['rubric_id']}")
            children[parent].append(rubric["rubric_id"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"rubric_dependency_cycle:{node}")
        if node in visited:
            return
        visiting.add(node)
        for child in children[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for rubric_id in ids:
        visit(rubric_id)


def basis_cuts_word(prompt: str, start: int, end: int) -> bool:
    if start > 0 and prompt[start - 1].isalnum() and prompt[start].isalnum():
        return True
    if end < len(prompt) and prompt[end - 1].isalnum() and prompt[end].isalnum():
        return True
    return False


def normalize_rebuild_offsets(prompt: str, rubric: dict[str, Any]) -> dict[str, Any]:
    """Expand a model-proposed evidence span to whole words and enclosing sentences."""
    item = dict(rubric)
    proposed_basis = str(item.get("basis") or "")
    if proposed_basis:
        occurrences = [match.start() for match in re.finditer(re.escape(proposed_basis), prompt)]
        if len(occurrences) == 1:
            item["basis_start"] = occurrences[0]
            item["basis_end"] = occurrences[0] + len(proposed_basis)
            item["basis_resolution"] = "exact_unique_model_basis"
    start, end = item.get("basis_start"), item.get("basis_end")
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(prompt)):
        return item
    original = (start, end)
    while start > 0 and prompt[start - 1].isalnum() and prompt[start].isalnum():
        start -= 1
    while end < len(prompt) and prompt[end - 1].isalnum() and prompt[end].isalnum():
        end += 1
    sentence_boundaries = list(re.finditer(r"[.!?](?:\s+|$)", prompt))
    left_candidates = [match.end() for match in sentence_boundaries if match.end() <= start]
    if left_candidates:
        start = max(left_candidates)
        while start < len(prompt) and prompt[start].isspace():
            start += 1
    right_boundary = next(
        (match.start() + 1 for match in sentence_boundaries if match.start() + 1 >= end),
        len(prompt),
    )
    end = right_boundary
    item["basis_start"], item["basis_end"] = start, end
    if (start, end) != original:
        item["offset_normalization"] = {
            "policy": "expand_to_whole_word_and_sentence_boundaries",
            "model_basis_start": original[0],
            "model_basis_end": original[1],
        }
    return item


def relation_subtype_from_question(question: str) -> str | None:
    checks = (
        ("overlap", r"\b(?:simultaneous(?:ly)?|at the same time|overlap(?:s|ping)?|while|during|meanwhile|concurrent(?:ly)?)\b|(?:同时|重叠|期间)"),
        ("background_of", r"\b(?:in the background|background to|behind)\b|(?:在.*背景|作为背景)"),
        ("foreground_of", r"\b(?:in the foreground|foreground to|over the top of)\b|(?:在.*前景|作为前景)"),
        ("before", r"\b(?:before|prior to|precede[sd]?|followed by)\b|(?:之前|先于|随后.*才)"),
        ("after", r"\b(?:after|following (?:the )?(?:sound|speech|music|event|noise|voice|narration|dialogue|monologue)|once .* (?:ends?|finishes?))\b|(?:之后|晚于|结束后)"),
    )
    found = [subtype for subtype, pattern in checks if re.search(pattern, question, re.I)]
    return found[0] if len(found) == 1 else None


def question_frame_language_with_prompt(question: str, prompt: str) -> str:
    """Ignore source-exact bilingual parenthetical glosses in language checks."""
    cleaned = question
    for match in reversed(list(re.finditer(r"\([^()]+\)", question))):
        value = match.group(0)
        if value in prompt and (
            re.search(r"[\u3400-\u9fff]", value)
            or re.search(r"[A-Za-z]", value)
        ):
            cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]
    return question_frame_language(cleaned)


def lexical_content_words(text: str) -> set[str]:
    words: set[str] = set()
    for value in re.findall(r"[A-Za-z]+", text.lower()):
        if value in QUESTION_STOP_WORDS or len(value) < 3:
            continue
        if value.endswith("ing") and len(value) > 5:
            value = value[:-3]
        elif value.endswith("ed") and len(value) > 4:
            value = value[:-2]
        elif value.endswith("s") and len(value) > 4:
            value = value[:-1]
        words.add(value)
    return words


def question_basis_coverage(question: str, basis: str) -> float:
    expected = lexical_content_words(question)
    if not expected:
        return 1.0
    observed = lexical_content_words(basis)
    return len(expected & observed) / len(expected)


def validate_rebuild_result(
    manifest: dict[str, Any], result: dict[str, Any], expected_ids: set[str] | None = None,
) -> None:
    expected = expected_ids or {row["rubric_id"] for row in manifest["affected_rubrics"]}
    rubrics = result.get("rubrics") or []
    actual = {row.get("rubric_id") for row in rubrics}
    if actual != expected or len(rubrics) != len(expected):
        raise ValueError(
            f"affected_rubric_id_set_mismatch:missing={sorted(expected - actual)};"
            f"extra={sorted(map(str, actual - expected))}; return exactly {sorted(expected)}"
        )
    prompt = manifest["prompt_prime"]
    prompt_quotes = quoted_spans(prompt)
    original_by_id = {
        str(row["rubric_id"]): row for row in manifest["affected_rubrics"]
    }
    target_language = manifest.get("language")
    direct_targets = set(map(str, manifest.get("direct_target_rubric_ids") or []))
    for rubric in rubrics:
        rubric_id = str(rubric.get("rubric_id"))
        start, end = rubric.get("basis_start"), rubric.get("basis_end")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(prompt)):
            raise ValueError(f"invalid_basis_offsets:{rubric.get('rubric_id')}")
        basis = prompt[start:end]
        question = str(rubric.get("question") or "")
        if not basis.strip() or not question.strip():
            raise ValueError(f"empty_rebuilt_rubric:{rubric.get('rubric_id')}")
        if basis_cuts_word(prompt, start, end):
            raise ValueError(f"basis_cuts_word:{rubric_id}:{start}:{end}")
        original_subtype = str(original_by_id[rubric_id].get("subtype") or "")
        returned_subtype = str(rubric.get("subtype") or "")
        original_kind = rubric_kind(original_by_id[rubric_id])
        if original_kind == "relation" and rubric_id in direct_targets:
            allowed = RELATION_FLIPS.get(original_subtype, set())
            if returned_subtype not in allowed:
                raise ValueError(
                    f"relation_subtype_not_flipped:{rubric_id}:{original_subtype}->{returned_subtype}"
                )
        elif returned_subtype != original_subtype:
            raise ValueError(
                f"subtype_changed_without_direct_relation_edit:"
                f"{rubric_id}:{original_subtype}->{returned_subtype}"
            )
        if original_kind == "relation":
            inferred_subtype = relation_subtype_from_question(question)
            if inferred_subtype != returned_subtype:
                raise ValueError(
                    f"relation_subtype_question_mismatch:"
                    f"{rubric_id}:{returned_subtype}!={inferred_subtype}"
                )
            basis_subtype = relation_subtype_from_question(basis)
            if basis_subtype != returned_subtype:
                raise ValueError(
                    f"relation_subtype_basis_mismatch:"
                    f"{rubric_id}:{returned_subtype}!={basis_subtype}"
                )
        coverage = question_basis_coverage(question, basis)
        if target_language == "en" and coverage < 0.30:
            raise ValueError(
                f"evidence_question_mismatch:{rubric_id}:coverage={coverage:.3f}; "
                "select an exact evidence span that directly supports the complete question"
            )
        if original_kind == "relation":
            question_families = detected_event_families(question)
            basis_families = detected_event_families(basis)
            missing_families = question_families - basis_families
            if missing_families:
                raise ValueError(
                    f"relation_evidence_families_missing:{rubric_id}:"
                    f"{sorted(missing_families)}"
                )
        basis_words = re.findall(r"[A-Za-z]+", basis.lower())
        if target_language == "en" and basis_words and basis_words[-1] in DANGLING_BASIS_END_WORDS:
            raise ValueError(
                f"incomplete_basis_boundary:{rubric_id}:ends_with={basis_words[-1]}"
            )
        property_terms = {
            "placement": {"foreground", "background"},
            "language": {"english", "chinese", "mandarin", "cantonese"},
            "tone": {"calm", "direct", "frantic", "shouting", "agitated", "soft", "loud"},
        }.get(returned_subtype, set())
        asked_properties = property_terms & lexical_content_words(question)
        if asked_properties and not (asked_properties & lexical_content_words(basis)):
            raise ValueError(
                f"property_missing_from_basis:{rubric_id}:{sorted(asked_properties)}"
            )
        # The evaluator receives the question, not the provenance basis. A
        # relation basis may span adjacent visual wording while its proposition
        # remains strictly audible; reject visual leakage in the proposition.
        if VISUAL_RE.search(question):
            raise ValueError(f"visual_rebuilt_rubric:{rubric.get('rubric_id')}")
        frame_language = question_frame_language_with_prompt(question, prompt)
        if target_language in {"zh", "en"} and frame_language != target_language:
            raise ValueError(
                f"wrong_question_language:{rubric_id}:{frame_language}!={target_language}"
            )
        if question.strip() == str(original_by_id[rubric_id].get("question") or "").strip():
            continue
        for quote in quoted_spans(question):
            if quote in prompt:
                continue
            if any(normalized_text(quote) == normalized_text(value) for value in prompt_quotes):
                raise ValueError(f"quoted_span_punctuation_drift:{rubric_id}:{quote}")
            raise ValueError(f"quoted_span_not_in_prompt_prime:{rubric_id}:{quote}")


def validate_stored_rebuild(manifest: dict[str, Any], row: dict[str, Any]) -> None:
    validate_rebuild_result(manifest, row)
    validate_graph(row["final_rubrics"])


async def rebuild_async(args: argparse.Namespace) -> dict[str, Any]:
    require_model_execution(args)
    if args.rubric_chunk_size <= 0:
        raise ValueError("--rubric-chunk-size must be positive")
    from openai import AsyncOpenAI

    run = args.run.resolve()
    manifests = read_jsonl(run / "rubrics/rebuild_manifest.jsonl")
    if args.limit:
        manifests = manifests[:args.limit]
    manifest_by_id = {row["sample_id"]: row for row in manifests}
    output = run / "model/rubric_rebuild_attempts.jsonl"
    completed = latest_successes(output, manifest_by_id, validate_stored_rebuild)
    pending = [row for row in manifests if row["sample_id"] not in completed]
    cached_chunks: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    if output.exists():
        for attempt_row in read_jsonl(output):
            sample_id = str(attempt_row.get("sample_id") or "")
            manifest = manifest_by_id.get(sample_id)
            if attempt_row.get("status") != "chunk_ok" or manifest is None:
                continue
            expected_ids = tuple(sorted(map(str, attempt_row.get("chunk_rubric_ids") or [])))
            proposed = list(attempt_row.get("proposed_rubrics") or [])
            try:
                validate_rebuild_result(manifest, {"rubrics": proposed}, set(expected_ids))
            except Exception:
                continue
            cached_chunks[(sample_id, expected_ids)] = {
                "rubrics": proposed,
                "request_sha256": attempt_row.get("request_sha256"),
            }
    endpoints = [value.rstrip("/") for value in args.endpoint]
    clients = [AsyncOpenAI(base_url=value, api_key="EMPTY", timeout=args.timeout) for value in endpoints]
    semaphores = [asyncio.Semaphore(args.concurrency_per_endpoint) for _ in endpoints]

    async def one(manifest: dict[str, Any], endpoint_index: int) -> dict[str, Any]:
        original_by_id = {row["rubric_id"]: row for row in manifest["affected_rubrics"]}
        affected_rows = sorted(manifest["affected_rubrics"], key=lambda row: row["rubric_id"])
        chunks = [
            affected_rows[index:index + args.rubric_chunk_size]
            for index in range(0, len(affected_rows), args.rubric_chunk_size)
        ]
        combined: dict[str, dict[str, Any]] = {}
        request_hashes: list[str] = []
        reused_chunk_count = 0
        for chunk_index, chunk in enumerate(chunks, 1):
            expected_ids = {row["rubric_id"] for row in chunk}
            cache_key = (manifest["sample_id"], tuple(sorted(map(str, expected_ids))))
            cached = cached_chunks.get(cache_key)
            if cached is not None:
                combined.update({row["rubric_id"]: row for row in cached["rubrics"]})
                if cached.get("request_sha256"):
                    request_hashes.append(str(cached["request_sha256"]))
                reused_chunk_count += 1
                continue
            previous_error: str | None = None
            chunk_succeeded = False
            for attempt in range(1, args.retries + 1):
                prompt = rebuild_request(manifest, chunk, previous_error)
                request_sha = text_sha256(prompt)
                raw = ""
                started = now()
                try:
                    async with semaphores[endpoint_index]:
                        parsed, raw = await call_model_json(
                            clients[endpoint_index], args.model, prompt, args.max_tokens, "rubrics",
                        )
                    proposed = {
                        row["rubric_id"]: normalize_rebuild_offsets(manifest["prompt_prime"], row)
                        for row in parsed.get("rubrics") or []
                    }
                    validate_rebuild_result(
                        manifest, {"rubrics": list(proposed.values())}, expected_ids,
                    )
                    combined.update(proposed)
                    request_hashes.append(request_sha)
                    append_jsonl(output, {
                        "schema_version": SCHEMA_VERSION,
                        "timestamp": now(), "started_at": started,
                        "stage": "rebuild", "sample_id": manifest["sample_id"],
                        "status": "chunk_ok", "attempt": attempt,
                        "chunk_index": chunk_index, "chunk_count": len(chunks),
                        "chunk_rubric_ids": sorted(expected_ids),
                        "endpoint": endpoints[endpoint_index], "model": args.model,
                        "request_sha256": request_sha,
                        "raw_model_output": raw,
                        "proposed_rubrics": list(proposed.values()),
                        "label_visibility": "none",
                    })
                    chunk_succeeded = True
                    break
                except Exception as exc:
                    previous_error = repr(exc)
                    append_jsonl(output, {
                        "schema_version": SCHEMA_VERSION,
                        "timestamp": now(), "started_at": started,
                        "stage": "rebuild", "sample_id": manifest["sample_id"],
                        "status": "chunk_error", "attempt": attempt,
                        "chunk_index": chunk_index, "chunk_count": len(chunks),
                        "chunk_rubric_ids": sorted(expected_ids),
                        "endpoint": endpoints[endpoint_index], "model": args.model,
                        "request_sha256": request_sha,
                        "raw_model_output": raw,
                        "error": previous_error,
                        "label_visibility": "none",
                    })
            if not chunk_succeeded:
                return {
                    "sample_id": manifest["sample_id"], "status": "error",
                    "error": previous_error, "failed_chunk": chunk_index,
                }

        validate_rebuild_result(manifest, {"rubrics": list(combined.values())})
        rebuilt: list[dict[str, Any]] = []
        for rubric_id, original in original_by_id.items():
            item = combined[rubric_id]
            start, end = item["basis_start"], item["basis_end"]
            rebuilt.append({
                **original,
                "subtype": str(item["subtype"]),
                "question": str(item["question"]),
                "basis": manifest["prompt_prime"][start:end],
                "basis_start": start,
                "basis_end": end,
                "rebuild_policy": "local_chunked_llm_question_and_offsets",
            })
        retained = [translate_retained_rubric(
            rubric, manifest["original_prompt"], manifest["prompt_prime"], manifest["patches"],
        ) for rubric in manifest["retained_rubrics"]]
        final_rubrics = sorted(rebuilt + retained, key=lambda row: row["rubric_id"])
        validate_graph(final_rubrics)
        row = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": now(), "stage": "rebuild", "sample_id": manifest["sample_id"],
            "status": "ok", "attempt": max(1, len(chunks)),
            "chunk_count": len(chunks),
            "reused_chunk_count": reused_chunk_count,
            "endpoint": endpoints[endpoint_index], "model": args.model,
            "request_sha256s": request_hashes,
            "rubrics": rebuilt,
            "retained_rubrics": retained,
            "final_rubrics": final_rubrics,
            "label_visibility": "none", "label_status": "pending",
        }
        validate_stored_rebuild(manifest, row)
        append_jsonl(output, row)
        return row

    tasks = [asyncio.create_task(one(row, index % len(endpoints))) for index, row in enumerate(pending)]
    results = [await task for task in asyncio.as_completed(tasks)]
    summary = {
        "scheduled": len(pending),
        "ok": sum(row.get("status") == "ok" for row in results),
        "error": sum(row.get("status") != "ok" for row in results),
        "already_completed": len(completed),
        "model": args.model, "endpoints": endpoints,
        "attempt_log": output_record(output) if output.exists() else None,
    }
    atomic_write_json(run / "rubrics/rebuild_summary.json", summary)
    return summary


def rebuild(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    with stage_event(run, "rebuild", {"model": args.model, "endpoints": args.endpoint}) as event:
        summary = asyncio.run(rebuild_async(args))
        event["summary"] = summary
        return summary


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    with stage_event(run, "audit") as event:
        pairs = read_jsonl(run / "pairs/unlabeled_pairs.jsonl")
        manifests = {row["sample_id"]: row for row in read_jsonl(run / "rubrics/rebuild_manifest.jsonl")}
        rebuilds = latest_successes(
            run / "model/rubric_rebuild_attempts.jsonl", manifests, validate_stored_rebuild,
        )
        if not args.allow_partial and len(rebuilds) != len(pairs):
            raise ValueError(f"rubric rebuild incomplete: {len(rebuilds)}/{len(pairs)}")
        final: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for pair in pairs:
            sample_id = pair["sample_id"]
            try:
                assert_no_label_keys(pair)
                if pair["label_visibility"] != "none" or pair["label_status"] != "pending":
                    raise ValueError("invalid_label_state")
                if not pair["audio_sha256"]:
                    raise ValueError("missing_audio_sha256")
                if file_sha256(Path(pair["audio_path"])) != pair["audio_sha256"]:
                    raise ValueError("audio_sha_mismatch")
                if sample_id not in manifests:
                    raise ValueError("missing_rebuild_manifest")
                rebuilt = rebuilds.get(sample_id)
                if rebuilt is None:
                    raise ValueError("missing_rebuilt_rubrics")
                final_row = {**pair, "rubrics": rebuilt["final_rubrics"]}
                assert_no_label_keys(final_row)
                validate_graph(final_row["rubrics"])
                final.append(final_row)
            except Exception as exc:
                errors.append({"sample_id": sample_id, "error": repr(exc)})
        if errors and not args.allow_partial:
            raise ValueError(f"audit failed for {len(errors)} rows; first={errors[:3]}")
        final_path = run / "output/path1_negative_unlabeled.jsonl"
        errors_path = run / "audit/errors.jsonl"
        atomic_write_jsonl(final_path, final)
        atomic_write_jsonl(errors_path, errors)
        summary = {
            "input_pairs": len(pairs), "accepted": len(final), "rejected": len(errors),
            "label_status": "pending", "label_visibility": "none",
            "categories": dict(Counter(row["disturbance_category"] for row in final)),
            "unique_sample_ids": len({row["sample_id"] for row in final}),
            "unique_parent_ids": len({row["parent_positive_id"] for row in final}),
            "audio_byte_identical": sum(bool(row["audio_byte_identical_to_parent"]) for row in final),
            "outputs": [output_record(final_path), output_record(errors_path)],
        }
        atomic_write_json(run / "audit/summary.json", summary)
        event["summary"] = summary
        return summary


def count_latest_ok(path: Path) -> int:
    if not path.exists():
        return 0
    latest: dict[str, str] = {}
    for row in read_jsonl(path):
        latest[str(row.get("sample_id"))] = str(row.get("status"))
    return sum(status == "ok" for status in latest.values())


def status(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    paths = {
        "frozen_sources": run / "inputs/path1_positive_frozen.jsonl",
        "planned": run / "plan/counterfactual_plan.jsonl",
        "patch_attempts": run / "model/patch_attempts.jsonl",
        "pairs": run / "pairs/unlabeled_pairs.jsonl",
        "rebuild_attempts": run / "model/rubric_rebuild_attempts.jsonl",
        "final": run / "output/path1_negative_unlabeled.jsonl",
    }
    result = {
        "run": str(run),
        "label_visibility": "none",
        "counts": {
            name: (len(read_jsonl(path)) if path.exists() and "attempts" not in name else count_latest_ok(path))
            for name, path in paths.items()
        },
        "files": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    require_model_execution(args)
    if args.skip_audio_hash:
        raise ValueError("run refuses --skip-audio-hash; it is available only for prepare preflight")
    results = {
        "prepare": prepare(args),
        "generate": generate(args),
        "materialize": materialize(args),
        "rebuild": rebuild(args),
        "audit": audit(args),
    }
    return results


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="Dedicated run directory")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", action="append", required=True, help="Explicit local OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--execute-token", required=True)
    parser.add_argument("--concurrency-per-endpoint", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--rubric-chunk-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Freeze label-blind inputs and build plan")
    add_common(prepare_parser)
    prepare_parser.add_argument("--input", type=Path, help="Legacy combined construction manifest")
    prepare_parser.add_argument("--positive-manifest", type=Path, help="Route 1 positive audio-caption manifest")
    prepare_parser.add_argument("--rubric-manifest", type=Path, help="Independent rubric-decomposition results")
    prepare_parser.add_argument("--expected-sources", type=int, default=0, help="Optional exact count check; 0 disables it")
    prepare_parser.add_argument("--target-count", type=int, default=0, help="0 means all balanced feasible sources")
    prepare_parser.add_argument("--skip-audio-hash", action="store_true", help="Testing only; forbidden for final audit")

    generate_parser = subparsers.add_parser("generate", help="Generate exact-span P-prime patches")
    add_common(generate_parser)
    add_model_args(generate_parser)

    materialize_parser = subparsers.add_parser("materialize", help="Freeze validated P-prime pairs")
    add_common(materialize_parser)
    materialize_parser.add_argument("--allow-partial", action="store_true")
    materialize_parser.add_argument("--limit", type=int, default=0)

    rebuild_parser = subparsers.add_parser("rebuild", help="Locally rebuild affected P-prime rubrics")
    add_common(rebuild_parser)
    add_model_args(rebuild_parser)

    audit_parser = subparsers.add_parser("audit", help="Audit and assemble unlabeled output")
    add_common(audit_parser)
    audit_parser.add_argument("--allow-partial", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show checkpoint counts")
    add_common(status_parser)

    run_parser = subparsers.add_parser("run", help="Run all stages with explicit model authorization")
    add_common(run_parser)
    run_parser.add_argument("--input", type=Path, help="Legacy combined construction manifest")
    run_parser.add_argument("--positive-manifest", type=Path, help="Route 1 positive audio-caption manifest")
    run_parser.add_argument("--rubric-manifest", type=Path, help="Independent rubric-decomposition results")
    run_parser.add_argument("--expected-sources", type=int, default=0, help="Optional exact count check; 0 disables it")
    run_parser.add_argument("--target-count", type=int, default=0)
    run_parser.add_argument("--skip-audio-hash", action="store_true")
    run_parser.add_argument("--allow-partial", action="store_true")
    add_model_args(run_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with run_lock(args.run.resolve()):
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "generate":
            result = generate(args)
        elif args.command == "materialize":
            result = materialize(args)
        elif args.command == "rebuild":
            result = rebuild(args)
        elif args.command == "audit":
            result = audit(args)
        elif args.command == "status":
            status(args)
            return
        elif args.command == "run":
            result = run_all(args)
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
