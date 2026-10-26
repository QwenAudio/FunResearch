from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import traceback
import uuid
import wave
import sys
from functools import partial
from pathlib import Path
from typing import Any


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required. FineAudio-Judge does not provide model services; "
            "set it to your own local checkout, model, or endpoint."
        )
    return value


REPO = Path(_required_env("AUDIO_OSCAR_ROOT")).expanduser().resolve()
sys.path.insert(0, str(REPO))
EXPECTED_AUDIO_OSCAR_COMMIT = "aca87d722eaa00265023817cf42f29f6a3c1ad75"

import pipeline as pipeline_module
from agents.final_editor import FinalAudioAuditor
from agents.audio_planner import DraftAudioPlanner
from agents.llm import OpenAILLM
from agents.speech_generator import TTSConfig
from agents.timestamp_planner import TimestampPlanner
from executor.speech_fa import SpeechFAExecutor
from executor.tta_quality_loop import AudioCriticSettings, TTAQualityLoop
from executor.voice_design_executor import VoiceDesignExecutor
from pipeline import AudioOscarPipeline, PipelineOptions
from tts import CosyVoice3, Qwen3TTSBase, VoxCPM2
from structure.plan import Plan
from utils.executor_utils import normalize_audio_type
from verified_speech_generation import VerifiedSpeechGenerationExecutor


def _is_transient_service_error(exc: BaseException) -> bool:
    """Return True for transport/service outages, not pipeline quality failures."""

    current: BaseException | None = exc
    seen: set[int] = set()
    class_markers = {
        "apiconnectionerror",
        "apitimeouterror",
        "connecterror",
        "connectionerror",
        "connecttimeout",
        "readtimeout",
        "remotedisconnected",
    }
    message_markers = (
        "connection refused",
        "connection reset",
        "connection error",
        "server disconnected",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__.lower() in class_markers:
            return True
        message = str(current).lower()
        if any(marker in message for marker in message_markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def _run_one_waiting_for_services(
    record: dict[str, Any], output_root: Path,
    runtime_provenance: dict[str, Any] | None = None,
):
    """Keep the same sample in place while a required model service restarts."""

    outage_count = 0
    while True:
        try:
            return run_one(record, output_root, runtime_provenance=runtime_provenance)
        except Exception as exc:
            if not _is_transient_service_error(exc):
                raise
            outage_count += 1
            wait_seconds = min(60, 10 * outage_count)
            print(
                "WAIT transient model-service outage; retrying the same sample "
                f"in {wait_seconds}s without consuming a sample attempt: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(wait_seconds)


PLANNER_URL = _required_env("AUDIO_OSCAR_PLANNER_URL")
PLANNER_MODEL = _required_env("AUDIO_OSCAR_PLANNER_MODEL")
PLANNER_TEMPERATURE = float(os.environ.get("AUDIO_OSCAR_PLANNER_TEMPERATURE", "0.1"))
PLANNER_TIMEOUT = float(os.environ.get("AUDIO_OSCAR_PLANNER_TIMEOUT", "900"))
SPEECH_FA_MODE = os.environ.get("AUDIO_OSCAR_SPEECH_FA_MODE", "llm_auto")
PLANNER_ENABLE_THINKING = os.environ.get("AUDIO_OSCAR_PLANNER_ENABLE_THINKING", "false").lower() == "true"
PAPER_EXPERIMENT_MODE = (
    os.environ.get("AUDIO_OSCAR_PAPER_EXPERIMENT_MODE", "false").lower() == "true"
)
ALLOW_DRAFT_TIMELINE_FALLBACK = (
    os.environ.get("AUDIO_OSCAR_ALLOW_DRAFT_TIMELINE_FALLBACK", "true").lower() == "true"
)
OMNI_URL = _required_env("AUDIO_OSCAR_OMNI_URL")
OMNI_MODEL = _required_env("AUDIO_OSCAR_OMNI_MODEL")
QWEN_TTS_URL = _required_env("AUDIO_OSCAR_QWEN_TTS_URL")
COSYVOICE_URL = _required_env("AUDIO_OSCAR_COSYVOICE_URL")
VOXCPM_URL = _required_env("AUDIO_OSCAR_VOXCPM_URL")
SPEECH_GENERATION_WORKERS = int(os.environ.get("AUDIO_OSCAR_SPEECH_GENERATION_WORKERS", "1"))
TTA_GENERATION_WORKERS = int(os.environ.get("AUDIO_OSCAR_TTA_GENERATION_WORKERS", "4"))
AUDIO_CRITIC_WORKERS = int(os.environ.get("AUDIO_OSCAR_AUDIO_CRITIC_WORKERS", "4"))
FA_URL = _required_env("AUDIO_OSCAR_FA_URL")
STABLE_AUDIO_URL = _required_env("AUDIO_OSCAR_STABLE_AUDIO_URL")
ACESTEP_URL = _required_env("AUDIO_OSCAR_ACESTEP_URL")
DEMUCS_HOST = _required_env("AUDIO_OSCAR_DEMUCS_HOST")
DEMUCS_PORT = _required_env("AUDIO_OSCAR_DEMUCS_PORT")
VOICE_DESIGN_PATH = _required_env("AUDIO_OSCAR_VOICE_DESIGN_MODEL")
BASE_TTS_PATH = _required_env("AUDIO_OSCAR_BASE_TTS_MODEL")
SONG_CONFIG_PATH = Path(_required_env("AUDIO_OSCAR_SONG_CONFIG_PATH"))
TTA_CONFIG_PATH = Path(_required_env("AUDIO_OSCAR_TTA_CONFIG_PATH"))
MUSIC_CONFIG_PATH = Path(_required_env("AUDIO_OSCAR_MUSIC_CONFIG_PATH"))
RUN_FINAL_EDIT = os.environ.get("AUDIO_OSCAR_RUN_FINAL_EDIT", "true").lower() == "true"


def audio_oscar_provenance() -> dict[str, Any]:
    if not REPO.is_dir():
        raise FileNotFoundError(f"AUDIO_OSCAR_ROOT is not a directory: {REPO}")
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot resolve Audio-Oscar git commit at {REPO}") from error
    commit = result.stdout.strip()
    allow_unpinned = os.environ.get("AUDIO_OSCAR_ALLOW_UNPINNED", "false").lower() == "true"
    if commit != EXPECTED_AUDIO_OSCAR_COMMIT and not allow_unpinned:
        raise RuntimeError(
            "Audio-Oscar commit mismatch: "
            f"expected {EXPECTED_AUDIO_OSCAR_COMMIT}, got {commit}. "
            "Use the pinned checkout plus the released patch, or set "
            "AUDIO_OSCAR_ALLOW_UNPINNED=true for an explicitly non-reproduction run."
        )
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    patch_path = Path(__file__).resolve().parent.parent / "patches/audio_oscar.patch"
    patch_applied = patch_path.is_file() and subprocess.run(
        ["git", "-C", str(REPO), "apply", "--reverse", "--check", str(patch_path)],
        capture_output=True, text=True,
    ).returncode == 0
    if not patch_applied and not allow_unpinned:
        raise RuntimeError(
            f"released Audio-Oscar patch is not applied cleanly at {REPO}: {patch_path}"
        )
    patch_sha256 = (
        hashlib.sha256(patch_path.read_bytes()).hexdigest() if patch_path.is_file() else None
    )
    return {
        "root": str(REPO),
        "git_commit": commit,
        "expected_git_commit": EXPECTED_AUDIO_OSCAR_COMMIT,
        "unpinned_override": allow_unpinned,
        "working_tree_dirty": bool(status),
        "working_tree_status": status,
        "released_patch": str(patch_path),
        "released_patch_applied": patch_applied,
        "released_patch_sha256": patch_sha256,
    }


def _sanitize_speech_plans(speech_plan, generation_plan) -> tuple[list[str], list[str]]:
    """Drop non-verbal empty TTS rows and restore accidentally blank real dialogue.

    Non-verbal vocalizations such as "a baby responds" belong to the TTA event
    plan.  A planner may nevertheless emit an empty speech row for them, which
    no TTS backend can execute.  Conversely, if the generation-plan LLM blanks
    text that was non-empty in the extracted speech plan, preserve the original
    dialogue instead of silently dropping it.
    """

    original_text_by_id = {
        str(getattr(segment, "segment_id", "")): str(getattr(segment, "text", ""))
        for segment in speech_plan.speech_segments
    }
    removed_ids = {
        segment_id
        for segment_id, text in original_text_by_id.items()
        if not text.strip()
    }
    speech_plan.speech_segments = [
        segment
        for segment in speech_plan.speech_segments
        if str(getattr(segment, "text", "")).strip()
    ]

    repaired_ids: list[str] = []
    retained_items = []
    for item in generation_plan.items:
        segment_id = str(getattr(item, "segment_id", ""))
        if segment_id in removed_ids:
            continue
        if not str(getattr(item, "text", "")).strip():
            original_text = original_text_by_id.get(segment_id, "")
            if not original_text.strip():
                removed_ids.add(segment_id)
                continue
            item.text = original_text
            repaired_ids.append(segment_id)
        retained_items.append(item)
    generation_plan.items = retained_items
    return sorted(removed_ids), sorted(repaired_ids)


class FullASGPipeline(AudioOscarPipeline):
    """Official Audio-Oscar orchestration with the paper's ASG model routing."""

    def __init__(self, *args, duration_contract: dict[str, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration_contract = duration_contract

    def _duration_planning_context(
        self, text_context: dict[str, Any], feedback: str = ""
    ) -> dict[str, Any]:
        if not self.duration_contract:
            return dict(text_context)
        context = dict(text_context)
        minimum = self.duration_contract["min_duration_seconds"]
        maximum = self.duration_contract["max_duration_seconds"]
        target = self.duration_contract["target_duration_seconds"]
        contract = (
            "\n\nPRODUCTION TIMING CONTRACT (planning metadata, not story content):\n"
            f"- Complete mixed timeline must be {minimum:.1f}-{maximum:.1f} seconds; "
            f"target approximately {target:.1f} seconds.\n"
            "- Distribute grounded audible events across the timeline. Do not meet the "
            "duration using artificial leading/trailing silence.\n"
            "- When the prompt contains a sustained ambience or music bed, extend that "
            "same grounded cue naturally across the scene.\n"
            "- Preserve all explicit events, exact speech, counts, and temporal relations."
        )
        if feedback:
            contract += f"\n- Previous plan failed validation: {feedback}"
        context["text"] = str(context.get("text", "")).rstrip() + contract
        return context

    def _duration_issues(self, segments: list[dict[str, Any]]) -> list[str]:
        if not self.duration_contract:
            return []
        if not segments:
            return ["empty timeline"]
        starts = [max(float(item.get("start_time", 0.0)), 0.0) for item in segments]
        ends = [max(float(item.get("end_time", 0.0)), 0.0) for item in segments]
        timeline_end = max(ends, default=0.0)
        minimum = self.duration_contract["min_duration_seconds"]
        maximum = self.duration_contract["max_duration_seconds"]
        target = self.duration_contract["target_duration_seconds"]
        required_end = max(minimum + 2.0, target - 4.0)
        issues: list[str] = []
        if timeline_end < required_end:
            issues.append(f"timeline ends at {timeline_end:.2f}s; require >= {required_end:.2f}s")
        if timeline_end > maximum:
            issues.append(f"timeline ends at {timeline_end:.2f}s; require <= {maximum:.2f}s")
        if min(starts, default=0.0) > 2.0:
            issues.append("timeline begins with more than 2 seconds of silence")
        intervals = sorted(
            (start, end) for start, end in zip(starts, ends) if end > start
        )
        occupied = 0.0
        if intervals:
            current_start, current_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    occupied += current_end - current_start
                    current_start, current_end = start, end
            occupied += current_end - current_start
        required_occupied = min(12.0, target * 0.35)
        if occupied < required_occupied:
            issues.append(
                f"only {occupied:.2f}s contain planned audio; require >= {required_occupied:.2f}s"
            )
        return issues

    def _run_event_planning(self, text_context, speech_results):
        if not self.duration_contract:
            return super()._run_event_planning(text_context, speech_results)
        errors = []
        feedback = ""
        for attempt in range(1, 5):
            context = self._duration_planning_context(text_context, feedback)
            event_plan = DraftAudioPlanner(self.llm).plan(context, speech_result=speech_results)
            segments = [item for item in event_plan.get("audio_seg", []) if isinstance(item, dict)]
            issues = self._duration_issues(segments)
            if not issues:
                if attempt > 1:
                    print(f"       Draft timeline passed duration checks on attempt {attempt}.")
                return event_plan
            feedback = "; ".join(issues)
            errors.append(f"attempt {attempt}: {feedback}")
            print(f"       Draft timeline duration retry {attempt}/4: {feedback}", flush=True)
        raise RuntimeError("Draft planner failed duration contract: " + " | ".join(errors))

    def _run_voice_design(self, role_profile):
        executor = VoiceDesignExecutor(
            base_url=QWEN_TTS_URL,
            model=VOICE_DESIGN_PATH,
            timeout=1200.0,
            output_dir=str(self.options.voice_dir),
        )
        return executor.execute(role_profile, max_workers=1, max_new_tokens=512)

    def _run_speech_planning_and_generation_plan(self, text_context, role_profile):
        speech_plan, generation_plan = super()._run_speech_planning_and_generation_plan(
            text_context, role_profile
        )
        removed_ids, repaired_ids = _sanitize_speech_plans(speech_plan, generation_plan)
        if removed_ids:
            print(
                "       Filtered non-verbal empty speech segments from TTS plan: "
                + ", ".join(removed_ids),
                flush=True,
            )
        if repaired_ids:
            print(
                "       Restored original text for blank generation-plan segments: "
                + ", ".join(repaired_ids),
                flush=True,
            )
        return speech_plan, generation_plan

    def _run_timestamp_planning(self, text_context, event_plan, speech_results):
        """Run the official timestamp planner with loss/stacking detection and retries."""

        draft_segments = [item for item in event_plan.get("audio_seg", []) if isinstance(item, dict)]
        feedback = ""
        errors = []
        for attempt in range(1, 5):
            context = self._duration_planning_context(text_context)
            if feedback:
                context["feedback"] = feedback
            payload = TimestampPlanner(self.llm).plan(
                text_context=context,
                event_plan=event_plan,
                speech_result={"segments": speech_results},
            )
            plan = Plan.from_json(payload)
            plan, restored_non_speech = _restore_missing_non_speech(draft_segments, plan)
            issues = _timeline_integrity_issues(draft_segments, plan)
            issues.extend(self._duration_issues([segment.to_dict() for segment in plan.audio_seg]))
            if not issues:
                if restored_non_speech:
                    print(
                        "       Restored omitted draft non-speech events after timestamp "
                        f"refinement: {restored_non_speech}.",
                        flush=True,
                    )
                if attempt > 1:
                    print(f"       Timestamp plan passed integrity checks on attempt {attempt}.")
                return plan
            message = "; ".join(issues)
            errors.append(f"attempt {attempt}: {message}")
            print(f"       Timestamp plan integrity retry {attempt}/4: {message}")
            feedback = (
                "The previous timestamp refinement was invalid: "
                f"{message}. Return the complete timeline, retain supported draft non-speech events "
                "with their segment_id values, and place sequential dialogue at distinct narrative times."
            )
        if ALLOW_DRAFT_TIMELINE_FALLBACK:
            # The upstream event plan is already a complete executable timeline built
            # from the generated speech durations.  If the refinement LLM repeatedly
            # collapses that inventory, retaining the complete upstream plan is safer
            # than accepting a partial rewrite or dropping the sample outright.
            fallback = Plan.from_json({"audio_seg": draft_segments})
            fallback_issues = _timeline_integrity_issues(draft_segments, fallback)
            fallback_issues.extend(
                self._duration_issues([segment.to_dict() for segment in fallback.audio_seg])
            )
            if not fallback_issues:
                print(
                    "       Timestamp planner exhausted retries; using the complete "
                    "upstream Audio-Oscar event timeline as a guarded fallback."
                )
                return fallback
        raise RuntimeError("Timestamp planner failed integrity checks: " + " | ".join(errors))

    def _run_speech_alignment(self, speech_results):
        raw_results = []
        for item in speech_results:
            text = str(item.get("text", ""))
            language = "Chinese" if re.search(r"[\u3400-\u9fff]", text) else "English"
            raw_results.extend(
                SpeechFAExecutor(
                    server_url=FA_URL,
                    timeout=600.0,
                    language=language,
                ).execute(speech_results=[item])
            )
        return pipeline_module._normalize_speech_results(raw_results)

    def _run_audio_generation(self, tta_generation_plan, result_path):
        loop = TTAQualityLoop(
            llm=self.llm,
            tta_config=self.tta_config,
            output_dir=self.options.speech_output_dir,
            audio_timeout=self.options.audio_timeout,
            generation_workers=self.options.tta_generation_workers,
            critic_settings=AudioCriticSettings(
                enabled=True,
                base_url=OMNI_URL,
                model=OMNI_MODEL,
                timeout=self.options.audio_timeout,
                threshold=self.options.audio_critic_threshold,
                max_retries=self.options.audio_critic_max_retries,
                workers=self.options.audio_critic_workers,
            ),
        )
        raw = loop.run(tta_plan=tta_generation_plan, result_json_path=result_path)
        return pipeline_module._normalize_audio_results(raw, result_path.parent)

    def _run_music_generation(self, music_generation_plan, result_path):
        from executor.music_generation import MusicGenerationExecutor

        executor = MusicGenerationExecutor(
            output_dir=str(self.options.speech_output_dir),
            server_url=ACESTEP_URL,
            stable_audio_server_url=STABLE_AUDIO_URL,
            music_config=self._get_music_config(),
            timeout=self.options.music_timeout,
        )
        raw = executor.execute(music_plan=music_generation_plan, result_json_path=result_path)
        return pipeline_module._normalize_audio_results(raw, result_path.parent)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not str(value.get("prompt", "")).strip():
            raise ValueError(f"Invalid prompt record at {path}:{line_number}")
        records.append(value)
    return records


def _timeline_integrity_issues(draft_segments: list[dict[str, Any]], plan: Plan) -> list[str]:
    final_segments = [segment.to_dict() for segment in plan.audio_seg]
    draft_non_speech = [
        item
        for item in draft_segments
        if normalize_audio_type(item.get("audio_type", "")) != "speech"
        and str(item.get("segment_id", "")).strip()
    ]
    final_non_speech = [
        item
        for item in final_segments
        if normalize_audio_type(item.get("audio_type", "")) != "speech"
    ]
    final_speech = [
        item
        for item in final_segments
        if normalize_audio_type(item.get("audio_type", "")) == "speech"
    ]
    issues = []
    if draft_non_speech:
        covered = sum(
            1
            for draft in draft_non_speech
            if any(_same_or_covering_non_speech_event(draft, final) for final in final_non_speech)
        )
        coverage = covered / len(draft_non_speech)
        if coverage < 0.75:
            issues.append(
                f"non-speech event retention {covered}/"
                f"{len(draft_non_speech)} ({coverage:.0%})"
            )
    if len(final_speech) > 1:
        starts = {round(float(item.get("start_time", 0.0)), 3) for item in final_speech}
        if len(starts) == 1:
            issues.append(f"all {len(final_speech)} speech segments share one start time")
    draft_end = max((float(item.get("end_time", 0.0)) for item in draft_segments), default=0.0)
    final_end = max((float(item.get("end_time", 0.0)) for item in final_segments), default=0.0)
    if draft_end >= 10.0 and final_end < draft_end * 0.6:
        issues.append(f"timeline collapsed from {draft_end:.2f}s to {final_end:.2f}s")
    return issues


def _event_text(value: Any) -> str:
    """Normalize an event label/description for conservative identity matching."""

    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", str(value or "").lower()).strip()


def _same_or_covering_non_speech_event(
    draft: dict[str, Any], final: dict[str, Any], tolerance: float = 0.05
) -> bool:
    """Return true when a refined event faithfully represents a draft event.

    Timestamp refinement is allowed to rename segment IDs and to turn several
    identical ambience spans into one covering span.  Those transformations
    must not be mistaken for event loss, otherwise the guard restores the draft
    copy and generates the same sound twice.
    """

    draft_id = str(draft.get("segment_id", "")).strip()
    final_id = str(final.get("segment_id", "")).strip()
    if draft_id and draft_id == final_id:
        return True
    if normalize_audio_type(draft.get("audio_type", "")) != normalize_audio_type(
        final.get("audio_type", "")
    ):
        return False
    if _event_text(draft.get("object")) != _event_text(final.get("object")):
        return False
    if _event_text(draft.get("description")) != _event_text(final.get("description")):
        return False
    try:
        draft_start = float(draft.get("start_time", 0.0))
        draft_end = float(draft.get("end_time", draft_start))
        final_start = float(final.get("start_time", 0.0))
        final_end = float(final.get("end_time", final_start))
    except (TypeError, ValueError):
        return False
    return final_start <= draft_start + tolerance and final_end >= draft_end - tolerance


def _dedupe_exact_non_speech(
    segments: list[dict[str, Any]], draft_ids: set[str]
) -> list[dict[str, Any]]:
    """Remove exact refined/restored duplicates, preferring upstream IDs."""

    result: list[dict[str, Any]] = []
    key_to_index: dict[tuple[Any, ...], int] = {}
    for item in segments:
        if normalize_audio_type(item.get("audio_type", "")) == "speech":
            result.append(item)
            continue
        try:
            start = round(float(item.get("start_time", 0.0)), 2)
            end = round(float(item.get("end_time", start)), 2)
        except (TypeError, ValueError):
            result.append(item)
            continue
        key = (
            normalize_audio_type(item.get("audio_type", "")),
            _event_text(item.get("object")),
            _event_text(item.get("description")),
            start,
            end,
        )
        existing_index = key_to_index.get(key)
        if existing_index is None:
            key_to_index[key] = len(result)
            result.append(item)
            continue
        existing_id = str(result[existing_index].get("segment_id", "")).strip()
        candidate_id = str(item.get("segment_id", "")).strip()
        if existing_id not in draft_ids and candidate_id in draft_ids:
            result[existing_index] = item
    return result


def _restore_missing_non_speech(
    draft_segments: list[dict[str, Any]], plan: Plan
) -> tuple[Plan, int]:
    """Restore atomic non-speech events that the refinement LLM summarized away.

    The timestamp model remains authoritative for the refined speech timeline
    and for every event it returned. Missing non-speech events are copied from
    the complete upstream Audio-Oscar event plan, where they already have an
    executable start/end interval. This preserves event inventory without
    reverting to a draft whose dialogue may still be stacked at time zero.
    """

    draft_ids = {
        str(item.get("segment_id", "")).strip()
        for item in draft_segments
        if str(item.get("segment_id", "")).strip()
    }
    final_segments = _dedupe_exact_non_speech(
        [segment.to_dict() for segment in plan.audio_seg], draft_ids
    )
    final_non_speech = [
        item
        for item in final_segments
        if normalize_audio_type(item.get("audio_type", "")) != "speech"
    ]
    missing = [
        dict(item)
        for item in draft_segments
        if normalize_audio_type(item.get("audio_type", "")) != "speech"
        and str(item.get("segment_id", "")).strip()
        and not any(
            _same_or_covering_non_speech_event(item, final) for final in final_non_speech
        )
    ]
    merged = _dedupe_exact_non_speech(final_segments + missing, draft_ids)
    merged.sort(
        key=lambda item: (
            float(item.get("start_time", 0.0)),
            float(item.get("end_time", 0.0)),
            str(item.get("segment_id", "")),
        )
    )
    return Plan.from_json({"audio_seg": merged}), len(missing)


def _plan_count(path: Path, key: str) -> int:
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload.get(key, []))


def _write_batch_state(output_root: Path, completed, failures, active: str = "") -> None:
    payload = {"active_sample": active, "completed": completed, "failures": failures}
    tmp = output_root / "batch_state.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output_root / "batch_state.json")


SAMPLE_LOCK_HEARTBEAT_SECONDS = 30.0
SAMPLE_LOCK_STALE_SECONDS = float(
    os.environ.get("AUDIO_OSCAR_SAMPLE_LOCK_STALE_SECONDS", "7200")
)


class _SampleLock:
    """Atomic-directory lock with a heartbeat for a cross-host shared FS."""

    def __init__(self, lock_dir: Path, sample_id: str, token: str):
        self.lock_dir = lock_dir
        self.sample_id = sample_id
        self.token = token
        self.owner_path = lock_dir / "owner.json"
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name=f"sample-lock-{sample_id}",
            daemon=True,
        )
        self._write_owner()
        self._thread.start()

    def _owner_payload(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "token": self.token,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "heartbeat_at": time.time(),
        }

    def _still_owner(self) -> bool:
        try:
            payload = json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        return payload.get("token") == self.token

    def _write_owner(self) -> None:
        temporary = self.lock_dir / f"owner.{self.token}.tmp"
        temporary.write_text(
            json.dumps(self._owner_payload(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.owner_path)
        os.utime(self.lock_dir, None)

    def _heartbeat(self) -> None:
        while not self._stop.wait(SAMPLE_LOCK_HEARTBEAT_SECONDS):
            if not self._still_owner():
                return
            try:
                self._write_owner()
            except OSError:
                return

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        # If a stale-lock recovery replaced this directory, never delete the
        # replacement owner's lock.
        if not self._still_owner():
            return
        try:
            self.owner_path.unlink(missing_ok=True)
            self.lock_dir.rmdir()
        except OSError:
            pass


def _try_acquire_sample_lock(output_root: Path, sample_id: str):
    """Acquire an atomic, cross-host lock for one shared sample directory.

    CPFS does not propagate POSIX ``flock`` between the DLC hosts, while an
    atomic directory creation is shared correctly.  A heartbeat makes crashed
    owners distinguishable from long-running generation requests.
    """

    lock_root = output_root / ".sample_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_dir = lock_root / f"{sample_id}.lockdir"
    for _ in range(2):
        try:
            lock_dir.mkdir()
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= SAMPLE_LOCK_STALE_SECONDS:
                return None
            stale_dir = lock_root / f"{lock_dir.name}.stale.{uuid.uuid4().hex}"
            try:
                lock_dir.rename(stale_dir)
            except (FileNotFoundError, FileExistsError, OSError):
                return None
            shutil.rmtree(stale_dir, ignore_errors=True)
            continue
        token = uuid.uuid4().hex
        try:
            return _SampleLock(lock_dir, sample_id, token)
        except Exception:
            shutil.rmtree(lock_dir, ignore_errors=True)
            raise
    return None


def _release_sample_lock(handle) -> None:
    if handle is not None:
        handle.release()


def configure_clients() -> None:
    import executor.speech_generation as speech_generation

    os.environ["AUDIO_OSCAR_DEMUCS_HOST"] = DEMUCS_HOST
    os.environ["AUDIO_OSCAR_DEMUCS_PORT"] = DEMUCS_PORT

    speech_generation.MODEL_CLASS_MAP["qwen3ttsbase"] = partial(
        Qwen3TTSBase,
        base_url=QWEN_TTS_URL,
        model=BASE_TTS_PATH,
        timeout=1200.0,
    )
    speech_generation.MODEL_CLASS_MAP["cosyvoice3"] = partial(
        CosyVoice3,
        base_url=COSYVOICE_URL,
        timeout=1200.0,
    )
    speech_generation.MODEL_CLASS_MAP["voxcpm2"] = partial(
        VoxCPM2,
        base_url=VOXCPM_URL,
        timeout=1200.0,
    )
    # A single TTS server may route several utterances to the same GPU-backed
    # model.  Keep this configurable so the official per-utterance model route
    # is preserved without issuing unsafe concurrent requests to that model.
    speech_generation.SpeechGenerationExecutor = partial(
        VerifiedSpeechGenerationExecutor,
        max_workers=SPEECH_GENERATION_WORKERS,
        asr_base_url=OMNI_URL,
        allow_qwen_fallback=not PAPER_EXPERIMENT_MODE,
    )
    pipeline_module.FinalAudioAuditor = partial(FinalAudioAuditor, base_url=OMNI_URL)


def validate_runtime_dependencies() -> None:
    """Fail before generation if the final mixer/DSP stack is incomplete."""

    from executor.dsp import apply_reverb  # noqa: F401
    from executor.mixer import AudioMixerExecutor  # noqa: F401


def _read_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if frames <= 0 or rate <= 0:
        raise ValueError(f"invalid WAV frames/rate: {frames}/{rate}")
    return frames / rate


def _record_duration_contract(record: dict[str, Any]) -> dict[str, float] | None:
    keys = {
        "min_duration_seconds",
        "max_duration_seconds",
        "target_duration_seconds",
    }
    if not keys.intersection(record):
        return None
    minimum = float(record.get("min_duration_seconds", 20.0))
    maximum = float(record.get("max_duration_seconds", 60.0))
    target = float(record.get("target_duration_seconds", 36.0))
    if minimum < 20.0:
        raise ValueError(f"min_duration_seconds must be >=20, got {minimum}")
    if maximum > 60.0 or maximum <= minimum:
        raise ValueError(f"invalid max_duration_seconds {maximum} for minimum {minimum}")
    if not minimum <= target <= maximum:
        raise ValueError(f"target_duration_seconds {target} is outside [{minimum}, {maximum}]")
    return {
        "min_duration_seconds": minimum,
        "max_duration_seconds": maximum,
        "target_duration_seconds": target,
    }


def run_one(
    record: dict[str, Any], output_root: Path,
    runtime_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(record)
    sample_id = str(record.get("id") or "sample")
    text = str(record["prompt"]).strip()
    computed_prompt_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    declared_prompt_sha256 = record.get("prompt_sha256")
    if declared_prompt_sha256 and declared_prompt_sha256 != computed_prompt_sha256:
        history = list(record.get("prompt_sha256_history") or [])
        if declared_prompt_sha256 not in history:
            history.append(declared_prompt_sha256)
        record["prompt_sha256_history"] = history
    record["prompt_sha256"] = computed_prompt_sha256
    duration_contract = _record_duration_contract(record)
    sample_root = output_root / "samples" / sample_id
    stages = sample_root / "stages"
    voices = sample_root / "voice"
    audio = sample_root / "audio"
    sample_root.mkdir(parents=True, exist_ok=True)
    (sample_root / "input.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    llm = OpenAILLM(
        model=PLANNER_MODEL,
        temperature=PLANNER_TEMPERATURE,
        timeout=PLANNER_TIMEOUT,
        extra_params={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": PLANNER_ENABLE_THINKING}
            }
        },
        api_key="EMPTY",
        base_url=PLANNER_URL,
    )
    options = PipelineOptions(
        output_root=stages,
        voice_dir=voices,
        speech_output_dir=audio,
        enable_script_intake=True,
        script_intake_mode="script",
        speech_fa_mode=SPEECH_FA_MODE,
        tta_config_path=TTA_CONFIG_PATH,
        music_config_path=MUSIC_CONFIG_PATH,
        song_config_path=SONG_CONFIG_PATH,
        audio_timeout=2400.0,
        tta_generation_workers=TTA_GENERATION_WORKERS,
        audio_critic_model=OMNI_MODEL,
        audio_critic_threshold=0.7,
        audio_critic_max_retries=2,
        audio_critic_workers=AUDIO_CRITIC_WORKERS,
        music_timeout=2400.0,
        song_timeout=3600.0,
        run_final_edit=RUN_FINAL_EDIT,
        strict=False,
    )
    tts_config = TTSConfig.from_file(REPO / "tts/tts_config.json")
    if PAPER_EXPERIMENT_MODE:
        # Match the paper's ASG-Bench experimental TTS menu. Qwen3-TTS remains
        # the voice-design model, but it is not offered as an utterance TTS or
        # last-resort fallback in this mode.
        tts_config.models = [
            model for model in tts_config.models
            if str(model.name).lower() in {"cosyvoice3", "voxcpm2"}
        ]
        if {str(model.name).lower() for model in tts_config.models} != {
            "cosyvoice3", "voxcpm2"
        }:
            raise RuntimeError("paper experiment mode requires CosyVoice3 and VoxCPM2")
    artifacts = FullASGPipeline(
        llm=llm,
        tts_config=tts_config,
        options=options,
        duration_contract=duration_contract,
    ).run({"text": text, "source": "path3_qwen35_prompt", "source_id": sample_id})

    expected_speech = _plan_count(stages / "04_speech_generation_plan.json", "speech_segments")
    expected_tta = _plan_count(stages / "09_tta_generation_plan.json", "tta_segments")
    expected_music = _plan_count(stages / "09_music_generation_plan.json", "music_segments")
    expected_song = _plan_count(stages / "09_song_generation_plan.json", "song_segments")
    expected_non_speech = expected_tta + expected_music + expected_song
    if len(artifacts.speech_results) < expected_speech:
        raise RuntimeError(f"incomplete speech output: {len(artifacts.speech_results)}/{expected_speech}")
    if len(artifacts.audio_results) < expected_non_speech:
        raise RuntimeError(f"incomplete non-speech output: {len(artifacts.audio_results)}/{expected_non_speech}")
    mixed = audio / "mixed_audio.wav"
    final_audit = stages / "13_final_edit_log.json"
    if not mixed.is_file() or mixed.stat().st_size == 0:
        raise RuntimeError("final mixed audio is missing or empty")
    final_mix_review = False
    final_mix_remixed = False
    final_audio = mixed
    if options.run_final_edit:
        if not final_audit.is_file():
            raise RuntimeError("final mix audit did not produce its log")
        final_audit_payload = json.loads(final_audit.read_text(encoding="utf-8"))
        if final_audit_payload.get("status") != "success":
            raise RuntimeError("final mix audit did not finish successfully")
        polished = audio / "mixed_audio_polished.wav"
        final_mix_review = True
        final_mix_remixed = bool(final_audit_payload.get("remixed"))
        if final_mix_remixed and (not polished.is_file() or polished.stat().st_size == 0):
            raise RuntimeError("final mix audit reports a remix but polished audio is missing or empty")
        # A previous failed attempt may have left a polished WAV behind. Only use
        # it when this attempt's final-edit log explicitly reports a remix.
        final_audio = polished if final_mix_remixed else mixed
    final_duration_seconds = _read_wav_duration(final_audio)
    if duration_contract:
        minimum = duration_contract["min_duration_seconds"]
        maximum = duration_contract["max_duration_seconds"]
        if final_duration_seconds < minimum or final_duration_seconds > maximum:
            raise RuntimeError(
                "final audio violates duration contract: "
                f"{final_duration_seconds:.3f}s not in [{minimum:.3f}, {maximum:.3f}]"
            )

    summary = {
        "sample_id": sample_id,
        "prompt": text,
        "paths": artifacts.paths,
        "speech_segments": len(artifacts.speech_results),
        "planned_speech_segments": expected_speech,
        "non_speech_segments": len(artifacts.audio_results),
        "planned_non_speech_segments": expected_non_speech,
        "final_mix_review": final_mix_review,
        "final_mix_remixed": final_mix_remixed,
        "final_audio": str(final_audio),
        "final_duration_seconds": round(final_duration_seconds, 6),
        "duration_contract": duration_contract,
        "mix_entries": len(artifacts.mix_results),
        "planner": {
            "base_url": PLANNER_URL,
            "model": PLANNER_MODEL,
            "temperature": PLANNER_TEMPERATURE,
            "enable_thinking": PLANNER_ENABLE_THINKING,
        },
        "paper_experiment_mode": PAPER_EXPERIMENT_MODE,
        "audio_oscar": runtime_provenance or audio_oscar_provenance(),
        "pipeline_options": {
            "tta_generation_workers": options.tta_generation_workers,
            "audio_critic_threshold": options.audio_critic_threshold,
            "audio_critic_max_retries": options.audio_critic_max_retries,
            "audio_critic_workers": options.audio_critic_workers,
            "run_final_edit": options.run_final_edit,
            "strict": options.strict,
        },
        "generator_configs": {
            "tts_models": [str(model.name) for model in tts_config.models],
            "tta_config_path": str(TTA_CONFIG_PATH),
            "tta_config": json.loads(TTA_CONFIG_PATH.read_text(encoding="utf-8")),
            "music_config_path": str(MUSIC_CONFIG_PATH),
            "song_config_path": str(SONG_CONFIG_PATH),
        },
        "speech_fa_mode": SPEECH_FA_MODE,
    }
    (sample_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (sample_root / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--max-sample-attempts", type=int, default=3)
    parser.add_argument("--no-batch-state", action="store_true")
    parser.add_argument("--summary-name", default="")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime_provenance = audio_oscar_provenance()
    configure_clients()
    validate_runtime_dependencies()
    prompts = load_prompts(args.prompts)
    if args.sample_id:
        wanted = set(args.sample_id)
        prompts = [item for item in prompts if str(item.get("id")) in wanted]
    completed = []
    failures = []
    for record in prompts:
        sample_id = str(record.get("id") or "sample")
        marker = args.output_root / "samples" / sample_id / "COMPLETE"
        if args.skip_complete and marker.is_file():
            print(f"SKIP complete sample: {sample_id}", flush=True)
            continue
        sample_lock = _try_acquire_sample_lock(args.output_root, sample_id)
        if sample_lock is None:
            print(f"SKIP locked sample: {sample_id}", flush=True)
            continue
        try:
            # A different worker may have completed the sample between the
            # optimistic marker check and this lock acquisition.
            if args.skip_complete and marker.is_file():
                print(f"SKIP complete-after-lock sample: {sample_id}", flush=True)
                continue
            if not args.no_batch_state:
                _write_batch_state(args.output_root, completed, failures, sample_id)
            last_error = None
            for attempt in range(1, max(1, args.max_sample_attempts) + 1):
                try:
                    print(
                        f"RUN sample={sample_id} attempt={attempt}/{max(1, args.max_sample_attempts)}",
                        flush=True,
                    )
                    completed.append(
                        _run_one_waiting_for_services(
                            record, args.output_root,
                            runtime_provenance=runtime_provenance,
                        )
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    traceback.print_exc()
                    if attempt < max(1, args.max_sample_attempts):
                        print(
                            f"RETRY sample={sample_id} after {type(exc).__name__}: {exc}",
                            flush=True,
                        )
            if last_error is not None:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "error": str(last_error),
                        "attempts": max(1, args.max_sample_attempts),
                    }
                )
            if not args.no_batch_state:
                _write_batch_state(args.output_root, completed, failures)
        finally:
            _release_sample_lock(sample_lock)
    result = {"completed": completed, "failures": failures}
    if args.summary_name:
        summary_name = Path(args.summary_name).name
    elif args.no_batch_state and args.sample_id:
        summary_name = "retry_summary_" + "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", item) for item in args.sample_id) + ".json"
    else:
        summary_name = "batch_summary.json"
    (args.output_root / summary_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
