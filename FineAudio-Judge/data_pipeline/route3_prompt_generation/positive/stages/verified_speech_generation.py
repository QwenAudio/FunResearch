from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock, local
from typing import Any

from openai import OpenAI

from executor.speech_generation import (
    SpeechGenerationExecutor,
    SpeechGenerationResult,
    _is_voxcpm,
)
from utils.io_utils import get_wav_duration


ASR_PROMPT = (
    "Transcribe only the intelligible spoken words in this audio. "
    "Preserve the spoken language. Do not describe sound quality, speaker "
    "traits, pauses, or non-speech sounds. Return only the transcript."
)


_NUMBER_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_NUMBER_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


def _integer_words(value: int) -> list[str]:
    """Return a comparison-only English expansion for a non-negative integer."""
    if value < 20:
        return [_NUMBER_ONES[value]]
    if value < 100:
        tens, ones = divmod(value, 10)
        return [_NUMBER_TENS[tens]] + (_integer_words(ones) if ones else [])
    if value < 1_000:
        hundreds, remainder = divmod(value, 100)
        return _integer_words(hundreds) + ["hundred"] + (
            _integer_words(remainder) if remainder else []
        )
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1_000)
        return _integer_words(thousands) + ["thousand"] + (
            _integer_words(remainder) if remainder else []
        )
    return [str(value)]


def _tokens(text: str) -> list[str]:
    # VoxCPM may realize the sentence-final particles 啊/呀 interchangeably.
    normalized = str(text).lower().replace("呀", "啊")
    raw_tokens = re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if token.isascii() and token.isdigit():
            # ASR commonly renders Arabic numerals as spoken English words
            # (for example, target "42" -> transcript "forty two").
            tokens.extend(_integer_words(int(token)))
        else:
            tokens.append(token)
    return tokens


def score_transcript(expected: str, transcript: str) -> dict[str, Any]:
    expected_tokens = _tokens(expected)
    transcript_tokens = _tokens(transcript)
    matcher = SequenceMatcher(a=expected_tokens, b=transcript_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    recall = matched / len(expected_tokens) if expected_tokens else 0.0
    precision = matched / len(transcript_tokens) if transcript_tokens else 0.0
    max_extra = max(1, round(len(expected_tokens) * 0.25))
    extras = max(0, len(transcript_tokens) - matched)
    passed = bool(
        expected_tokens
        and transcript_tokens
        and recall >= 0.85
        and precision >= 0.75
        and extras <= max_extra
    )
    return {
        "expected_tokens": len(expected_tokens),
        "transcript_tokens": len(transcript_tokens),
        "matched_tokens": matched,
        "target_recall": round(recall, 4),
        "transcript_precision": round(precision, 4),
        "extra_tokens": extras,
        "max_extra_tokens": max_extra,
        "passed": passed,
    }


class VerifiedSpeechGenerationExecutor(SpeechGenerationExecutor):
    """VoxCPM mode separation plus per-stem ASR content verification.

    The upstream executor combines VoxCPM's prompt-transcript cloning mode with
    a leading style instruction. VoxCPM treats those as separate operating
    modes, so this runtime layer tries valid modes independently and preserves
    every attempt for audit. Non-Vox routes remain unchanged.
    """

    def __init__(
        self,
        *args: Any,
        asr_base_url: str,
        asr_model: str | None = None,
        voxcpm_retries: int = 2,
        allow_qwen_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        normalized_asr_url = asr_base_url.rstrip("/")
        if not normalized_asr_url.endswith("/v1"):
            normalized_asr_url += "/v1"
        self.asr_base_url = normalized_asr_url
        self.asr_model = asr_model
        self.voxcpm_retries = max(1, int(voxcpm_retries))
        self.allow_qwen_fallback = bool(allow_qwen_fallback)
        self._verification_local = local()
        self._audit_lock = Lock()

    def _asr_client_and_model(self) -> tuple[OpenAI, str]:
        client = getattr(self._verification_local, "asr_client", None)
        model = getattr(self._verification_local, "asr_model", None)
        if client is None:
            client = OpenAI(base_url=self.asr_base_url, api_key="EMPTY", timeout=300)
            model = self.asr_model or client.models.list().data[0].id
            self._verification_local.asr_client = client
            self._verification_local.asr_model = model
        return client, str(model)

    def _transcribe(self, path: Path) -> str:
        client, model = self._asr_client_and_model()
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded,
                                "format": path.suffix.lstrip("."),
                            },
                        },
                        {"type": "text", "text": ASR_PROMPT},
                    ],
                }
            ],
            temperature=0.0,
            max_completion_tokens=256,
        )
        return str(response.choices[0].message.content or "").strip()

    def _append_audit(self, row: dict[str, Any]) -> None:
        audit_path = self.output_dir / "tts_content_audit.jsonl"
        with self._audit_lock:
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _generate_for_item(self, item, index: int, total: int):
        if not _is_voxcpm(item.model):
            return super()._generate_for_item(item, index, total)

        reference_audio = None
        if item.name:
            candidate = self.voice_dir / f"{item.name}.wav"
            if candidate.exists():
                reference_audio = str(candidate)
        prompt_text = None
        if self.role_profile:
            for role in self.role_profile.roles:
                if role.name == item.name:
                    prompt_text = role.text
                    break
        if not reference_audio:
            raise ValueError(f"VoxCPM2 requires reference audio for role {item.name!r}")

        attempts_dir = self.output_dir / "voxcpm_attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        attempts: list[tuple[str, str]] = [
            ("controllable_refined", "voxcpm2")
            for _ in range(self.voxcpm_retries)
        ]
        if prompt_text:
            attempts.append(("ultimate_plain", "voxcpm2"))
        if self.allow_qwen_fallback:
            attempts.append(("qwen3ttsbase_fallback", "qwen3ttsbase"))

        selected: Path | None = None
        selected_mode = ""
        last_score: dict[str, Any] | None = None
        for attempt_number, (mode, model_name) in enumerate(attempts, 1):
            parameters = dict(item.parameters)
            parameters["reference_audio"] = reference_audio
            if mode == "controllable_refined":
                parameters["text"] = item.text
                parameters.pop("prompt_wav_path", None)
                parameters.pop("prompt_text", None)
            elif mode == "ultimate_plain":
                parameters.pop("refined_text", None)
                parameters["text"] = item.text
                parameters["prompt_wav_path"] = reference_audio
                parameters["prompt_text"] = prompt_text
            else:
                parameters = {
                    "text": item.text,
                    "reference_audio": reference_audio,
                    "language": "Auto",
                    "x_vector_only_mode": True,
                }

            attempt_path = attempts_dir / (
                f"{index:04d}_{item.segment_id}_{item.name}_"
                f"attempt{attempt_number}_{mode}.wav"
            )
            model_instance = self._get_model_instance(model_name)
            result_path = Path(
                model_instance.text_to_speech(
                    output_file=str(attempt_path), parameters=parameters
                )
            )
            transcript = self._transcribe(result_path)
            score = score_transcript(item.text, transcript)
            last_score = score
            audit = {
                "segment_id": item.segment_id,
                "name": item.name,
                "planned_model": item.model,
                "attempt": attempt_number,
                "mode": mode,
                "actual_model": model_name,
                "target_text": item.text,
                "asr_transcript": transcript,
                "score": score,
                "audio_path": str(result_path.resolve()),
                "duration": get_wav_duration(result_path),
            }
            self._append_audit(audit)
            print(
                f"  - VoxCPM ASR gate attempt={attempt_number} mode={mode} "
                f"pass={score['passed']} transcript={transcript!r}",
                flush=True,
            )
            if score["passed"]:
                selected = result_path
                selected_mode = mode
                break

        if selected is None:
            raise RuntimeError(
                f"TTS content mismatch after {len(attempts)} attempts for "
                f"{item.segment_id}: {last_score}"
            )

        output_file = self.output_dir / f"{index:04d}_{item.name}.wav"
        shutil.copy2(selected, output_file)
        duration = get_wav_duration(output_file)
        result = SpeechGenerationResult(
            segment_id=item.segment_id,
            name=item.name,
            text=item.text,
            audio_path=str(output_file.resolve()),
            duration=duration,
        )
        self._append_audit(
            {
                "segment_id": item.segment_id,
                "status": "selected",
                "selected_mode": selected_mode,
                "result": asdict(result),
            }
        )
        return result
