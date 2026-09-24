#!/usr/bin/env python3
"""Minimal ASR editing loop for one segmented recording.

The agent has four model-visible steps:

    summary -> scan -> check -> edit

The program itself only sends requests, cuts the exact segment audio, validates
the returned JSON, and writes the model's edited segment back.  It contains no
phonetic thresholds, numeric rules, confidence rules, or second judge.
References are never read here.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from agentic_ger.config import PromptPack, load_prompt_pack
from agentic_ger.utils import REPO_ROOT, service_headers


ROOT = REPO_ROOT
DEFAULT_PROMPT_PACK = ROOT / "prompts/zh"
THINKING_MAX_TOKENS = 32_768
SAMPLING_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
}
LLM_REQUEST_TIMEOUT_SECONDS = 7_200
LLM_CONNECT_TIMEOUT_SECONDS = 60
LLM_ATTEMPTS = 3
SCHEMA_REPAIR_RAW_CHARACTERS = 16_384


def text_of(segment: dict[str, Any]) -> str:
    return str(
        segment.get("text_working")
        or segment.get("text_final")
        or segment.get("text_original")
        or segment.get("text")
        or ""
    )


def original_text(segment: dict[str, Any]) -> str:
    return str(
        segment.get("text_original")
        or segment.get("text_working")
        or segment.get("text_final")
        or segment.get("text")
        or ""
    )


def transcript_lines(segments: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{int(row['id'])}] {text_of(row)}" for row in segments)


@dataclass
class Trace:
    path: Path
    sequence: int = 0

    def write(self, event: str, **value: Any) -> None:
        self.sequence += 1
        row = {"seq": self.sequence, "time": time.time(), "event": event, **value}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Counters:
    llm_requests: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_responses_with_reasoning: int = 0
    llm_reasoning_characters: int = 0
    asr_requests: int = 0
    loops: int = 0
    inspections: int = 0
    accepted: int = 0
    kept: int = 0


class SchemaViolation(ValueError):
    """A model result does not satisfy the requested schema subset."""


class RequestDeadlineExceeded(TimeoutError):
    """An HTTP response exceeded its absolute wall-clock deadline."""


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the JSON Schema keywords used by the versioned prompt packs."""
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaViolation(f"{path} is not in enum")
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise SchemaViolation(f"{path} must be an object")
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                raise SchemaViolation(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise SchemaViolation(
                    f"{path} has extra properties: {sorted(extra)}"
                )
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], f"{path}.{key}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise SchemaViolation(f"{path} must be an array")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise SchemaViolation(f"{path} exceeds maxItems")
        for index, item in enumerate(value):
            validate_schema(item, schema.get("items") or {}, f"{path}[{index}]")
        return
    if expected == "string":
        if not isinstance(value, str):
            raise SchemaViolation(f"{path} must be a string")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise SchemaViolation(f"{path} exceeds maxLength")
        return
    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaViolation(f"{path} must be an integer")
        return
    if expected is not None:
        raise ValueError(f"unsupported schema type at {path}: {expected}")


def normalize_schema_root(
    value: Any, schema: dict[str, Any]
) -> tuple[Any, str]:
    """Normalize the one root-shape error observed in production-v3."""
    if (
        schema.get("type") == "object"
        and isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
    ):
        return value[0], "unwrap_single_object_array"
    return value, ""


def load_schema_json(
    content: str, schema: dict[str, Any]
) -> tuple[Any, str]:
    """Load JSON, repairing one uniquely recoverable missing string terminator."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError as original_error:
        candidates: list[Any] = []
        for index, character in enumerate(content):
            if character not in "}]":
                continue
            candidate = content[:index] + '"' + content[index:]
            try:
                candidate_value = json.loads(candidate)
                candidate_value, _ = normalize_schema_root(candidate_value, schema)
                validate_schema(candidate_value, schema)
            except (json.JSONDecodeError, SchemaViolation):
                continue
            candidates.append(candidate_value)
        if len(candidates) != 1:
            raise original_error
        return candidates[0], "insert_missing_string_terminator"
    value, normalization = normalize_schema_root(value, schema)
    return value, normalization


def is_schema_echo(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether an answer is the requested schema rather than its value."""
    if not isinstance(value, dict):
        return False
    if value == schema:
        return True
    schema_markers = {"type", "properties", "required"}
    required = set(schema.get("required") or [])
    return schema_markers.issubset(value) and not required.issubset(value)


def post_json_with_deadline(
    url: str,
    payload: dict[str, Any],
    deadline_seconds: float,
) -> dict[str, Any]:
    """POST JSON while enforcing a deadline despite response keepalive bytes."""
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    started = time.monotonic()
    response = requests.post(
        url,
        json=payload,
        stream=True,
        headers=service_headers("correction"),
        timeout=(
            min(LLM_CONNECT_TIMEOUT_SECONDS, deadline_seconds),
            deadline_seconds,
        ),
    )
    body = bytearray()
    try:
        # The cloud adapter writes a single blank byte every 30 seconds while
        # waiting.  Read byte-wise only for responses without Content-Length
        # so those keepalives cannot reset the socket timeout forever.
        chunk_size = 65_536 if response.headers.get("content-length") else 1
        for chunk in response.iter_content(chunk_size=chunk_size):
            if time.monotonic() - started > deadline_seconds:
                raise RequestDeadlineExceeded(
                    f"LLM request exceeded {deadline_seconds:g}s total deadline"
                )
            body.extend(chunk)
        if time.monotonic() - started > deadline_seconds:
            raise RequestDeadlineExceeded(
                f"LLM request exceeded {deadline_seconds:g}s total deadline"
            )
        try:
            value = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"LLM HTTP {response.status_code} returned invalid JSON: {error}"
            ) from error
        if response.status_code >= 400:
            detail = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            raise RuntimeError(f"LLM HTTP {response.status_code}: {detail}")
        if not isinstance(value, dict):
            raise RuntimeError("LLM response body must be a JSON object")
        return value
    finally:
        response.close()


def decode_chat_completion(body: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(body, dict):
        raise RuntimeError("LLM response body must be a JSON object")
    if body.get("error") is not None:
        detail = json.dumps(body["error"], ensure_ascii=False, separators=(",", ":"))
        raise RuntimeError(f"LLM upstream error: {detail}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response is missing a non-empty choices array")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("LLM response choices[0] must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response choices[0].message must be an object")
    return choice, message


def schema_repair_messages(
    original_messages: list[dict[str, Any]],
    raw_content: str,
    error: Exception,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    excerpt = raw_content[:SCHEMA_REPAIR_RAW_CHARACTERS]
    if len(raw_content) > SCHEMA_REPAIR_RAW_CHARACTERS:
        excerpt += "\n[previous output truncated]"
    messages = [dict(item) for item in original_messages]
    if excerpt:
        messages.append({"role": "assistant", "content": excerpt})
    messages.append(
        {
            "role": "user",
            "content": (
                "Regenerate the JSON object. Return exactly one JSON object, "
                "not an array. Its top-level keys must be exactly: "
                f"{', '.join((schema.get('properties') or {}).keys())}. "
                "Do not return, quote, or describe the JSON Schema itself. "
                "The previous answer violated "
                f"the requested schema: {error}. Correct only the invalid "
                "structure or size, obey every schema constraint, and return "
                "JSON only."
            ),
        }
    )
    return messages


@dataclass
class LlmClient:
    url: str
    model: str
    trace: Trace
    counters: Counters
    disable_thinking: bool
    thinking_max_tokens: int
    reasoning_effort: str | None
    sampling: dict[str, float | int]
    request_timeout_seconds: int
    json_format_prompt: str

    def call(
        self,
        label: str,
        prompt: str,
        data: Any,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        if not self.disable_thinking:
            max_tokens = max(max_tokens, self.thinking_max_tokens)
        system_prompt = prompt
        if self.json_format_prompt:
            system_prompt = f"{prompt} {self.json_format_prompt}"
        original_messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": original_messages,
            "seed": 0,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": label[-48:], "strict": True, "schema": schema},
            },
        }
        payload["chat_template_kwargs"] = {
            "enable_thinking": not self.disable_thinking,
            "preserve_thinking": True,
        }
        payload.update(self.sampling)
        if not self.disable_thinking:
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
                payload["chat_template_kwargs"]["reasoning_effort"] = (
                    self.reasoning_effort
                )
        tokenize_url = self.url.split("/v1/chat/completions", 1)[0] + "/tokenize"
        def check_context(
            candidate_messages: list[dict[str, Any]],
            attempt: int,
            repair: bool,
        ) -> tuple[bool, int, int]:
            token_response = requests.post(
                tokenize_url,
                headers=service_headers("correction"),
                json={
                    "model": self.model,
                    "messages": candidate_messages,
                    "chat_template_kwargs": payload["chat_template_kwargs"],
                },
                timeout=60,
            )
            token_response.raise_for_status()
            token_info = token_response.json()
            prompt_tokens = int(token_info["count"])
            context_limit = int(token_info["max_model_len"])
            fits = prompt_tokens + max_tokens <= context_limit
            self.trace.write(
                "context.check",
                label=label,
                attempt=attempt,
                repair=repair,
                prompt_tokens=prompt_tokens,
                output_tokens=max_tokens,
                context_limit=context_limit,
                fits=fits,
            )
            return fits, prompt_tokens, context_limit

        fits, prompt_tokens, context_limit = check_context(
            original_messages, 1, False
        )
        if not fits:
            raise RuntimeError(
                f"{label} needs at least {prompt_tokens + max_tokens} context tokens, "
                f"but the vLLM service exposes {context_limit}; restart it with a "
                "larger --max-model-len"
            )
        self.trace.write(
            "llm.request",
            label=label,
            prompt=prompt,
            data=data,
            thinking_enabled=not self.disable_thinking,
            reasoning_effort=(
                self.reasoning_effort if not self.disable_thinking else None
            ),
            max_tokens=max_tokens,
            sampling={
                key: payload[key]
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                )
                if key in payload
            },
        )
        last_error: Exception | None = None
        for attempt in range(1, LLM_ATTEMPTS + 1):
            response_details: dict[str, Any] = {}
            try:
                body = post_json_with_deadline(
                    self.url,
                    payload,
                    self.request_timeout_seconds,
                )
                choice, message = decode_chat_completion(body)
                reasoning = str(
                    message.get("reasoning")
                    or message.get("reasoning_content")
                    or ""
                )
                content = message.get("content")
                usage = body.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                response_details = {
                    "finish_reason": choice.get("finish_reason"),
                    "raw_content": content,
                    "reasoning": reasoning,
                    "usage": usage,
                }
                self.counters.llm_requests += 1
                self.counters.llm_prompt_tokens += int(
                    usage.get("prompt_tokens") or 0
                )
                self.counters.llm_completion_tokens += int(
                    usage.get("completion_tokens") or 0
                )
                if reasoning:
                    self.counters.llm_responses_with_reasoning += 1
                    self.counters.llm_reasoning_characters += len(reasoning)
                if not isinstance(content, str) or not content:
                    raise RuntimeError(
                        "LLM returned no final content "
                        f"(finish_reason={choice.get('finish_reason')}, "
                        f"reasoning_characters={len(reasoning)})"
                    )
                result, normalization = load_schema_json(content, schema)
                schema_echo = is_schema_echo(result, schema)
                response_details["schema_echo"] = schema_echo
                if normalization:
                    self.trace.write(
                        "llm.normalized",
                        label=label,
                        attempt=attempt,
                        action=normalization,
                        schema_echo=schema_echo,
                    )
                validate_schema(result, schema)
                self.trace.write(
                    "llm.result",
                    label=label,
                    result=result,
                    reasoning=reasoning,
                    usage=usage,
                )
                return result
            except Exception as error:  # noqa: BLE001
                last_error = error
                self.trace.write(
                    "llm.error",
                    label=label,
                    attempt=attempt,
                    error=str(error),
                    error_type=type(error).__name__,
                    **response_details,
                )
                if attempt < LLM_ATTEMPTS:
                    repair = isinstance(error, (json.JSONDecodeError, SchemaViolation))
                    raw_content = response_details.get("raw_content")
                    if repair and isinstance(raw_content, str):
                        if response_details.get("schema_echo"):
                            raw_content = ""
                        candidate_messages = schema_repair_messages(
                            original_messages, raw_content, error, schema
                        )
                        fits, prompt_tokens, context_limit = check_context(
                            candidate_messages, attempt + 1, True
                        )
                        raw_included = bool(raw_content)
                        if not fits:
                            candidate_messages = schema_repair_messages(
                                original_messages, "", error, schema
                            )
                            fits, prompt_tokens, context_limit = check_context(
                                candidate_messages, attempt + 1, True
                            )
                            raw_included = False
                        if not fits:
                            raise RuntimeError(
                                f"{label} repair needs at least "
                                f"{prompt_tokens + max_tokens} context tokens, "
                                f"but the service exposes {context_limit}"
                            ) from error
                        payload["messages"] = candidate_messages
                    else:
                        raw_included = False
                    self.trace.write(
                        "llm.retry",
                        label=label,
                        next_attempt=attempt + 1,
                        strategy="schema_feedback" if repair else "same_request",
                        raw_content_included=raw_included,
                    )
                    time.sleep(attempt)
        raise RuntimeError(f"{label} failed: {last_error}")


def summarize(
    client: LlmClient,
    segments: list[dict[str, Any]],
    prompt_pack: PromptPack,
) -> dict[str, Any]:
    return client.call(
        "initial_summary",
        prompt_pack.summary.prompt,
        {"full_transcript": transcript_lines(segments)},
        prompt_pack.summary.schema,
        prompt_pack.summary.max_tokens,
    )


def memory_view(memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in (
                "segment_id",
                "focus",
                "decision",
                "before_segment",
                "after_segment",
                "reason",
            )
            if key in row
        }
        for row in memory
    ]


def scan(
    client: LlmClient,
    prompt_pack: PromptPack,
    summary: dict[str, Any],
    memory: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    k: int,
    label: str,
) -> list[dict[str, Any]]:
    answer = client.call(
        label,
        prompt_pack.scan.prompt.format(k=k),
        {
            "summary": summary,
            "working_memory": memory_view(memory),
            "full_transcript": transcript_lines(segments),
        },
        prompt_pack.scan_schema(k),
        prompt_pack.scan.max_tokens,
    )
    by_id = {int(row["id"]): row for row in segments}
    result = []
    for suspect in answer["suspects"]:
        segment_id = int(suspect["segment_id"])
        focus = str(suspect["focus"])
        if segment_id in by_id and focus and focus in text_of(by_id[segment_id]):
            result.append(
                {
                    "segment_id": segment_id,
                    "focus": focus,
                    "reason": str(suspect["reason"]),
                }
            )
    return result


def cut_wav(source: Path, destination: Path, start: float, end: float) -> None:
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        start_frame = max(0, round(start * rate))
        end_frame = min(reader.getnframes(), round(end * rate))
        reader.setpos(start_frame)
        frames = reader.readframes(max(1, end_frame - start_frame))
        parameters = reader.getparams()
    frame_width = parameters.nchannels * parameters.sampwidth
    clip_frames = len(frames) // frame_width
    with wave.open(str(destination), "wb") as writer:
        # Some source files use a sentinel/incorrect nframes value in the WAV
        # header.  Carrying it through setparams() can overflow the 32-bit WAV
        # data-size field even when the extracted clip is only a few seconds.
        writer.setnchannels(parameters.nchannels)
        writer.setsampwidth(parameters.sampwidth)
        writer.setframerate(parameters.framerate)
        writer.setcomptype(parameters.comptype, parameters.compname)
        writer.setnframes(clip_frames)
        writer.writeframes(frames)


def parse_asr_text(body: dict[str, Any]) -> str:
    text = str(body["choices"][0]["message"]["content"])
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text.replace("</asr_text>", "").strip()


def transcribe(
    audio: Path,
    segment: dict[str, Any],
    asr_url: str,
    asr_model: str,
    temporary_dir: Path,
    trace: Trace,
    counters: Counters,
) -> str:
    segment_id = int(segment["id"])
    start, end = float(segment["start"]), float(segment["end"])
    clip = temporary_dir / f"segment_{segment_id:06d}.wav"
    cut_wav(audio, clip, start, end)
    encoded = base64.b64encode(clip.read_bytes()).decode("ascii")
    response = requests.post(
        asr_url,
        headers=service_headers("asr"),
        json={
            "model": asr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{encoded}"},
                        }
                    ],
                }
            ],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 512,
        },
        timeout=240,
    )
    response.raise_for_status()
    text = parse_asr_text(response.json())
    counters.asr_requests += 1
    trace.write(
        "asr.result", segment_id=segment_id, start=start, end=end, asr_text=text
    )
    return text


def check(
    client: LlmClient,
    prompt_pack: PromptPack,
    summary: dict[str, Any],
    memory: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    current_pair: dict[str, Any],
) -> dict[str, Any]:
    segment_id = int(current_pair["suspect"]["segment_id"])
    return client.call(
        f"check_segment_{segment_id}",
        prompt_pack.check.prompt,
        {
            "summary": summary,
            "working_memory": memory_view(memory),
            "full_transcript": transcript_lines(segments),
            "all_suspect_evidence_pairs": pairs,
            "current_pair_to_decide": current_pair,
        },
        prompt_pack.check.schema,
        prompt_pack.check.max_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--llm-url", default="http://127.0.0.1:8002/v1/chat/completions")
    parser.add_argument("--llm-model", default="qwen3.8-27b")
    parser.add_argument("--asr-url", default="http://127.0.0.1:7879/v1/chat/completions")
    parser.add_argument("--asr-model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--max-loops", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=48)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--thinking-max-tokens", type=int, default=THINKING_MAX_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        choices=("xhigh", "medium", "low"),
        default=None,
        help="optional model-native reasoning-effort control",
    )
    parser.add_argument(
        "--sampling-json",
        default="",
        help="model-specific sampling object; defaults to Qwen3.8 parameters",
    )
    parser.add_argument("--llm-timeout", type=int, default=LLM_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=DEFAULT_PROMPT_PACK,
        help="versioned directory containing prompts, schemas, and stage budgets",
    )
    parser.add_argument("--expected-prompt-pack-sha256", default="")
    parser.add_argument("--expected-agent-sha256", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sampling_json:
        sampling = json.loads(args.sampling_json)
        if not isinstance(sampling, dict) or set(sampling) - SAMPLING_KEYS:
            raise SystemExit("sampling-json contains unsupported fields")
    elif args.disable_thinking:
        sampling = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
    else:
        sampling = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
    prompt_pack = load_prompt_pack(args.prompt_pack)
    if (
        args.expected_prompt_pack_sha256
        and prompt_pack.sha256 != args.expected_prompt_pack_sha256
    ):
        raise SystemExit("prompt pack changed after the batch started")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.expected_agent_sha256 and source_hash != args.expected_agent_sha256:
        raise SystemExit("agent.py changed after the batch started")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    trace_path.unlink(missing_ok=True)
    trace = Trace(trace_path)
    counters = Counters()
    client = LlmClient(
        args.llm_url,
        args.llm_model,
        trace,
        counters,
        args.disable_thinking,
        args.thinking_max_tokens,
        args.reasoning_effort,
        sampling,
        args.llm_timeout,
        prompt_pack.json_format_prompt,
    )

    raw = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = copy.deepcopy(raw)
    for segment in baseline.get("segments") or []:
        text = original_text(segment)
        segment["text_original"] = text
        segment["text_working"] = text
        segment["text_final"] = text
    working = copy.deepcopy(baseline)
    segments = working.get("segments") or []
    (output_dir / "baseline_transcript.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    trace.write(
        "run.start",
        recording_id=working.get("recording_id"),
        baseline=str(args.baseline.resolve()),
        audio=str(args.audio.resolve()),
        agent_sha256=source_hash,
        thinking_enabled=not args.disable_thinking,
        thinking_max_tokens=args.thinking_max_tokens,
        reasoning_effort=(args.reasoning_effort if not args.disable_thinking else None),
        llm_timeout_seconds=args.llm_timeout,
        **prompt_pack.metadata(),
    )

    started = time.monotonic()
    summary = summarize(client, segments, prompt_pack)
    trace.write("summary", summary=summary)
    memory: list[dict[str, Any]] = []
    checked: set[tuple[int, str, str]] = set()
    stop_reason = "max_loops"
    with tempfile.TemporaryDirectory(
        prefix=".audio_tmp_", dir=output_dir
    ) as temporary:
        temporary_dir = Path(temporary)
        for loop_index in range(1, args.max_loops + 1):
            counters.loops += 1
            suspects = scan(
                client,
                prompt_pack,
                summary,
                memory,
                segments,
                args.k,
                f"scan_loop_{loop_index}",
            )
            by_id = {int(row["id"]): row for row in segments}
            unique_suspects = []
            seen_segments: set[int] = set()
            for suspect in suspects:
                segment_id = int(suspect["segment_id"])
                key = (segment_id, suspect["focus"], text_of(by_id[segment_id]))
                if key not in checked and segment_id not in seen_segments:
                    unique_suspects.append(suspect)
                    seen_segments.add(segment_id)
            trace.write(
                "scan.result", loop_index=loop_index, suspects=unique_suspects
            )
            if not unique_suspects:
                stop_reason = "no_new_suspects"
                break

            # First obtain all suspect/evidence pairs from the same transcript snapshot.
            pairs = []
            for suspect in unique_suspects:
                segment_id = int(suspect["segment_id"])
                segment = by_id[segment_id]
                before = text_of(segment)
                asr_text = transcribe(
                    args.audio.resolve(),
                    segment,
                    args.asr_url,
                    args.asr_model,
                    temporary_dir,
                    trace,
                    counters,
                )
                pairs.append(
                    {
                        "suspect": suspect,
                        "asr_text": asr_text,
                        "before_segment": before,
                    }
                )
                checked.add((segment_id, suspect["focus"], before))
            trace.write("evidence.ready", loop_index=loop_index, pairs=pairs)

            # Judge pairs one by one, but give every judgment the same full transcript.
            decisions = []
            for pair in pairs:
                counters.inspections += 1
                decision = check(
                    client,
                    prompt_pack,
                    summary,
                    memory,
                    segments,
                    pairs,
                    pair,
                )
                decisions.append((pair, decision))
                trace.write(
                    "check.result",
                    loop_index=loop_index,
                    pair=pair,
                    decision=decision,
                )

            # Commit only after every pair in this loop has been judged.
            for pair, decision in decisions:
                suspect = pair["suspect"]
                segment_id = int(suspect["segment_id"])
                before = pair["before_segment"]
                edited = str(decision["edited_segment"])
                if decision["decision"] != "edit" or not edited or edited == before:
                    counters.kept += 1
                    memory.append(
                        {
                            "segment_id": segment_id,
                            "focus": suspect["focus"],
                            "decision": "keep",
                            "reason": str(decision["reason"]),
                        }
                    )
                    trace.write(
                        "patch.decision",
                        segment_id=segment_id,
                        decision="keep",
                        check=decision,
                    )
                    continue

                segment = by_id[segment_id]
                segment["text_working"] = edited
                segment["text_final"] = edited
                patch = {
                    "segment_id": segment_id,
                    "before_segment": before,
                    "after_segment": edited,
                    "old_text": before,
                    "new_text": edited,
                    "focus": suspect["focus"],
                    "decision": "edit",
                    "asr_text": pair["asr_text"],
                    "reason": str(decision["reason"]),
                }
                memory.append(patch)
                counters.accepted += 1
                trace.write(
                    "patch.decision",
                    segment_id=segment_id,
                    decision="accept",
                    patch=patch,
                )
                if counters.accepted >= args.max_patches:
                    stop_reason = "max_patches"
                    break
            if counters.accepted >= args.max_patches:
                break

    (output_dir / "final_transcript.json").write_text(
        json.dumps(working, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "status": "complete",
        "recording_id": working.get("recording_id"),
        "agent_sha256": source_hash,
        "llm_model": args.llm_model,
        "asr_model": args.asr_model,
        **prompt_pack.metadata(),
        "stop_reason": stop_reason,
        "wall_seconds": time.monotonic() - started,
        "counters": vars(counters),
        "summary": summary,
        "working_memory": memory,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    trace.write("run.finish", result=result)
    print(json.dumps({"status": "complete", **result["counters"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
