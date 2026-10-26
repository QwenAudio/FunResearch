"""Qwen3-Omni plus LoRA inference for the three paper protocols.

Heavy model imports are lazy so prompt, parsing, scoring, and evaluation code
remains usable on CPU-only machines.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .protocols import (
    SINGLE_RUBRIC_SYSTEM_PROMPT,
    batch_guided_json_regex,
    build_batch_rubric_prompt,
    build_holistic_messages,
    build_single_rubric_messages,
    order_and_chunk_rubrics,
    parse_batch_labels,
    parse_holistic_score,
    parse_yes_no,
    rubric_id,
    validate_rubrics,
)

SYSTEM_PROMPT = SINGLE_RUBRIC_SYSTEM_PROMPT
build_judge_messages = build_single_rubric_messages


@dataclass(frozen=True)
class JudgePrediction:
    rubric_id: str
    question: str
    answer: str | None
    parse_mode: str
    raw_response: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "question": self.question,
            "answer": self.answer,
            "parse_mode": self.parse_mode,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class BatchChunkRecord:
    chunk_index: int
    chunk_count: int
    question_ids: tuple[str, ...]
    prompt_sha256: str
    raw_response: str
    used_guided_json_fallback: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "question_ids": list(self.question_ids),
            "prompt_sha256": self.prompt_sha256,
            "raw_response": self.raw_response,
            "used_guided_json_fallback": self.used_guided_json_fallback,
        }


@dataclass(frozen=True)
class BatchJudgeResult:
    predictions: tuple[JudgePrediction, ...]
    chunks: tuple[BatchChunkRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictions": [row.as_dict() for row in self.predictions],
            "chunks": [row.as_dict() for row in self.chunks],
        }


def _checked_audio_path(audio_path: str | Path) -> Path:
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _set_qwen_runtime_environment() -> None:
    os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "0")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
    os.environ.setdefault("FPS_MAX_FRAMES", "12")
    os.environ.setdefault("MAX_PIXELS", "1003520")


class SingleRubricJudge:
    """Paper Table 2/3 protocol: one audio-rubric pair per request."""

    def __init__(
        self,
        model: str | Path,
        *,
        adapter: str | Path | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        batch_size: int = 8,
        seed: int = 20260919,
        enforce_eager: bool = False,
        guided_yes_no: bool = True,
    ) -> None:
        try:
            import torch
            from swift.infer_engine import RequestConfig, VllmEngine
        except ImportError as exc:
            raise RuntimeError(
                "Judge inference requires torch, ms-swift, vLLM, and the "
                "Qwen3-Omni runtime; install the inference dependencies first."
            ) from exc

        _set_qwen_runtime_environment()
        self._batch_size = batch_size
        engine_kwargs: dict[str, Any] = {}
        if adapter:
            engine_kwargs.update({"adapters": [str(adapter)], "max_lora_rank": 64})
        self._engine = VllmEngine(
            str(model),
            torch_dtype=torch.bfloat16,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=batch_size,
            limit_mm_per_prompt={"audio": 1, "image": 1, "video": 0},
            seed=seed,
            enforce_eager=enforce_eager,
            **engine_kwargs,
        )
        self._request_config = RequestConfig(
            max_tokens=8, temperature=0.0, top_p=1.0, top_k=-1,
            repetition_penalty=1.0, seed=seed, stream=False,
            structured_outputs_regex=r"(?:yes|no)" if guided_yes_no else None,
        )

    def predict(
        self,
        audio_path: str | Path,
        rubrics: Iterable[dict[str, Any]],
    ) -> list[JudgePrediction]:
        from swift.infer_engine import InferRequest

        path = _checked_audio_path(audio_path)
        rows = validate_rubrics(rubrics)
        predictions: list[JudgePrediction] = []
        for start in range(0, len(rows), self._batch_size):
            batch = rows[start:start + self._batch_size]
            requests = [InferRequest(
                messages=build_single_rubric_messages(
                    str(row["question"]), str(row.get("subtype") or "")
                ),
                audios=[str(path)],
            ) for row in batch]
            responses = self._engine.infer(requests, self._request_config, use_tqdm=False)
            if len(responses) != len(batch):
                raise RuntimeError("inference response count mismatch")
            for row, response in zip(batch, responses):
                raw = response.choices[0].message.content or ""
                answer, mode = parse_yes_no(raw)
                predictions.append(JudgePrediction(
                    rubric_id=rubric_id(row),
                    question=str(row["question"]), answer=answer,
                    parse_mode=mode, raw_response=raw,
                ))
        return predictions


class BatchRubricJudge:
    """Paper Table 1 AAJ/Ours protocol: joint structured-JSON rubrics."""

    def __init__(
        self,
        model: str | Path,
        *,
        adapter: str | Path | None = None,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        batch_size: int = 4,
        max_tokens: int = 4096,
        chunk_size: int = 80,
        temperature: float = 0.0,
        seed: int = 20260919,
        ordering_backend: str = "qwen",
        guided_json_fallback: bool = True,
        enforce_eager: bool = False,
    ) -> None:
        try:
            import torch
            from swift.infer_engine import RequestConfig, VllmEngine
        except ImportError as exc:
            raise RuntimeError(
                "Batch rubric inference requires torch, ms-swift, vLLM, and "
                "the Qwen3-Omni runtime."
            ) from exc
        _set_qwen_runtime_environment()
        self._request_config_type = RequestConfig
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._chunk_size = chunk_size
        self._ordering_backend = ordering_backend
        self._guided_json_fallback = guided_json_fallback
        engine_kwargs: dict[str, Any] = {}
        if adapter:
            engine_kwargs.update({"adapters": [str(adapter)], "max_lora_rank": 64})
        self._engine = VllmEngine(
            str(model),
            torch_dtype=torch.bfloat16,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=batch_size,
            limit_mm_per_prompt={"audio": 1, "image": 1, "video": 0},
            seed=seed,
            enforce_eager=enforce_eager,
            **engine_kwargs,
        )

    def _request_config(self, rubrics: list[dict[str, Any]], guided: bool) -> Any:
        return self._request_config_type(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            seed=self._seed,
            stream=False,
            structured_outputs_regex=batch_guided_json_regex(rubrics) if guided else None,
        )

    def _infer_chunk(
        self,
        audio_path: Path,
        rubrics: list[dict[str, Any]],
        *,
        guided: bool,
    ) -> tuple[list[dict[str, str]], str]:
        from swift.infer_engine import InferRequest

        prompt = build_batch_rubric_prompt(rubrics)
        request = InferRequest(
            messages=[{"role": "user", "content": "<audio>\n\n" + prompt}],
            audios=[str(audio_path)],
        )
        response = self._engine.infer(
            [request], self._request_config(rubrics, guided), use_tqdm=False
        )[0]
        raw = response.choices[0].message.content or ""
        return parse_batch_labels(raw, rubrics), raw

    def predict(
        self,
        audio_path: str | Path,
        rubrics: Iterable[dict[str, Any]],
        *,
        sample_id: str,
    ) -> BatchJudgeResult:
        path = _checked_audio_path(audio_path)
        original = validate_rubrics(rubrics)
        by_id = {rubric_id(row): row for row in original}
        labels: dict[str, str] = {}
        chunks: list[BatchChunkRecord] = []
        ordered_chunks = order_and_chunk_rubrics(
            sample_id,
            original,
            chunk_size=self._chunk_size,
            ordering_backend=self._ordering_backend,
        )
        for chunk_index, chunk in enumerate(ordered_chunks):
            used_fallback = False
            try:
                parsed, raw = self._infer_chunk(path, chunk, guided=False)
            except ValueError:
                if not self._guided_json_fallback:
                    raise
                parsed, raw = self._infer_chunk(path, chunk, guided=True)
                used_fallback = True
            for row in parsed:
                labels[row["rubric_id"]] = row["answer"]
            chunks.append(BatchChunkRecord(
                chunk_index=chunk_index,
                chunk_count=len(ordered_chunks),
                question_ids=tuple(rubric_id(row) for row in chunk),
                prompt_sha256=hashlib.sha256(
                    build_batch_rubric_prompt(chunk).encode("utf-8")
                ).hexdigest(),
                raw_response=raw,
                used_guided_json_fallback=used_fallback,
            ))
        predictions = tuple(
            JudgePrediction(
                rubric_id=identifier,
                question=str(row["question"]),
                answer=labels[identifier],
                parse_mode="structured_json",
                raw_response="",
            )
            for identifier, row in ((rubric_id(row), row) for row in original)
        )
        if set(labels) != set(by_id):
            raise RuntimeError("batch inference did not return exact rubric coverage")
        return BatchJudgeResult(predictions=predictions, chunks=tuple(chunks))


@dataclass(frozen=True)
class HolisticPrediction:
    score: int | None
    parse_mode: str
    raw_response: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "parse_mode": self.parse_mode,
            "raw_response": self.raw_response,
        }


class HolisticJudge:
    """Paper Table 1 Holistic protocol: independent 0--100 scoring."""

    def __init__(
        self,
        model: str | Path,
        *,
        adapter: str | Path | None = None,
        max_tokens: int = 32,
        temperature: float = 0.0,
        seed: int = 20260924,
        device_map: str = "auto",
        force_use_cache: bool = False,
    ) -> None:
        try:
            import torch
            from swift.infer_engine import RequestConfig, TransformersEngine
        except ImportError as exc:
            raise RuntimeError(
                "Holistic inference requires torch, ms-swift, transformers, "
                "and the Qwen3-Omni runtime."
            ) from exc
        _set_qwen_runtime_environment()
        torch.manual_seed(seed)
        self._engine = TransformersEngine(
            str(model),
            adapters=[str(adapter)] if adapter else None,
            max_batch_size=1,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_impl="flash_attn",
        )
        if force_use_cache:
            configs = [
                getattr(self._engine.model, "config", None),
                getattr(self._engine.model, "generation_config", None),
            ]
            model_config = getattr(self._engine.model, "config", None)
            thinker_config = getattr(model_config, "thinker_config", None)
            configs.extend([thinker_config, getattr(thinker_config, "text_config", None)])
            thinker = getattr(self._engine.model, "thinker", None)
            configs.extend([
                getattr(thinker, "config", None),
                getattr(getattr(thinker, "model", None), "config", None),
            ])
            for config in configs:
                if config is not None and hasattr(config, "use_cache"):
                    config.use_cache = True
        self._request_config = RequestConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=seed,
            stream=False,
        )

    def predict(self, audio_path: str | Path, prompt: str) -> HolisticPrediction:
        from swift.infer_engine import InferRequest

        path = _checked_audio_path(audio_path)
        response = self._engine.infer(
            [InferRequest(messages=build_holistic_messages(prompt), audios=[str(path)])],
            self._request_config,
            use_tqdm=False,
        )[0]
        raw = response.choices[0].message.content or ""
        score, parse_mode = parse_holistic_score(raw)
        return HolisticPrediction(score=score, parse_mode=parse_mode, raw_response=raw)


# The end-to-end FineAudio-Judge score in Table 1 uses joint rubric batches.
FineAudioJudge = BatchRubricJudge
