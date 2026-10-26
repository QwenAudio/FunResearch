from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, TypeVar
import uuid

from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

TEMPERATURE_ENV = "FINEAUDIO_DECOMPOSITION_TEMPERATURE"
SEED_ENV = "FINEAUDIO_DECOMPOSITION_SEED"


class VllmJsonClient:
    def __init__(self, *, base_url: str, model: str, api_key: str | None = None,
                 concurrency: int = 16, timeout: float = 600.0,
                 temperature: float | None = None, seed: int | None = None,
                 audit_path: Path | None = None) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "set OPENAI_API_KEY for the selected endpoint; unauthenticated local "
                "servers may use an explicit placeholder value"
            )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        if temperature is None:
            temperature = float(os.environ.get(TEMPERATURE_ENV, "0.0"))
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if seed is None:
            seed_text = os.environ.get(SEED_ENV, "").strip()
            seed = int(seed_text) if seed_text else None
        self.temperature = temperature
        self.seed = seed
        self.audit_path = audit_path
        self._audit_lock = asyncio.Lock()

    async def _write_audit(self, record: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        async with self._audit_lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def generate(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int,
        audit_context: dict[str, Any] | None = None,
    ) -> tuple[T, str]:
        last_error: Exception | None = None
        logical_call_id = uuid.uuid4().hex
        for attempt in range(3):
            raw = ""
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            request_seed = self.seed + attempt if self.seed is not None else None
            try:
                async with self.semaphore:
                    request: dict[str, Any] = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"},
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    }
                    if request_seed is not None:
                        request["seed"] = request_seed
                    response = await self.client.chat.completions.create(
                        **request,
                    )
                raw = response.choices[0].message.content or ""
                parsed = schema.model_validate(json.loads(raw))
                await self._write_audit({
                    "logical_call_id": logical_call_id,
                    "client_attempt": attempt + 1,
                    "started_at": started_at,
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "model": self.model,
                    "temperature": self.temperature,
                    "seed": request_seed,
                    "thinking_enabled": False,
                    "max_tokens": max_tokens,
                    "response_schema": schema.__name__,
                    "context": audit_context or {},
                    "system": system,
                    "user": prompt,
                    "status": "ok",
                    "raw_model_output": raw,
                })
                return parsed, raw
            except Exception as exc:
                last_error = exc
                await self._write_audit({
                    "logical_call_id": logical_call_id,
                    "client_attempt": attempt + 1,
                    "started_at": started_at,
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "model": self.model,
                    "temperature": self.temperature,
                    "seed": request_seed,
                    "thinking_enabled": False,
                    "max_tokens": max_tokens,
                    "response_schema": schema.__name__,
                    "context": audit_context or {},
                    "system": system,
                    "user": prompt,
                    "status": "error",
                    "raw_model_output": raw,
                    "error": repr(exc),
                })
                await asyncio.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"JSON generation failed after retries: {last_error}") from last_error
