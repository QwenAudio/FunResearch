from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int


class ChatClient:
    """Small dependency-free client for an OpenAI-compatible chat endpoint."""

    def __init__(self, api_base: str, model: str, timeout: int = 240) -> None:
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def json_completion(
        self,
        prompt: str,
        sampling: SamplingConfig,
        seed: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        return parse_json_object(content)


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def prompt_from_response(response: dict[str, Any], key: str = "prompt") -> str:
    value = response.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model response must contain a non-empty string field: {key}")
    return value.strip()
