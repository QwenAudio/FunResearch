"""Single-prompt wrapper around the reference batch pipeline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl
from .pipeline import PIPELINE_NAME, run


async def _run_one(
    prompt: str,
    output_dir: Path,
    *,
    base_url: str,
    model: str,
    tokenizer: str,
    timeout: float,
    temperature: float | None,
    seed: int | None,
) -> dict[str, Any]:
    sample_id = "prompt_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, [{"sample_id": sample_id, "route": "single_prompt", "prompt": prompt}])
    args = argparse.Namespace(
        manifest=manifest,
        output_dir=output_dir,
        base_url=base_url,
        model=model,
        tokenizer=tokenizer,
        concurrency=1,
        timeout=timeout,
        scene_graph_tokens=8192,
        context_window=32768,
        question_tokens=4096,
        question_output_tokens_per_question=160,
        question_safety_margin=2048,
        question_max_tuples_per_chunk=64,
        stop_after_relation_audit=False,
        temperature=temperature,
        seed=seed,
    )
    await run(args)
    rows = read_jsonl(output_dir / "results.jsonl")
    if len(rows) != 1 or rows[0].get("sample_id") != sample_id:
        raise RuntimeError("decomposition did not produce exactly one matching result")
    if rows[0].get("pipeline_name") != PIPELINE_NAME:
        raise RuntimeError("unexpected rubric pipeline name")
    return rows[0]


async def decompose_prompt(
    prompt: str,
    *,
    base_url: str,
    model: str,
    tokenizer: str,
    output_dir: str | Path,
    timeout: float = 600.0,
    temperature: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the exact final-evaluation decomposition stages for one prompt.

    ``output_dir`` is required so the resolved configuration, stage
    checkpoints, and LLM instructions are retained for audit.
    """
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not prompt:
        raise ValueError("prompt must be a non-empty string")
    for name, value in (("base_url", base_url), ("model", model), ("tokenizer", tokenizer)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return await _run_one(
        prompt, destination, base_url=base_url.strip(), model=model.strip(),
        tokenizer=tokenizer.strip(), timeout=timeout, temperature=temperature,
        seed=seed,
    )
