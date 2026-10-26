"""Minimal offline vLLM backend used for caption validation and finalization."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from common import parse_json_object
from content_safety import profanity_categories


SYSTEM_PROMPT = (
    "You are a meticulous data engineer. Follow the requested JSON schema and "
    "use only evidence present in the input."
)


class VllmTextBatchRunner:
    def __init__(
        self,
        model_path: str,
        tp: int,
        max_num_seqs: int,
        max_model_len: int,
        submit_batch_size: int,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.submit_batch_size = submit_batch_size
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.92,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            language_model_only=True,
            seed=1234,
        )

    def render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(
        self, prompts: list[str], max_tokens: int, *, json_object: bool = False
    ) -> list[str]:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=(
                StructuredOutputsParams(json_object=True) if json_object else None
            ),
        )
        results: list[str] = []
        for start in range(0, len(prompts), self.submit_batch_size):
            values = [
                self.render(prompt)
                for prompt in prompts[start : start + self.submit_batch_size]
            ]
            results.extend(
                item.outputs[0].text.strip()
                for item in self.llm.generate(values, params)
            )
        return results

    def generate_json(
        self, prompts: list[str], max_tokens: int
    ) -> list[tuple[dict[str, Any], str]]:
        raw = self.generate(prompts, max_tokens, json_object=True)
        parsed: list[dict[str, Any] | None] = []
        bad_indices: list[int] = []
        for index, text in enumerate(raw):
            try:
                parsed.append(parse_json_object(text))
            except Exception:
                parsed.append(None)
                bad_indices.append(index)
        if bad_indices:
            retries = self.generate(
                [prompts[index] for index in bad_indices],
                min(max_tokens * 2, max_tokens + 4000),
                json_object=True,
            )
            still_bad: list[int] = []
            for index, retry in zip(bad_indices, retries, strict=True):
                raw[index] += "\n\n[RETRIED_WITH_LARGER_BUDGET]\n" + retry
                try:
                    parsed[index] = parse_json_object(retry)
                except Exception:
                    still_bad.append(index)
        else:
            still_bad = []
        if still_bad:
            repairs = self.generate(
                [
                    "Repair only the JSON syntax in the content below so that it "
                    "becomes one strictly valid JSON object. Preserve all existing "
                    "fields and values. Return only the repaired JSON object:\n"
                    + raw[index]
                    for index in still_bad
                ],
                min(max_tokens * 2, max_tokens + 4000),
                json_object=True,
            )
            for index, repair in zip(still_bad, repairs, strict=True):
                raw[index] += "\n\n[REPAIRED]\n" + repair
                try:
                    parsed[index] = parse_json_object(repair)
                except Exception as exc:
                    parsed[index] = {
                        "_json_generation_failed": True,
                        "item_index": index,
                        "error": str(exc),
                    }
        return [
            (value or {}, text)
            for value, text in zip(parsed, raw, strict=True)
        ]


def caption_gate_errors(gate: dict[str, Any], *, detective: bool) -> list[str]:
    """Validate the coverage-gate JSON without adding semantic judgments."""
    errors: list[str] = []
    caption = gate.get("audio_caption")
    if not isinstance(caption, str) or len(caption.strip()) < 80:
        errors.append("missing_or_short_revised_caption")
    checks = gate.get("coverage_checks") or []
    minimum_checks = 3 if detective else 1
    if not isinstance(checks, list) or len(checks) < minimum_checks:
        errors.append("insufficient_coverage_checks")

    def canonical(value: Any) -> str:
        import re

        return "".join(
            re.findall(r"[a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", str(value).lower())
        )

    compact_caption = canonical(caption) if isinstance(caption, str) else ""
    invalid_checks = []
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            invalid_checks.append(str(index))
            continue
        evidence = item.get("caption_evidence")
        compact_evidence = canonical(evidence) if isinstance(evidence, str) else ""
        longest_match = (
            SequenceMatcher(None, compact_evidence, compact_caption)
            .find_longest_match()
            .size
            if compact_evidence and compact_caption
            else 0
        )
        evidence_is_grounded = (
            compact_evidence in compact_caption
            or longest_match >= min(8, max(5, len(compact_evidence) // 4))
        )
        if (
            item.get("covered") is not True
            or len(compact_evidence) < 2
            or not evidence_is_grounded
        ):
            invalid_checks.append(str(index))
    if invalid_checks:
        errors.append("invalid_coverage_checks:" + ",".join(invalid_checks))
    if gate.get("omitted_audible_evidence"):
        errors.append("omitted_audible_evidence_reported")
    if gate.get("contains_profanity") is not False:
        errors.append("model_gate_detected_profanity")
    categories = profanity_categories(caption)
    if categories:
        errors.append("deterministic_profanity_gate:" + ",".join(categories))
    return errors
