from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .prompts import event_label_map, question_prompt, question_tasks, target_language


def pack_question_chunks(
    *,
    prompt: str,
    tuples: list[dict[str, Any]],
    dependencies: list[dict[str, Any]],
    count_tokens: Callable[[str], int],
    context_window: int,
    max_output_tokens: int,
    output_tokens_per_question: int,
    safety_margin: int,
    max_tuples_per_chunk: int,
    target_language_override: str | None = None,
) -> list[dict[str, Any]]:
    """Greedily pack tuple-aligned question requests under a token budget."""
    if not tuples:
        return []
    dependency_map = {item["tuple_id"]: item for item in dependencies}
    labels = event_label_map(tuples)
    if target_language_override:
        language = (
            "CHINESE (中文)"
            if target_language_override.casefold().startswith("zh")
            else "ENGLISH"
        )
    else:
        language = target_language(prompt)

    def make_plan(chunk: list[dict[str, Any]]) -> dict[str, Any]:
        chunk_dependencies = [dependency_map[item["tuple_id"]] for item in chunk]
        tasks = question_tasks(chunk, labels, language)
        rendered = question_prompt(language, tasks)
        input_tokens = count_tokens(rendered)
        required_output_tokens = max(256, output_tokens_per_question * len(chunk))
        output_tokens = min(max_output_tokens, required_output_tokens)
        return {
            "tuples": chunk,
            "dependencies": chunk_dependencies,
            "question_tasks": tasks,
            "target_language": language,
            "input_tokens": input_tokens,
            "required_output_tokens": required_output_tokens,
            "output_tokens": output_tokens,
            "total_reserved_tokens": input_tokens + output_tokens + safety_margin,
        }

    plans: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for item in tuples:
        candidate = current + [item]
        plan = make_plan(candidate)
        exceeds_budget = plan["total_reserved_tokens"] > context_window
        exceeds_output = plan["required_output_tokens"] > max_output_tokens
        exceeds_count = len(candidate) > max_tuples_per_chunk
        if current and (exceeds_budget or exceeds_output or exceeds_count):
            plans.append(make_plan(current))
            current = [item]
            plan = make_plan(current)
            if (plan["total_reserved_tokens"] > context_window
                    or plan["required_output_tokens"] > max_output_tokens):
                raise ValueError(
                    f"Question tuple {item['tuple_id']} exceeds the context budget alone: "
                    f"reserved={plan['total_reserved_tokens']} window={context_window} "
                    f"required_output={plan['required_output_tokens']} max_output={max_output_tokens}"
                )
        else:
            current = candidate
    if current:
        plans.append(make_plan(current))
    return plans
