"""Rubric-level decisions, flat aggregation, and pairwise ranking."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable


def normalized_yes_probability(yes_logprob: float, no_logprob: float) -> float:
    """Return exp(yes)/(exp(yes)+exp(no)) without numerical overflow."""
    difference = no_logprob - yes_logprob
    if difference >= 0:
        value = math.exp(-difference)
        return value / (1.0 + value)
    return 1.0 / (1.0 + math.exp(difference))


def aggregate_rubric_scores(
    predictions: Iterable[dict[str, Any]],
    *,
    missing: str = "exclude",
) -> dict[str, Any]:
    """Aggregate rubrics with the flat scoring used in the final experiments."""
    rows = list(predictions)
    if missing not in {"exclude", "zero", "error"}:
        raise ValueError("missing must be exclude, zero, or error")
    values: list[float] = []
    by_axis: dict[str, list[float]] = defaultdict(list)
    answers: Counter[str] = Counter()
    for row in rows:
        if row.get("yes_probability") is not None:
            value = float(row["yes_probability"])
            if not 0.0 <= value <= 1.0:
                raise ValueError("yes_probability must be within [0, 1]")
            answer = "yes" if value >= 0.5 else "no"
        else:
            answer = str(row.get("answer") or "").lower()
            if answer in {"yes", "no"}:
                value = 1.0 if answer == "yes" else 0.0
            elif missing == "zero":
                value, answer = 0.0, "missing"
            elif missing == "error":
                raise ValueError(f"missing answer for rubric {row.get('rubric_id')}")
            else:
                answers["missing"] += 1
                continue
        values.append(value)
        answers[answer] += 1
        axis = str(row.get("axis") or "unspecified")
        by_axis[axis].append(value)
    return {
        "score": sum(values) / len(values) if values else None,
        "rubric_count": len(rows),
        "scored_count": len(values),
        "answers": dict(sorted(answers.items())),
        "by_axis": {
            axis: {"score": sum(group) / len(group), "count": len(group)}
            for axis, group in sorted(by_axis.items())
        },
    }


def rank_pair(first: dict[str, Any], second: dict[str, Any], *, tie_tolerance: float = 0.0) -> str:
    a = first.get("score")
    b = second.get("score")
    if a is None or b is None:
        raise ValueError("both candidates require a score")
    if abs(float(a) - float(b)) <= tie_tolerance:
        return "tie"
    return "first" if float(a) > float(b) else "second"
