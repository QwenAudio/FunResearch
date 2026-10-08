"""Pairwise accuracy with the paper evaluation implementation's tie convention."""

from __future__ import annotations

import math


def pairwise_accuracy(rows, *, tie_threshold: float = 1e-8) -> dict:
    """Rows contain score_a, score_b and preferred ('a' or 'b').

    Predictions within tie_threshold receive half credit, matching the original
    evaluation scripts. Human/proxy label ties must be excluded beforehand.
    """
    if not math.isfinite(tie_threshold) or tie_threshold < 0:
        raise ValueError("tie_threshold must be finite and non-negative")
    credits, ties = [], 0
    for index, row in enumerate(rows):
        if row.get("preferred") not in ("a", "b"):
            raise ValueError(f"Row {index + 1}: preferred must be 'a' or 'b'")
        a, b = float(row["score_a"]), float(row["score_b"])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"Row {index + 1}: scores must be finite")
        if abs(a - b) <= tie_threshold:
            credits.append(0.5)
            ties += 1
        else:
            credits.append(float(("a" if a > b else "b") == row["preferred"]))
    if not credits:
        raise ValueError("No labeled comparisons supplied")
    return {"n": len(credits), "correct": sum(credits), "ties": ties,
            "pairacc_percent": 100.0 * sum(credits) / len(credits)}
