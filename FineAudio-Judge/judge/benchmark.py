"""Load and score the released FineAudio-Judge benchmark formats.

The public benchmark intentionally uses different schemas for rubric QA and
human preference.  These helpers turn both schemas into the model-facing
records consumed by :mod:`judge.judge` and implement the exact
paper scoring rules.
"""

from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_audio_path(dataset_root: str | Path, value: str) -> Path:
    """Resolve a repository-relative audio path and reject path traversal."""

    root = Path(dataset_root).expanduser().resolve()
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_absolute() and root not in resolved.parents:
        raise ValueError(f"audio path escapes dataset root: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_rubric_qa(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Load the 33,498-row QA file as 700 model-facing audio records."""

    root = Path(dataset_root).expanduser().resolve()
    rows = read_jsonl(root / "rubric_qa/rubric_qa.jsonl")
    grouped: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    audio_paths: dict[str, Path] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        identifier = str(row["rubric_id"])
        key = (sample_id, identifier)
        if key in seen:
            raise ValueError(f"duplicate QA key: {key}")
        seen.add(key)
        audio_value = str(row["audio"])
        if audio_value not in audio_paths:
            audio_paths[audio_value] = resolve_audio_path(root, audio_value)
        audio_path = audio_paths[audio_value]
        item = grouped.setdefault(sample_id, {
            "sample_id": sample_id,
            "subset": row["subset"],
            "rubric_set_id": row["rubric_set_id"],
            "language": row["language"],
            "audio": row["audio"],
            "audio_path": str(audio_path),
            "questions": [],
        })
        invariant = (
            item["subset"], item["rubric_set_id"], item["language"],
            item["audio"], item["audio_path"],
        )
        current = (
            row["subset"], row["rubric_set_id"], row["language"],
            row["audio"], str(audio_path),
        )
        if invariant != current:
            raise ValueError(f"inconsistent metadata for QA sample {sample_id}")
        item["questions"].append({
            "rubric_id": identifier,
            "question": row["question"],
            "subtype": row.get("subtype"),
            "axis": row.get("axis"),
            "scope": row.get("scope"),
            "domain": row.get("domain", []),
            "parent_ids": row.get("parent_ids", []),
            "reference_label": row["label"],
            "vote_pattern": row.get("vote_pattern"),
        })
    output = sorted(grouped.values(), key=lambda row: row["sample_id"])
    for item in output:
        item["questions"].sort(key=lambda row: row["rubric_id"])
    if len(rows) != sum(len(item["questions"]) for item in output):
        raise RuntimeError("QA grouping lost questions")
    return output


def load_preference_candidates(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Join candidate groups with prompt rubrics into 300 inference jobs."""

    root = Path(dataset_root).expanduser().resolve()
    groups = read_jsonl(root / "human_preference/candidate_groups.jsonl")
    rubrics = {
        str(row["group_id"]): row
        for row in read_jsonl(root / "human_preference/prompt_rubrics.jsonl")
    }
    if len(rubrics) != 100:
        raise ValueError(f"expected 100 preference rubric groups, found {len(rubrics)}")
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        group_id = str(group["group_id"])
        rubric_row = rubrics.get(group_id)
        if rubric_row is None:
            raise ValueError(f"missing rubrics for {group_id}")
        if str(group["prompt"]) != str(rubric_row["prompt"]):
            raise ValueError(f"prompt mismatch for {group_id}")
        questions = list(rubric_row["rubrics"])
        if int(rubric_row["rubric_count"]) != len(questions):
            raise ValueError(f"rubric_count mismatch for {group_id}")
        for candidate in group["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            key = (group_id, candidate_id)
            if key in seen:
                raise ValueError(f"duplicate preference candidate: {key}")
            seen.add(key)
            jobs.append({
                "sample_id": f"{group_id}:{candidate_id}",
                "group_id": group_id,
                "candidate_id": candidate_id,
                "language": group["language"],
                "category": group["category"],
                "prompt": group["prompt"],
                "audio": candidate["audio"],
                "audio_path": str(resolve_audio_path(root, str(candidate["audio"]))),
                "questions": questions,
            })
    return sorted(jobs, key=lambda row: (row["group_id"], row["candidate_id"]))


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("reference_label") in {"yes", "no"}
        and row.get("pred_answer") in {"yes", "no"}
    ]
    correct = sum(row["reference_label"] == row["pred_answer"] for row in valid)
    recalls: dict[str, float | None] = {}
    for label in ("yes", "no"):
        labeled = [row for row in valid if row["reference_label"] == label]
        recalls[label] = (
            sum(row["pred_answer"] == label for row in labeled) / len(labeled)
            if labeled else None
        )
    recall_values = [value for value in recalls.values() if value is not None]
    return {
        "questions": len(rows),
        "scored_questions": len(valid),
        "missing_or_unparsed": len(rows) - len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else None,
        "yes_recall": recalls["yes"],
        "no_recall": recalls["no"],
        "balanced_accuracy": sum(recall_values) / len(recall_values) if recall_values else None,
    }


def rubric_qa_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute question-micro and sample-macro QA metrics."""

    values = list(rows)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in values:
        key = (str(row.get("sample_id")), str(row.get("rubric_id")))
        if key not in latest or row.get("status") == "success":
            latest[key] = row
    valid = list(latest.values())
    per_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        per_sample[str(row["sample_id"])].append(row)
        per_subset[str(row["subset"])].append(row)

    def sample_macro(sample_rows: dict[str, list[dict[str, Any]]]) -> float | None:
        scores = []
        for group in sample_rows.values():
            metric = _binary_metrics(group)
            if metric["accuracy"] is not None:
                scores.append(float(metric["accuracy"]))
        return sum(scores) / len(scores) if scores else None

    by_subset: dict[str, Any] = {}
    for subset, subset_rows in sorted(per_subset.items()):
        subset_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset_rows:
            subset_samples[str(row["sample_id"])].append(row)
        by_subset[subset] = {
            **_binary_metrics(subset_rows),
            "samples": len(subset_samples),
            "sample_macro_accuracy": sample_macro(subset_samples),
        }
    return {
        "protocol": "single_rubric",
        "overall": {
            **_binary_metrics(valid),
            "samples": len(per_sample),
            "sample_macro_accuracy": sample_macro(per_sample),
        },
        "by_subset": by_subset,
    }


def preference_report(
    dataset_root: str | Path,
    score_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Score candidate yes-ratios using the frozen paper tie policy."""

    root = Path(dataset_root).expanduser().resolve()
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in score_rows:
        if row.get("status") != "success":
            continue
        key = (str(row["group_id"]), str(row["candidate_id"]))
        latest[key] = row

    fractions: dict[tuple[str, str], Fraction] = {}
    for key, row in latest.items():
        yes = int(row["yes_count"])
        total = int(row["rubric_count"])
        if total <= 0 or yes < 0 or yes > total:
            raise ValueError(f"invalid preference score for {key}: {yes}/{total}")
        fractions[key] = Fraction(yes, total)

    pair_rows = read_jsonl(root / "human_preference/human_preference_pairwise.jsonl")
    pair_details: list[dict[str, Any]] = []
    for row in pair_rows:
        group_id = str(row["group_id"])
        first = str(row["candidate_a"]["candidate_id"])
        second = str(row["candidate_b"]["candidate_id"])
        first_score = fractions.get((group_id, first))
        second_score = fractions.get((group_id, second))
        predicted: str | None = None
        if first_score is not None and second_score is not None and first_score != second_score:
            predicted = first if first_score > second_score else second
        pair_details.append({
            "pair_id": row["pair_id"],
            "group_id": group_id,
            "candidate_a": first,
            "candidate_b": second,
            "candidate_a_score": float(first_score) if first_score is not None else None,
            "candidate_b_score": float(second_score) if second_score is not None else None,
            "human_winner": row["label"],
            "predicted_winner": predicted,
            "score_tie": first_score is not None and first_score == second_score,
            "correct": predicted == row["label"],
            "consensus_level": row["consensus_level"],
            "vote_pattern": row["vote_pattern"],
            "covered": first_score is not None and second_score is not None,
        })

    def pair_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        covered = [row for row in rows if row["covered"]]
        correct = sum(bool(row["correct"]) for row in covered)
        ties = sum(bool(row["score_tie"]) for row in covered)
        return {
            "eligible_pairs": len(rows),
            "covered_pairs": len(covered),
            "missing_pairs": len(rows) - len(covered),
            "correct_pairs": correct,
            "accuracy": correct / len(covered) if covered else None,
            "ties_counted_as_incorrect": ties,
        }

    ranking_rows = read_jsonl(root / "human_preference/human_preference_ranking.jsonl")
    ranking_details: list[dict[str, Any]] = []
    for row in ranking_rows:
        group_id = str(row["group_id"])
        human = [str(candidate_id) for candidate_id in row["label"]]
        scores = {candidate_id: fractions.get((group_id, candidate_id)) for candidate_id in human}
        covered = all(score is not None for score in scores.values())
        tied = covered and len(set(scores.values())) < len(scores)
        predicted = None
        if covered and not tied:
            predicted = sorted(human, key=lambda candidate_id: scores[candidate_id], reverse=True)
        ranking_details.append({
            "group_id": group_id,
            "candidate_scores": {
                candidate_id: float(score) if score is not None else None
                for candidate_id, score in scores.items()
            },
            "human_ranking": human,
            "predicted_ranking": predicted,
            "score_tie": tied,
            "exact_match": predicted == human,
            "covered": covered,
            "consensus_level": row["consensus_level"],
            "modal_votes": row["modal_votes"],
        })

    def ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        covered = [row for row in rows if row["covered"]]
        exact = sum(bool(row["exact_match"]) for row in covered)
        ties = sum(bool(row["score_tie"]) for row in covered)
        return {
            "eligible_groups": len(rows),
            "covered_groups": len(covered),
            "missing_groups": len(rows) - len(covered),
            "exact_matches": exact,
            "exact_ranking_accuracy": exact / len(covered) if covered else None,
            "ties_counted_as_incorrect": ties,
        }

    return {
        "protocol": "joint_rubric",
        "scoring": "candidate yes_count / rubric_count; exact score ties count as incorrect",
        "candidate_scores": {
            "expected": 300,
            "available": len(fractions),
        },
        "pairwise": {
            "overall": pair_metrics(pair_details),
            "by_consensus_level": {
                level: pair_metrics([row for row in pair_details if row["consensus_level"] == level])
                for level in ("high", "medium", "low")
            },
            "details": pair_details,
        },
        "ranking": {
            "overall": ranking_metrics(ranking_details),
            "by_consensus_level": {
                level: ranking_metrics([
                    row for row in ranking_details if row["consensus_level"] == level
                ])
                for level in ("strict_majority", "unique_plurality")
            },
            "details": ranking_details,
        },
    }
