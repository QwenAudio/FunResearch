from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubric_decomposition.models import (  # noqa: E402
    AtomicTuple, AudioSceneGraph, DependencyRecord, RubricQuestion,
)
from rubric_decomposition.tuples import build_tuples  # noqa: E402
from rubric_decomposition.validation import (  # noqa: E402
    build_dependencies, validate_dependencies, validate_questions,
    validate_scene_graph,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_audio_matches_parent(row: dict[str, Any]) -> bool:
    if row.get("audio_byte_identical_to_parent") is not True:
        return False
    source = Path(str(row.get("source_audio_path") or ""))
    evaluation = Path(str(row.get("evaluation_audio_path") or ""))
    if not source.is_file() or not evaluation.is_file():
        return False
    source_hash = sha256_file(source)
    return (
        source_hash == sha256_file(evaluation)
        and (not row.get("source_audio_sha256") or row["source_audio_sha256"] == source_hash)
    )


def canonical_rubric_errors(row: dict[str, Any]) -> list[str]:
    try:
        prompt = str(row["prompt"])
        graph = AudioSceneGraph.model_validate(row["scene_graph"])
        tuples = [AtomicTuple.model_validate(item) for item in row["atomic_tuples"]]
        dependencies = [
            DependencyRecord.model_validate(item) for item in row["dependencies"]
        ]
        questions = [RubricQuestion.model_validate(item) for item in row["questions"]]
    except (KeyError, TypeError, ValueError) as error:
        return [f"schema:{error}"]
    errors = validate_scene_graph(prompt, graph)
    rebuilt_tuples = build_tuples(graph)
    if [item.model_dump() for item in rebuilt_tuples] != [item.model_dump() for item in tuples]:
        errors.append("tuple_graph_mismatch")
    try:
        rebuilt_dependencies = build_dependencies(rebuilt_tuples)
    except ValueError as error:
        errors.append(f"dependency_rebuild:{error}")
    else:
        if [item.model_dump() for item in rebuilt_dependencies] != [
            item.model_dump() for item in dependencies
        ]:
            errors.append("dependency_tuple_mismatch")
    errors.extend(validate_dependencies(tuples, dependencies))
    errors.extend(
        validate_questions(
            prompt, tuples, dependencies, questions,
            language_override=row.get("language"),
        )
    )
    validation = row.get("validation") or {}
    if validation.get("valid") is not True:
        errors.append("decomposition_validation_not_valid")
    return errors


def attach_prompt_rubrics(
    prompt_pairs: list[dict[str, Any]], rubric_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rubric_rows = read_jsonl(rubric_path)
    by_id: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for row in rubric_rows:
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if not sample_id:
            errors.append({"sample_id": None, "errors": ["missing_sample_id"]})
            continue
        if sample_id in by_id:
            errors.append({"sample_id": sample_id, "errors": ["duplicate_sample_id"]})
            continue
        by_id[sample_id] = row

    enriched: list[dict[str, Any]] = []
    expected_ids = {str(row["sample_id"]) for row in prompt_pairs}
    extra_ids = sorted(set(by_id) - expected_ids)
    if extra_ids:
        errors.append({"sample_id": None, "errors": [f"unexpected_rubric_ids:{extra_ids[:5]}"]})
    for pair in prompt_pairs:
        sample_id = str(pair["sample_id"])
        rubric = by_id.get(sample_id)
        if rubric is None:
            errors.append({"sample_id": sample_id, "errors": ["missing_rubric_row"]})
            enriched.append(pair)
            continue
        prompt = str(pair.get("prompt_prime") or pair.get("evaluation_prompt") or "")
        if str(rubric.get("prompt") or "") != prompt:
            errors.append({"sample_id": sample_id, "errors": ["prompt_mismatch"]})
            enriched.append(pair)
            continue
        row_errors = canonical_rubric_errors(rubric)
        if row_errors:
            errors.append({"sample_id": sample_id, "errors": row_errors})
            enriched.append(pair)
            continue
        enriched.append({
            **pair,
            "scene_graph": rubric["scene_graph"],
            "atomic_tuples": rubric["atomic_tuples"],
            "dependencies": rubric["dependencies"],
            "questions": rubric["questions"],
            "rubric_decomposition_contract": "rubric_decomposition",
            "evaluation_rubrics_status": "canonical_regenerated_from_P_prime",
        })
    return enriched, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Route 3 controlled negative pairs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--prompt-rubrics", type=Path, required=True,
        help="Canonical rubric-decomposition results for every prompt-side P-prime row.",
    )
    args = parser.parse_args()
    prompt_path = args.root / "prompt_side/manifests/prompt_side_pairs.jsonl"
    audio_path = args.root / "manifests/audio_side_pairs.jsonl"
    prompt = read_jsonl(prompt_path)
    prompt, rubric_errors = attach_prompt_rubrics(prompt, args.prompt_rubrics)
    audio = read_jsonl(audio_path)
    rows = prompt + audio
    sources = [str(row["source_pair_id"]) for row in rows]
    distribution = Counter(
        (str(row["pair_type"]), str(row["language"]), str(row["target_band"]))
        for row in rows
    )
    expected = Counter({
        (pair_type, language, band): 125
        for pair_type in ("P_prime_A", "P_A_prime")
        for language in ("zh", "en")
        for band in ("mid", "low")
    })
    checks = {
        "prompt_side_500": len(prompt) == 500,
        "audio_side_500": len(audio) == 500,
        "unique_positive_sources": len(sources) == len(set(sources)) == 1000,
        "balanced_cells": distribution == expected,
        "prompt_rubrics_canonical": not rubric_errors and all(
            row.get("evaluation_rubrics_status") == "canonical_regenerated_from_P_prime"
            for row in prompt
        ),
        "prompt_audio_is_parent": all(prompt_audio_matches_parent(row) for row in prompt),
        "audio_generation_complete": all(
            row.get("generation_status") == "complete" and row.get("evaluation_audio_path")
            for row in audio
        ),
    }
    output = args.root / "manifests/negative_pairs.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "valid": all(checks.values()),
        "checks": checks,
        "total": len(rows),
        "distribution": {"|".join(key): value for key, value in sorted(distribution.items())},
        "rubric_errors": rubric_errors,
        "output": str(output.resolve()),
    }
    (args.root / "manifests/audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
