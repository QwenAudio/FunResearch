#!/usr/bin/env python3
"""Fail-closed structural, provenance, and semantic audit for Route 2 negatives."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
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
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def canonical_rubric_errors(row: dict[str, Any]) -> list[str]:
    try:
        graph = AudioSceneGraph.model_validate(row["scene_graph"])
        tuples = [AtomicTuple.model_validate(item) for item in row["atomic_tuples"]]
        dependencies = [
            DependencyRecord.model_validate(item) for item in row["dependencies"]
        ]
        fields = set(RubricQuestion.model_fields)
        questions = [
            RubricQuestion.model_validate({key: value for key, value in item.items() if key in fields})
            for item in row["questions"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        return [f"schema:{error}"]
    errors = validate_scene_graph(row["caption"], graph)
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
    return errors + validate_dependencies(tuples, dependencies) + validate_questions(
        row["caption"], tuples, dependencies, questions,
        language_override=row["language"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    errors: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    plan = read_jsonl(args.plan)
    prompt_jobs = read_jsonl(run / "prepared/prompt_jobs.jsonl")
    audio_recipes = read_jsonl(run / "prepared/audio_recipes.jsonl")
    plan_count = len(plan)
    planned_branch_counts = Counter(row["branch"] for row in plan)
    planned_categories = Counter(row["perturbation_category"] for row in plan)
    expected_prompt_count = planned_branch_counts["prompt_side"]
    expected_audio_count = planned_branch_counts["audio_side"]
    require(plan_count > 0, "plan_is_empty")
    require(len(prompt_jobs) == expected_prompt_count, f"prompt_jobs:{len(prompt_jobs)}/{expected_prompt_count}")
    require(len(audio_recipes) == expected_audio_count, f"audio_recipes:{len(audio_recipes)}/{expected_audio_count}")
    plan_ids = [row["negative_id"] for row in plan]
    parent_ids = [row["parent_positive_id"] for row in plan]
    require(len(set(plan_ids)) == plan_count, "plan_negative_ids_not_unique")
    require(len(set(parent_ids)) == plan_count, "parent_positives_not_exactly_once")
    require(set(planned_categories) == {"event", "attribute", "relation", "all_three"},
            f"plan_category_coverage:{dict(planned_categories)}")
    positive_rubrics = read_jsonl(run / "positive_construction_labels.jsonl")
    positive_schema_failures = [
        (row.get("sample_id"), canonical_rubric_errors(row))
        for row in positive_rubrics if canonical_rubric_errors(row)
    ]
    require(not positive_schema_failures,
            f"positive_canonical_rubric_failures:{len(positive_schema_failures)}")
    for branch in ("prompt_side", "audio_side"):
        branch_rows = [row for row in plan if row["branch"] == branch]
        require(len(branch_rows) == planned_branch_counts[branch],
                f"{branch}_plan_rows:{len(branch_rows)}")

    speech_pool = read_jsonl(run / "assets/speech_pool.jsonl")
    nonspeech_pool = read_jsonl(run / "assets/nonspeech_pool.jsonl")
    speech_paths = [str(row.get("audio_path") or row.get("source_path")) for row in speech_pool]
    nonspeech_paths = [str(row.get("source_path") or row.get("audio_path")) for row in nonspeech_pool]
    require(len(set(speech_paths)) == len(speech_paths), "speech_pool_paths_not_unique")
    require(len(set(nonspeech_paths)) == len(nonspeech_paths), "nonspeech_pool_paths_not_unique")
    require(not any("instructttseval" in json.dumps(row, ensure_ascii=False).casefold()
                    for row in speech_pool + nonspeech_pool), "instructttseval_in_replacement_pool")

    # Prompt-side generation: select the last successful row just as the freezer does.
    authoritative_path = run / "prompt_side/generated_authoritative.jsonl"
    generated_source = authoritative_path if authoritative_path.exists() else run / "prompt_side/generated.jsonl"
    generated_ok: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(generated_source):
        if row.get("status") == "ok":
            current = generated_ok.get(row["sample_id"])
            deterministic_models = {
                "deterministic_relation_reversal", "deterministic_missing_job_seed",
            }
            if current and current.get("model") in deterministic_models \
                    and row.get("model") not in deterministic_models:
                continue
            generated_ok[row["sample_id"]] = row
    jobs = {row["sample_id"]: row for row in prompt_jobs}
    require(set(generated_ok) == set(jobs), f"prompt_generated_id_set:{len(generated_ok)}/{len(jobs)}")
    prompt_patch_failures = []
    for sample_id, job in jobs.items():
        row = generated_ok.get(sample_id)
        if row is None:
            continue
        source = job["prompt"]
        patches = sorted(row.get("patches") or [], key=lambda item: item["start"])
        claimed = [rid for patch in patches for rid in patch.get("changed_rubric_ids") or []]
        local_errors = []
        if row.get("prompt") != source or row.get("prompt_prime") == source:
            local_errors.append("prompt_identity")
        if set(claimed) != set(job["selected_rubric_ids"]) or len(claimed) != len(set(claimed)):
            local_errors.append("rubric_claim_set")
        if any(left["end"] > right["start"] for left, right in zip(patches, patches[1:])):
            local_errors.append("overlapping_patches")
        pieces, cursor = [], 0
        for patch in patches:
            start, end = int(patch["start"]), int(patch["end"])
            if source[start:end] != patch["original_span"] or patch["original_span"] == patch["replacement_span"]:
                local_errors.append("invalid_exact_span")
            pieces.extend((source[cursor:start], patch["replacement_span"]))
            cursor = end
        pieces.append(source[cursor:])
        if "".join(pieces) != row.get("prompt_prime"):
            local_errors.append("reconstruction_mismatch")
        audio_path = Path(job["audio_path"])
        if not audio_path.exists() or sha256(audio_path) != job["parent_audio_sha256"]:
            local_errors.append("parent_audio_hash")
        if local_errors:
            prompt_patch_failures.append({"sample_id": sample_id, "errors": sorted(set(local_errors))})
    require(not prompt_patch_failures, f"prompt_patch_failures:{len(prompt_patch_failures)}")

    prompt_pairs = read_jsonl(run / "prompt_side/pairs.jsonl")
    prompt_pair_ids = {row["negative_id"] for row in prompt_pairs}
    require(len(prompt_pairs) == expected_prompt_count and prompt_pair_ids == set(jobs),
            f"prompt_pairs:{len(prompt_pairs)}")
    require(all(row.get("audio_byte_identical_to_parent") is True for row in prompt_pairs),
            "prompt_audio_not_byte_identical")
    require(all(row.get("outside_patch_equality") is True for row in prompt_pairs),
            "prompt_outside_patch_not_equal")
    semantic_audits = read_jsonl(run / "prompt_side/semantic_repair_audit.jsonl")
    require(
        len(semantic_audits) == expected_prompt_count
        and {row.get("sample_id") for row in semantic_audits} == prompt_pair_ids
        and all(row.get("pass") is True for row in semantic_audits),
        f"prompt_semantic_direct_target_audit:{len(semantic_audits)}",
    )
    all_three_ids = {
        row["sample_id"] for row in prompt_jobs
        if row["path2_perturbation_category"] == "all_three"
    }
    relation_realization_ids = {
        row.get("sample_id") for row in semantic_audits
        if row.get("selected_relation_realization") in {
            "false_via_selected_endpoint_identity_removal",
            "explicit_local_relation_reversal",
        }
    }
    require(
        relation_realization_ids == all_three_ids,
        f"prompt_all_three_relation_realization:{len(relation_realization_ids)}/{len(all_three_ids)}",
    )

    pprime = read_jsonl(run / "prompt_side/construction_labels.jsonl")
    pprime_ids = {row.get("sample_id") for row in pprime}
    require(len(pprime) == expected_prompt_count and pprime_ids == prompt_pair_ids,
            f"pprime_rubrics:{len(pprime)}")
    require(all((row.get("validation") or {}).get("valid") is True for row in pprime),
            "pprime_contains_invalid_rows")
    require(all(bool(row.get("questions")) for row in pprime), "pprime_contains_empty_questions")
    require(all(row.get("rubric_count") == len(row.get("questions") or []) for row in pprime),
            "pprime_rubric_count_mismatch")
    require(all(row.get("construction_label_method") ==
                "deterministic_transform_of_validated_scene_and_label_graph" for row in pprime),
            "pprime_unexpected_decomposition_method")
    require(all((row.get("validation") or {}).get("all_parent_ids_resolve") is True for row in pprime),
            "pprime_unresolved_parent_ids")
    pprime_schema_failures = [
        (row.get("sample_id"), canonical_rubric_errors(row))
        for row in pprime if canonical_rubric_errors(row)
    ]
    require(not pprime_schema_failures,
            f"pprime_canonical_rubric_failures:{len(pprime_schema_failures)}")
    pprime_question_count = sum(len(row.get("questions") or []) for row in pprime)
    require(pprime_question_count == sum(row.get("rubric_count", 0) for row in pprime),
            f"pprime_question_count:{pprime_question_count}")

    # Audio-side render and deterministic semantic labels.
    validation = read_jsonl(run / "audio_side/mix/validation_report.jsonl")
    require(len(validation) == expected_audio_count and all(row.get("valid") is True for row in validation),
            f"audio_validation:{sum(row.get('valid') is True for row in validation)}/{expected_audio_count}")
    audio_pairs = read_jsonl(run / "audio_side/pairs.jsonl")
    audio_pair_ids = {row["negative_id"] for row in audio_pairs}
    recipe_ids = {row["negative_id"] for row in audio_recipes}
    require(len(audio_pairs) == expected_audio_count and audio_pair_ids == recipe_ids,
            f"audio_pairs:{len(audio_pairs)}")
    require(len({row.get("audio_sha256") for row in audio_pairs}) == expected_audio_count,
            "audio_hashes_not_unique")
    missing_audio = []
    hash_mismatch = []
    for row in audio_pairs:
        path = Path(row["audio_path"])
        if not path.exists():
            missing_audio.append(row["negative_id"])
        elif sha256(path) != row.get("audio_sha256"):
            hash_mismatch.append(row["negative_id"])
    require(not missing_audio, f"missing_audio:{len(missing_audio)}")
    require(not hash_mismatch, f"audio_hash_mismatch:{len(hash_mismatch)}")
    require(not any("instructttseval" in json.dumps(row.get("events") or [], ensure_ascii=False).casefold()
                    for row in audio_pairs), "instructttseval_in_audio_pairs")
    role_metadata_failures = []
    for row in audio_pairs:
        for event in row.get("events") or []:
            replacement_role = event.get("replacement_source_role")
            speech_only_values = [
                event.get("text"), event.get("speaker"), event.get("speaker_description"),
                event.get("style_prompt"), event.get("selected_caption_attrs"),
                event.get("primary_attrs"), event.get("all_source_attrs"),
                event.get("annotation_format"),
            ]
            if event.get("role") != "speech" and any(value not in (None, [], {}) for value in speech_only_values):
                role_metadata_failures.append((row["negative_id"], event.get("event_id"), "stale_speech_metadata"))
            if event.get("mixing_role") == "speech" and event.get("role") != replacement_role:
                role_metadata_failures.append((row["negative_id"], event.get("event_id"), "replacement_role_mismatch"))
    require(not role_metadata_failures, f"audio_role_metadata_failures:{len(role_metadata_failures)}")

    audio_rubrics = read_jsonl(run / "audio_side/construction_labels.jsonl")
    require(len(audio_rubrics) == expected_audio_count and {row["sample_id"] for row in audio_rubrics} == recipe_ids,
            f"audio_rubrics:{len(audio_rubrics)}")
    audio_schema_failures = [
        (row.get("sample_id"), canonical_rubric_errors(row))
        for row in audio_rubrics if canonical_rubric_errors(row)
    ]
    require(not audio_schema_failures,
            f"audio_canonical_rubric_failures:{len(audio_schema_failures)}")
    recipe_map = {row["negative_id"]: row for row in audio_recipes}
    rubric_label_failures = []
    for row in audio_rubrics:
        recipe = recipe_map[row["sample_id"]]
        false_ids = set(recipe["hierarchy_affected_rubric_ids"])
        questions = row.get("questions") or []
        if len(questions) != row.get("rubric_count"):
            rubric_label_failures.append((row["sample_id"], "rubric_count"))
            continue
        for question in questions:
            expected = "no" if question["rubric_id"] in false_ids else "yes"
            if question.get("construction_answer") != expected or not question.get("observed_audio_evidence"):
                rubric_label_failures.append((row["sample_id"], question["rubric_id"]))
                break
    require(not rubric_label_failures, f"audio_rubric_label_failures:{len(rubric_label_failures)}")

    audio_summary_path = run / "audio_side/summary.json"
    audio_summary = json.loads(audio_summary_path.read_text(encoding="utf-8")) if audio_summary_path.exists() else {}
    require(audio_summary.get("semantic_realization_failures") == 0,
            f"audio_semantic_failures:{audio_summary.get('semantic_realization_failures')}")
    if audio_summary.get("target_band_fail", 0):
        warnings.append(
            "The intended level is a sampling stratum; use final_deterministic_consistency_level as "
            f"authority. Outside intended bands: {audio_summary['target_band_fail']}."
        )

    combined_ids = prompt_pair_ids | audio_pair_ids
    require(len(combined_ids) == plan_count and not (prompt_pair_ids & audio_pair_ids),
            f"combined_negative_ids:{len(combined_ids)}")
    combined_categories = Counter(row["perturbation_category"] for row in prompt_pairs + audio_pairs)
    require(dict(combined_categories) == dict(planned_categories),
            f"combined_category_distribution:{dict(combined_categories)}")

    report = {
        "schema": "route2_negative_final_audit",
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "plan": len(plan), "prompt_pairs": len(prompt_pairs), "audio_pairs": len(audio_pairs),
            "prompt_construction_labels": len(pprime),
            "audio_construction_labels": len(audio_rubrics),
            "pprime_questions": pprime_question_count,
            "audio_valid": sum(row.get("valid") is True for row in validation),
        },
        "categories": dict(combined_categories),
        "audio_levels": {
            "intended": audio_summary.get("intended_levels", {}),
            "observed": audio_summary.get("final_levels", {}),
            "target_band_pass": audio_summary.get("target_band_pass"),
            "target_band_fail": audio_summary.get("target_band_fail"),
        },
        "prompt_assurance": {
            "exact_nonoverlapping_patches": len(prompt_patch_failures) == 0,
            "audio_byte_identity": all(row.get("audio_byte_identical_to_parent") is True for row in prompt_pairs),
            "prompt_construction_labels_regenerated": len(pprime) == expected_prompt_count,
            "pprime_construction_label_method": "deterministic_transform_of_validated_scene_and_label_graph",
            "direct_selected_facts_audited": len(semantic_audits) == expected_prompt_count and all(
                row.get("pass") is True for row in semantic_audits
            ),
            "perceptual_labels": "not_run_by_design",
        },
        "audio_assurance": {
            "signal_and_timeline_valid": len(validation) == expected_audio_count and all(row.get("valid") is True for row in validation),
            "direct_targets_and_retained_facts_audited": audio_summary.get("semantic_realization_failures") == 0,
            "replacement_roles_and_type_metadata_consistent": not role_metadata_failures,
            "unique_audio_hashes": len({row.get("audio_sha256") for row in audio_pairs}),
        },
        "artifacts": {
            "prompt_pairs": str(run / "prompt_side/pairs.jsonl"),
            "prompt_construction_labels": str(run / "prompt_side/construction_labels.jsonl"),
            "audio_pairs": str(run / "audio_side/pairs.jsonl"),
            "audio_construction_labels": str(run / "audio_side/construction_labels.jsonl"),
        },
        "hashes": {
            "plan": sha256(args.plan) if args.plan.exists() else None,
            "prompt_pairs": sha256(run / "prompt_side/pairs.jsonl") if (run / "prompt_side/pairs.jsonl").exists() else None,
            "prompt_construction_labels": sha256(run / "prompt_side/construction_labels.jsonl") if (run / "prompt_side/construction_labels.jsonl").exists() else None,
            "audio_pairs": sha256(run / "audio_side/pairs.jsonl") if (run / "audio_side/pairs.jsonl").exists() else None,
            "audio_construction_labels": sha256(run / "audio_side/construction_labels.jsonl") if (run / "audio_side/construction_labels.jsonl").exists() else None,
        },
    }
    output = run / "final_audit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
