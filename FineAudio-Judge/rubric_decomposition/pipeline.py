from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from .chunking import pack_question_chunks
from .io import append_jsonl, read_jsonl, write_jsonl
from .llm_client import VllmJsonClient
from .models import (AtomicTuple, AudioSceneGraph, DependencyRecord,
                     GeneratedQuestion, QuestionResponse, Relation,
                     RelationReviewResponse, RelationVerificationResponse,
                     SceneGraphResponse)
from .prompts import (SYSTEM, deterministic_question, event_label_map,
                      question_prompt, question_tasks, relation_review_prompt,
                      relation_verification_prompt, scene_graph_prompt,
                      scene_graph_review_prompt)
from .tuples import build_tuples
from .validation import (build_dependencies, canonicalize_questions,
                         canonicalize_scene_graph, validate_dependencies,
                         validate_questions, validate_scene_graph)

PIPELINE_NAME = "rubric_decomposition"

WARNING_PREFIXES: tuple[str, ...] = ()


def _validate_runtime_args(args: argparse.Namespace) -> None:
    for name in ("base_url", "model", "tokenizer"):
        value = getattr(args, name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    positive_values = {
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "scene_graph_tokens": args.scene_graph_tokens,
        "context_window": args.context_window,
        "question_tokens": args.question_tokens,
        "question_output_tokens_per_question": args.question_output_tokens_per_question,
        "question_safety_margin": args.question_safety_margin,
        "question_max_tuples_per_chunk": args.question_max_tuples_per_chunk,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if args.question_tokens + args.question_safety_margin >= args.context_window:
        raise ValueError(
            "question_tokens plus question_safety_margin must be smaller than context_window"
        )


def _validate_manifest(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest must contain at least one row")
    sample_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        for field in ("sample_id", "route", "prompt"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"manifest row {index} requires non-empty string field {field!r}")
        sample_id = row["sample_id"].strip()
        if sample_id in sample_ids:
            raise ValueError(f"manifest contains duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)


def _write_run_config(
    path: Path, args: argparse.Namespace, client: VllmJsonClient
) -> None:
    manifest_digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    config = {
        "pipeline_name": PIPELINE_NAME,
        "backend_contract": "qwen_vllm_openai_chat_json_object",
        "manifest_sha256": manifest_digest,
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "temperature": client.temperature,
        "seed": client.seed,
        "thinking_enabled": False,
        "llm_call_audit": "llm_call_audit.jsonl",
        "scene_graph_tokens": args.scene_graph_tokens,
        "context_window": args.context_window,
        "question_tokens": args.question_tokens,
        "question_output_tokens_per_question": args.question_output_tokens_per_question,
        "question_safety_margin": args.question_safety_margin,
        "question_max_tuples_per_chunk": args.question_max_tuples_per_chunk,
        "stop_after_relation_audit": args.stop_after_relation_audit,
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hard_errors(errors: list[str]) -> list[str]:
    return [item for item in errors if not item.startswith(WARNING_PREFIXES)]


_VALIDATION_MESSAGES = {
    "duplicate_ids": "Record IDs must be unique.",
    "duplicate_event_labels": "Event labels must be textually unique.",
    "explicit_silence_coverage": "Represent every explicitly stated silence or pause.",
    "unknown_attribute_owner": "The attribute owner must reference an existing Event or SCENE.",
    "unknown_relation_endpoint": "Both relation endpoints must reference existing Events.",
    "basis_not_verbatim": "The evidence must be one continuous verbatim span from the audio prompt.",
    "event_label_language_mismatch": "The Event label must use the requested output language.",
    "attribute_value_language_mismatch": "The Attribute value must use the requested output language.",
    "event_label_attribute_overlap": "Keep descriptive properties in Attributes rather than duplicating them in Event labels.",
    "speech_content_owner_not_speech": "A speech-content Attribute must belong to a speech Event.",
    "quoted_speech_missing": "Represent every explicitly quoted utterance exactly once.",
    "quoted_speech_duplicate": "Represent every explicitly quoted utterance exactly once.",
    "self_relation": "A Relation must connect two distinct Events.",
    "before_relation_cycle": "Temporal before Relations must not form a cycle.",
    "event_prompt_order_mismatch": "Assign Event IDs in audio-prompt mention order.",
    "relation_audit_noncanonical_after": "Use the canonical before form for sequential Relations.",
    "relation_decision_coverage_or_order_mismatch": "Return exactly one decision for every candidate, preserving candidate order.",
    "invalid_temporal_verdict": "Use one of the permitted temporal verdicts.",
    "invalid_non_temporal_verdict": "Use one of the permitted non-temporal verdicts.",
    "question_generation_structure": "Return one well-formed question entry for every requested task.",
    "question_count_mismatch": "Return exactly one question for every task.",
    "question_tuple_coverage_mismatch": "Return exactly one question for every supplied tuple ID.",
    "rubric_id_mismatch": "Preserve the required rubric ID assignment.",
    "tuple_metadata_mismatch": "Preserve the metadata associated with the task.",
    "question_basis_mismatch": "Preserve the task's evidence span exactly.",
    "target_ids_mismatch": "Preserve the task's target IDs exactly.",
    "question_parents_mismatch": "Preserve the required question dependencies.",
    "not_a_question": "End the wording with the appropriate question mark.",
    "question_language_mismatch": "Write the question in the requested output language.",
    "question_mixed_language": "Do not mix languages except for verbatim content from the audio prompt.",
    "unbalanced_curly_quotes": "Use balanced curly quotation marks.",
    "unbalanced_double_quotes": "Use balanced double quotation marks.",
    "question_exposes_graph_id": "Do not expose internal graph IDs in the question.",
    "presence_question_missing_event_label": "Include the supplied Event label in the presence question.",
    "presence_question_attribute_overlap": "A presence question must not include an Attribute from the Event.",
    "question_exposes_attribute_type": "Express the property naturally instead of exposing its schema field name.",
    "attribute_question_missing_value": "Include the supplied Attribute value.",
    "attribute_question_missing_unit": "Include the supplied unit exactly once.",
    "attribute_question_sibling_overlap": "Ask only about the requested Attribute, not a sibling Attribute.",
    "relation_question_attribute_overlap": "Ask only about the requested Relation, not an endpoint Attribute.",
}


def _format_validation_feedback(errors: list[str]) -> str:
    """Render internal validation findings as stable, model-readable feedback."""
    feedback: list[dict[str, Any]] = []
    for error in errors:
        code, *context = error.split(":")
        item: dict[str, Any] = {
            "issue": _VALIDATION_MESSAGES.get(
                code, code.replace("_", " ").capitalize() + "."
            )
        }
        if context:
            item["context"] = context
        feedback.append(item)
    return json.dumps(feedback, ensure_ascii=False, indent=2)


def _normalize_relation_verdict(relation_type: str, verdict: str) -> str:
    """Map semantically equivalent direction words across verdict subtypes.

    Structured-output models sometimes answer with the correct endpoint
    orientation but choose the temporal/non-temporal spelling of that
    orientation. This normalization is local to one relation and never creates
    a relation that the verifier marked unsupported.
    """
    temporal = relation_type in {"before", "after", "overlap"}
    if temporal:
        if verdict == "keep_non_temporal":
            return "overlap" if relation_type == "overlap" else "source_before_target"
        if verdict == "reverse_non_temporal":
            return "overlap" if relation_type == "overlap" else "target_before_source"
        return verdict
    if verdict == "source_before_target":
        return "keep_non_temporal"
    if verdict == "target_before_source":
        return "reverse_non_temporal"
    if verdict == "overlap":
        if relation_type in {
            "same_source", "different_source", "same_speaker", "different_speaker"
        }:
            return "keep_non_temporal"
        return "unsupported"
    return verdict


async def _run_pending(rows: list[dict[str, Any]], output: Path, concurrency: int,
                       worker: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]) -> None:
    # Only successful attempts are checkpoints. An error row is retained for
    # auditability and retried automatically on the next invocation.
    done = {row["sample_id"] for row in read_jsonl(output)
            if row.get("status") == "ok"}
    pending = [row for row in rows if row["sample_id"] not in done]
    lock = asyncio.Lock()

    async def one(row: dict[str, Any]) -> None:
        try:
            value = await worker(row)
        except Exception as exc:
            value = {"sample_id": row["sample_id"], "status": "error", "error": repr(exc)}
        async with lock:
            append_jsonl(output, value)

    for start in range(0, len(pending), max(concurrency * 4, 1)):
        await asyncio.gather(*(one(row) for row in pending[start:start + concurrency * 4]))


def _load_chat_token_counter(tokenizer_path: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def count(user_prompt: str) -> int:
        encoded = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return len(encoded)

    return count


async def run(args: argparse.Namespace) -> None:
    _validate_runtime_args(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(args.manifest)
    _validate_manifest(manifest)
    client = VllmJsonClient(base_url=args.base_url, model=args.model,
                            concurrency=args.concurrency, timeout=args.timeout,
                            temperature=getattr(args, "temperature", None),
                            seed=getattr(args, "seed", None),
                            audit_path=output / "llm_call_audit.jsonl")
    _write_run_config(output / "run_config.json", args, client)
    count_tokens = _load_chat_token_counter(args.tokenizer)

    stage1_path = output / "stage1_scene_graph.jsonl"
    async def stage1(row: dict[str, Any]) -> dict[str, Any]:
        base_prompt = scene_graph_prompt(row["prompt"])
        instructions: list[dict[str, Any]] = [
            {"stage": "stage1_scene_graph", "attempt": 1,
             "system": SYSTEM, "user": base_prompt}
        ]
        parsed, raw = await client.generate(
            system=SYSTEM, prompt=base_prompt, schema=SceneGraphResponse,
            max_tokens=args.scene_graph_tokens,
            audit_context={"sample_id": row["sample_id"],
                           "stage": "stage1_scene_graph", "stage_attempt": 1},
        )
        parsed.scene_graph = canonicalize_scene_graph(row["prompt"], parsed.scene_graph)
        candidate = parsed.scene_graph.model_dump()
        review_errors = _hard_errors(validate_scene_graph(row["prompt"], parsed.scene_graph))

        # A separate semantic audit catches ownership and relation mistakes
        # that a generator can overlook while constructing a large graph.
        for attempt in range(1, 6):
            user_prompt = scene_graph_review_prompt(row["prompt"], candidate)
            if review_errors:
                user_prompt += (
                    "\n\nVALIDATION FEEDBACK FROM THE PREVIOUS ATTEMPT:\n"
                    f"{_format_validation_feedback(review_errors)}"
                )
            instructions.append({"stage": "stage1_semantic_review", "attempt": attempt,
                                 "system": SYSTEM, "user": user_prompt})
            parsed, raw = await client.generate(
                system=SYSTEM, prompt=user_prompt, schema=SceneGraphResponse,
                max_tokens=args.scene_graph_tokens,
                audit_context={"sample_id": row["sample_id"],
                               "stage": "stage1_semantic_review",
                               "stage_attempt": attempt},
            )
            parsed.scene_graph = canonicalize_scene_graph(row["prompt"], parsed.scene_graph)
            candidate = parsed.scene_graph.model_dump()
            errors = validate_scene_graph(row["prompt"], parsed.scene_graph)
            review_errors = _hard_errors(errors)
            if not review_errors:
                return {
                    "sample_id": row["sample_id"], "status": "ok",
                    "scene_graph": candidate,
                    "validation_errors": errors, "raw_model_output": raw,
                    "llm_instructions": instructions,
                }
        return {
            "sample_id": row["sample_id"], "status": "error",
            "error": "scene_graph_review_hard_validation:" + ",".join(review_errors),
            "last_candidate": candidate,
            "llm_instructions": instructions,
        }

    # A newer deterministic canonicalizer may repair a previously retained
    # candidate without another model call. Promote only candidates that pass
    # the complete current hard-validation suite.
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    retained_by_id: dict[str, dict[str, Any]] = {}
    for retained in read_jsonl(stage1_path):
        if retained.get("last_candidate"):
            retained_by_id[retained["sample_id"]] = retained
    for sid, retained in retained_by_id.items():
        prompt = manifest_by_id[sid]["prompt"]
        graph = canonicalize_scene_graph(
            prompt, AudioSceneGraph.model_validate(retained["last_candidate"])
        )
        errors = validate_scene_graph(prompt, graph)
        if not _hard_errors(errors):
            append_jsonl(stage1_path, {
                "sample_id": sid, "status": "ok",
                "scene_graph": graph.model_dump(),
                "validation_errors": errors,
                "raw_model_output": json.dumps(retained["last_candidate"], ensure_ascii=False),
                "llm_instructions": retained.get("llm_instructions", []),
                "recovered_from_last_candidate": True,
            })
    # Re-apply the current deterministic canonicalizer to successful
    # checkpoints as well. This lets newly added completeness repairs upgrade
    # an earlier valid-looking graph while preserving its actual LLM calls.
    successful_by_id = {
        row["sample_id"]: row for row in read_jsonl(stage1_path)
        if row.get("status") == "ok" and row.get("scene_graph")
    }
    for sid, checkpoint in successful_by_id.items():
        prompt = manifest_by_id[sid]["prompt"]
        graph = canonicalize_scene_graph(
            prompt, AudioSceneGraph.model_validate(checkpoint["scene_graph"])
        )
        candidate = graph.model_dump()
        errors = validate_scene_graph(prompt, graph)
        if candidate != checkpoint["scene_graph"] and not _hard_errors(errors):
            append_jsonl(stage1_path, {
                **checkpoint,
                "scene_graph": candidate,
                "validation_errors": errors,
                "recanonicalized_checkpoint": True,
            })
    await _run_pending(manifest, stage1_path, args.concurrency, stage1)

    stage1_rows = {row["sample_id"]: row for row in read_jsonl(stage1_path) if row.get("status") == "ok"}

    relation_path = output / "stage1b_relation_audit.jsonl"
    relation_inputs = [row for row in manifest if row["sample_id"] in stage1_rows]

    async def relation_audit(row: dict[str, Any]) -> dict[str, Any]:
        base = stage1_rows[row["sample_id"]]
        graph = AudioSceneGraph.model_validate(base["scene_graph"])
        candidate_relations = [item.model_dump() for item in graph.relations]
        instructions: list[dict[str, Any]] = []
        review_errors: list[str] = []
        raw = ""
        for attempt in range(1, 4):
            user_prompt = relation_review_prompt(
                row["prompt"],
                [item.model_dump() for item in graph.events],
                candidate_relations,
            )
            if review_errors:
                user_prompt += (
                    "\n\nVALIDATION FEEDBACK FROM THE PREVIOUS ATTEMPT:\n"
                    f"{_format_validation_feedback(review_errors)}"
                )
            instructions.append({
                "stage": "stage1b_relation_audit", "attempt": attempt,
                "system": SYSTEM, "user": user_prompt,
            })
            parsed, raw = await client.generate(
                system=SYSTEM, prompt=user_prompt,
                schema=RelationReviewResponse,
                max_tokens=min(args.scene_graph_tokens, 8192),
                audit_context={"sample_id": row["sample_id"],
                               "stage": "stage1b_relation_audit",
                               "stage_attempt": attempt},
            )
            graph.relations = parsed.relations
            graph = canonicalize_scene_graph(row["prompt"], graph)
            candidate_relations = [item.model_dump() for item in graph.relations]
            review_errors = _hard_errors(validate_scene_graph(row["prompt"], graph))
            review_errors.extend(
                f"relation_audit_noncanonical_after:{item.id}"
                for item in graph.relations if item.type == "after"
            )
            if not review_errors:
                return {
                    "sample_id": row["sample_id"], "status": "ok",
                    "scene_graph": graph.model_dump(),
                    "validation_errors": [], "raw_model_output": raw,
                    "llm_instructions": instructions,
                }
        return {
            "sample_id": row["sample_id"], "status": "error",
            "error": "relation_audit_hard_validation:" + ",".join(review_errors),
            "last_relations": candidate_relations,
            "llm_instructions": instructions,
        }

    await _run_pending(relation_inputs, relation_path, args.concurrency, relation_audit)
    relation_rows = {
        row["sample_id"]: row for row in read_jsonl(relation_path)
        if row.get("status") == "ok" and row.get("scene_graph")
    }
    verification_path = output / "stage1c_relation_verification.jsonl"
    verification_inputs = [row for row in manifest if row["sample_id"] in relation_rows]

    async def verify_relations(row: dict[str, Any]) -> dict[str, Any]:
        audited = relation_rows[row["sample_id"]]
        graph = AudioSceneGraph.model_validate(audited["scene_graph"])
        candidates = [item.model_dump() for item in graph.relations]
        candidate_by_id = {item["id"]: item for item in candidates}
        expected_ids = [item["id"] for item in candidates]
        instructions: list[dict[str, Any]] = []
        prior_errors: list[str] = []
        raw = ""
        for attempt in range(1, 4):
            user_prompt = relation_verification_prompt(
                row["prompt"],
                [item.model_dump() for item in graph.events],
                candidates,
            )
            if prior_errors:
                user_prompt += (
                    "\n\nVALIDATION FEEDBACK FROM THE PREVIOUS ATTEMPT:\n"
                    f"{_format_validation_feedback(prior_errors)}"
                )
            instructions.append({
                "stage": "stage1c_relation_verification", "attempt": attempt,
                "system": SYSTEM, "user": user_prompt,
            })
            parsed, raw = await client.generate(
                system=SYSTEM, prompt=user_prompt,
                schema=RelationVerificationResponse,
                max_tokens=min(args.scene_graph_tokens, 8192),
                audit_context={"sample_id": row["sample_id"],
                               "stage": "stage1c_relation_verification",
                               "stage_attempt": attempt},
            )
            decision_ids = [item.candidate_id for item in parsed.decisions]
            prior_errors = []
            if decision_ids != expected_ids:
                prior_errors.append("relation_decision_coverage_or_order_mismatch")
                continue

            materialized: list[Relation] = []
            for decision in parsed.decisions:
                source = candidate_by_id[decision.candidate_id]
                verdict = _normalize_relation_verdict(source["type"], decision.verdict)
                temporal = source["type"] in {"before", "after", "overlap"}
                if verdict == "unsupported":
                    continue
                if temporal and verdict not in {
                    "source_before_target", "target_before_source", "overlap"
                }:
                    prior_errors.append(
                        f"invalid_temporal_verdict:{decision.candidate_id}:{verdict}"
                    )
                    continue
                if not temporal and verdict not in {
                    "keep_non_temporal", "reverse_non_temporal"
                }:
                    prior_errors.append(
                        f"invalid_non_temporal_verdict:{decision.candidate_id}:{verdict}"
                    )
                    continue
                source_id, target_id = source["source_id"], source["target_id"]
                relation_type = source["type"]
                if verdict == "target_before_source":
                    source_id, target_id = target_id, source_id
                    relation_type = "before"
                elif verdict == "source_before_target":
                    relation_type = "before"
                elif verdict == "overlap":
                    relation_type = "overlap"
                elif verdict == "reverse_non_temporal":
                    source_id, target_id = target_id, source_id
                materialized.append(Relation(
                    id=f"REL{len(materialized) + 1}",
                    type=relation_type,
                    source_id=source_id,
                    target_id=target_id,
                    basis=source["basis"],
                ))
            if prior_errors:
                continue
            graph.relations = materialized
            graph = canonicalize_scene_graph(row["prompt"], graph)
            prior_errors = _hard_errors(validate_scene_graph(row["prompt"], graph))
            if not prior_errors:
                return {
                    "sample_id": row["sample_id"], "status": "ok",
                    "scene_graph": graph.model_dump(),
                    "validation_errors": [], "raw_model_output": raw,
                    "llm_instructions": instructions,
                    "relation_decisions": [item.model_dump() for item in parsed.decisions],
                }
        return {
            "sample_id": row["sample_id"], "status": "error",
            "error": "relation_verification_hard_validation:" + ",".join(prior_errors),
            "llm_instructions": instructions,
        }

    await _run_pending(
        verification_inputs, verification_path, args.concurrency, verify_relations
    )
    verified_rows = {
        row["sample_id"]: row for row in read_jsonl(verification_path)
        if row.get("status") == "ok" and row.get("scene_graph")
    }
    for sid, checkpoint in list(verified_rows.items()):
        prompt = manifest_by_id[sid]["prompt"]
        graph = canonicalize_scene_graph(
            prompt, AudioSceneGraph.model_validate(checkpoint["scene_graph"])
        )
        candidate = graph.model_dump()
        errors = validate_scene_graph(prompt, graph)
        if candidate != checkpoint["scene_graph"] and not _hard_errors(errors):
            updated = {
                **checkpoint,
                "scene_graph": candidate,
                "validation_errors": errors,
                "recanonicalized_checkpoint": True,
            }
            append_jsonl(verification_path, updated)
            verified_rows[sid] = updated
    if args.stop_after_relation_audit:
        print(json.dumps({
            "pipeline_name": PIPELINE_NAME,
            "samples": len(manifest),
            "relation_audited": len(relation_rows),
            "relation_verified": len(verified_rows),
        }, ensure_ascii=False))
        return
    tuple_rows: list[dict[str, Any]] = []
    for row in manifest:
        stage = verified_rows.get(row["sample_id"])
        if not stage: continue
        graph = AudioSceneGraph.model_validate(stage["scene_graph"])
        tuple_rows.append({"sample_id": row["sample_id"], "status": "ok",
                           "tuples": [item.model_dump() for item in build_tuples(graph)]})
    write_jsonl(output / "stage2_atomic_tuples.jsonl", tuple_rows)
    tuple_map = {row["sample_id"]: row for row in tuple_rows}

    dep_path = output / "stage3_dependencies.jsonl"
    dep_inputs = [row for row in manifest if row["sample_id"] in tuple_map]
    dependency_rows: list[dict[str, Any]] = []
    for row in dep_inputs:
        try:
            tuple_models = [AtomicTuple.model_validate(item) for item in tuple_map[row["sample_id"]]["tuples"]]
            dependencies = build_dependencies(tuple_models)
            dependency_rows.append({
                "sample_id": row["sample_id"],
                "status": "ok",
                "dependency_strategy": "deterministic_event_presence",
                "dependencies": [item.model_dump() for item in dependencies],
                "validation_errors": validate_dependencies(tuple_models, dependencies),
            })
        except Exception as exc:
            dependency_rows.append({"sample_id": row["sample_id"], "status": "error", "error": repr(exc)})
    write_jsonl(dep_path, dependency_rows)

    dep_rows = {row["sample_id"]: row for row in read_jsonl(dep_path) if row.get("status") == "ok"}
    question_path = output / "stage4_questions.jsonl"
    question_inputs = [row for row in manifest if row["sample_id"] in tuple_map and row["sample_id"] in dep_rows]
    async def stage4(row: dict[str, Any]) -> dict[str, Any]:
        tuples = tuple_map[row["sample_id"]]["tuples"]
        deps = dep_rows[row["sample_id"]]["dependencies"]
        chunk_plans = pack_question_chunks(
            prompt=row["prompt"],
            tuples=tuples,
            dependencies=deps,
            count_tokens=count_tokens,
            context_window=args.context_window,
            max_output_tokens=args.question_tokens,
            output_tokens_per_question=args.question_output_tokens_per_question,
            safety_margin=args.question_safety_margin,
            max_tuples_per_chunk=args.question_max_tuples_per_chunk,
            target_language_override=row.get("language"),
        )

        async def generate_chunk(
            plan: dict[str, Any], attempt: int, prior_errors: list[str]
        ) -> tuple[QuestionResponse, str, str]:
            user_prompt = question_prompt(plan["target_language"], plan["question_tasks"])
            if prior_errors:
                relevant_ids = {item["tuple_id"] for item in plan["tuples"]}
                relevant_errors = [
                    item for item in prior_errors
                    if any(tuple_id in item for tuple_id in relevant_ids)
                ]
                if relevant_errors:
                    user_prompt += (
                        "\n\nVALIDATION FEEDBACK FROM THE PREVIOUS ATTEMPT:\n"
                        "The previous output did not satisfy all question-generation "
                        "requirements. Regenerate every question in this chunk and address "
                        "the feedback below:\n"
                        f"{_format_validation_feedback(relevant_errors)}"
                    )
            parsed, raw = await client.generate(
                system=SYSTEM,
                prompt=user_prompt,
                schema=QuestionResponse,
                max_tokens=plan["output_tokens"],
                audit_context={
                    "sample_id": row["sample_id"], "stage": "stage4_questions",
                    "stage_attempt": attempt,
                    "tuple_ids": [item["tuple_id"] for item in plan["tuples"]],
                },
            )
            return parsed, raw, user_prompt

        tuple_models = [AtomicTuple.model_validate(item) for item in tuples]
        dep_models = [DependencyRecord.model_validate(item) for item in deps]
        prior_errors: list[str] = []
        instructions: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            chunk_results = await asyncio.gather(*(
                generate_chunk(plan, attempt, prior_errors) for plan in chunk_plans
            ))
            instructions.extend([
                {"stage": "stage4_questions", "attempt": attempt, "chunk_index": index,
                 "tuple_ids": [item["tuple_id"] for item in plan["tuples"]],
                 "system": SYSTEM, "user": user_prompt}
                for index, (plan, (_, _, user_prompt)) in enumerate(
                    zip(chunk_plans, chunk_results), 1
                )
            ])
            parsed_questions = [
                question for parsed, _, _ in chunk_results for question in parsed.questions
            ]
            task_by_id = {
                task["tuple_id"]: task
                for plan in chunk_plans for task in plan["question_tasks"]
            }
            generated_ids = {item.tuple_id for item in parsed_questions}
            fallback_tuple_ids = [
                item.tuple_id for item in tuple_models if item.tuple_id not in generated_ids
            ]
            parsed_questions.extend(
                GeneratedQuestion(
                    tuple_id=tuple_id,
                    question=deterministic_question(
                        task_by_id[tuple_id], chunk_plans[0]["target_language"]
                    ),
                )
                for tuple_id in fallback_tuple_ids
            )
            raw = "\n".join(raw_chunk for _, raw_chunk, _ in chunk_results)
            try:
                effective_questions = canonicalize_questions(
                    tuple_models, dep_models, parsed_questions, row["prompt"],
                    row.get("language"),
                )
            except ValueError as exc:
                prior_errors = [f"question_generation_structure:{exc}"]
                continue
            errors = validate_questions(
                row["prompt"], tuple_models, dep_models, effective_questions,
                row.get("language"),
            )
            prior_errors = _hard_errors(errors)
            if prior_errors and attempt == 3:
                # A multi-task chunk can occasionally leak a sibling task's
                # value into one question even after retries. Replace only the
                # first (offending) tuple named by each validator finding with
                # the deterministic single-task renderer, then revalidate the
                # complete set. The fallback sees only that tuple's task.
                offender_ids = {
                    match.group(1)
                    for error in prior_errors
                    if (match := re.search(r":(T\d+)(?::|$)", error))
                }
                if offender_ids:
                    parsed_questions = [
                        item for item in parsed_questions
                        if item.tuple_id not in offender_ids
                    ]
                    parsed_questions.extend(
                        GeneratedQuestion(
                            tuple_id=tuple_id,
                            question=deterministic_question(
                                task_by_id[tuple_id], chunk_plans[0]["target_language"]
                            ),
                        )
                        for tuple_id in offender_ids
                    )
                    effective_questions = canonicalize_questions(
                        tuple_models, dep_models, parsed_questions, row["prompt"],
                        row.get("language"),
                    )
                    errors = validate_questions(
                        row["prompt"], tuple_models, dep_models, effective_questions,
                        row.get("language"),
                    )
                    prior_errors = _hard_errors(errors)
                    fallback_tuple_ids = list(dict.fromkeys(
                        fallback_tuple_ids + [
                            item.tuple_id for item in tuple_models
                            if item.tuple_id in offender_ids
                        ]
                    ))
            if not prior_errors:
                return {"sample_id": row["sample_id"], "status": "ok",
                        "model_questions": [item.model_dump() for item in parsed_questions],
                        "questions": [item.model_dump() for item in effective_questions],
                        "question_chunks": [{key: value for key, value in plan.items()
                                             if key not in {"tuples", "dependencies", "question_tasks"}}
                                            | {"tuple_ids": [item["tuple_id"] for item in plan["tuples"]]}
                                            for plan in chunk_plans],
                        "fallback_tuple_ids": fallback_tuple_ids,
                        "llm_instructions": instructions,
                        "validation_errors": errors, "raw_model_output": raw}
        raise ValueError("question_hard_validation:" + ",".join(prior_errors))
    await _run_pending(question_inputs, question_path, args.concurrency, stage4)

    question_rows = {row["sample_id"]: row for row in read_jsonl(question_path) if row.get("status") == "ok"}
    # Re-apply deterministic graph metadata and punctuation on every run. This
    # upgrades checkpointed question wording without another model call.
    for sid, question_row in question_rows.items():
        if sid not in tuple_map or sid not in dep_rows:
            continue
        tuple_models = [AtomicTuple.model_validate(item) for item in tuple_map[sid]["tuples"]]
        dep_models = [DependencyRecord.model_validate(item) for item in dep_rows[sid]["dependencies"]]
        model_questions = [GeneratedQuestion.model_validate(item)
                           for item in question_row.get("model_questions", question_row["questions"])]
        manifest_row = next(item for item in manifest if item["sample_id"] == sid)
        # A newly canonicalized scene graph can add or remove tuples while an
        # earlier successful question checkpoint still reflects the old tuple
        # set. Reconcile it deterministically instead of crashing the summary
        # pass or forcing unrelated model questions to be regenerated.
        language_value = str(manifest_row.get("language") or "").casefold()
        if language_value.startswith("zh"):
            target_language = "CHINESE (中文)"
        elif language_value.startswith("en"):
            target_language = "ENGLISH"
        else:
            target_language = (
                "CHINESE (中文)" if any("\u4e00" <= char <= "\u9fff"
                                       for char in manifest_row["prompt"])
                else "ENGLISH"
            )
        checkpoint_by_tuple = {item.tuple_id: item for item in model_questions}
        tuple_dicts = [item.model_dump() for item in tuple_models]
        task_by_tuple = {
            task["tuple_id"]: task
            for task in question_tasks(
                tuple_dicts, event_label_map(tuple_dicts), target_language
            )
        }
        model_questions = [
            checkpoint_by_tuple.get(item.tuple_id)
            or GeneratedQuestion(
                tuple_id=item.tuple_id,
                question=deterministic_question(
                    task_by_tuple[item.tuple_id], target_language
                ),
            )
            for item in tuple_models
        ]
        question_row["model_questions"] = [item.model_dump() for item in model_questions]
        effective_questions = canonicalize_questions(
            tuple_models, dep_models, model_questions, manifest_row["prompt"],
            manifest_row.get("language"),
        )
        checkpoint_errors = validate_questions(
            manifest_row["prompt"], tuple_models, dep_models, effective_questions,
            manifest_row.get("language"),
        )
        if hard_checkpoint_errors := _hard_errors(checkpoint_errors):
            offender_ids = {
                tuple_id
                for error in hard_checkpoint_errors
                for tuple_id in re.findall(r"T\d+", error)
            }
            repaired_by_tuple = {item.tuple_id: item for item in model_questions}
            for tuple_id in offender_ids:
                if tuple_id in task_by_tuple:
                    repaired_by_tuple[tuple_id] = GeneratedQuestion(
                        tuple_id=tuple_id,
                        question=deterministic_question(
                            task_by_tuple[tuple_id], target_language
                        ),
                    )
            model_questions = [repaired_by_tuple[item.tuple_id] for item in tuple_models]
            effective_questions = canonicalize_questions(
                tuple_models, dep_models, model_questions, manifest_row["prompt"],
                manifest_row.get("language"),
            )
            checkpoint_errors = validate_questions(
                manifest_row["prompt"], tuple_models, dep_models, effective_questions,
                manifest_row.get("language"),
            )
            if _hard_errors(checkpoint_errors):
                model_questions = [
                    GeneratedQuestion(
                        tuple_id=item.tuple_id,
                        question=deterministic_question(
                            task_by_tuple[item.tuple_id], target_language
                        ),
                    )
                    for item in tuple_models
                ]
                effective_questions = canonicalize_questions(
                    tuple_models, dep_models, model_questions, manifest_row["prompt"],
                    manifest_row.get("language"),
                )
                checkpoint_errors = validate_questions(
                    manifest_row["prompt"], tuple_models, dep_models,
                    effective_questions, manifest_row.get("language"),
                )
            question_row["model_questions"] = [
                item.model_dump() for item in model_questions
            ]
            question_row["fallback_tuple_ids"] = list(dict.fromkeys(
                (question_row.get("fallback_tuple_ids") or [])
                + sorted(offender_ids)
            ))
        question_row["questions"] = [item.model_dump() for item in effective_questions]
        question_row["validation_errors"] = checkpoint_errors
    results: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in manifest:
        sid = row["sample_id"]
        errors = {
            "scene_graph": (verified_rows.get(sid) or {}).get("validation_errors", ["missing_stage1c"]),
            "dependencies": (dep_rows.get(sid) or {}).get("validation_errors", ["missing_stage3"]),
            "questions": (question_rows.get(sid) or {}).get("validation_errors", ["missing_stage4"]),
        }
        warnings = {
            stage: [item for item in values if item.startswith(WARNING_PREFIXES)]
            for stage, values in errors.items()
        }
        hard_errors = {
            stage: _hard_errors(values)
            for stage, values in errors.items()
        }
        valid = all(not value for value in hard_errors.values())
        llm_instructions = []
        llm_instructions.extend((stage1_rows.get(sid) or {}).get("llm_instructions", []))
        llm_instructions.extend((relation_rows.get(sid) or {}).get("llm_instructions", []))
        llm_instructions.extend((verified_rows.get(sid) or {}).get("llm_instructions", []))
        llm_instructions.extend((question_rows.get(sid) or {}).get("llm_instructions", []))
        result = {**row, "pipeline_name": PIPELINE_NAME,
                  "scene_graph": (verified_rows.get(sid) or {}).get("scene_graph"),
                  "atomic_tuples": (tuple_map.get(sid) or {}).get("tuples"),
                  "dependency_strategy": (dep_rows.get(sid) or {}).get("dependency_strategy"),
                  "dependencies": (dep_rows.get(sid) or {}).get("dependencies"),
                  "model_questions": (question_rows.get(sid) or {}).get("model_questions"),
                  "questions": (question_rows.get(sid) or {}).get("questions"),
                  "question_chunks": (question_rows.get(sid) or {}).get("question_chunks"),
                  "fallback_tuple_ids": (question_rows.get(sid) or {}).get("fallback_tuple_ids", []),
                  "llm_instructions": llm_instructions,
                  "validation": {"valid": valid, "errors": hard_errors, "warnings": warnings}}
        results.append(result)
        validation.append({"sample_id": sid, "valid": valid,
                           "errors": hard_errors, "warnings": warnings})
    write_jsonl(output / "results.jsonl", results)
    write_jsonl(output / "validation_report.jsonl", validation)
    write_jsonl(
        output / "llm_instructions.jsonl",
        [
            {"sample_id": row["sample_id"], **instruction}
            for row in results for instruction in row.get("llm_instructions", [])
        ],
    )
    summary = {
        "pipeline_name": PIPELINE_NAME,
        "samples": len(results), "valid": sum(item["valid"] for item in validation),
        "invalid": sum(not item["valid"] for item in validation),
        "by_route": {},
    }
    for route in sorted({row["route"] for row in results}):
        subset = [row for row in results if row["route"] == route]
        summary["by_route"][route] = {"samples": len(subset), "valid": sum(row["validation"]["valid"] for row in subset)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose a JSONL manifest of audio prompts into validated rubrics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--temperature", type=float,
        help="sampling temperature; defaults to FINEAUDIO_DECOMPOSITION_TEMPERATURE or 0",
    )
    parser.add_argument(
        "--seed", type=int,
        help="sampling seed; defaults to FINEAUDIO_DECOMPOSITION_SEED when set",
    )
    parser.add_argument("--scene-graph-tokens", type=int, default=8192)
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--question-tokens", type=int, default=4096)
    parser.add_argument("--question-output-tokens-per-question", type=int, default=160)
    parser.add_argument("--question-safety-margin", type=int, default=2048)
    parser.add_argument("--question-max-tuples-per-chunk", type=int, default=64)
    parser.add_argument("--stop-after-relation-audit", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
