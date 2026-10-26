from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

from client import ChatClient, SamplingConfig, prompt_from_response
from generate_prompts import prompt_language_matches
from provenance import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
    sha256_json,
    update_run_stage,
    utc_now,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_is_approved(stage_root: Path) -> bool:
    approval_path = stage_root / "batch_approval.json"
    if not approval_path.is_file():
        return False
    return load_json(approval_path).get("approved") is True


def post_generation_qc_passed(stage_root: Path) -> bool:
    """Return whether QC covers every currently accepted prompt.

    A resumed or repartitioned shard can have a historical requested count in
    its manifest.  Finalization retry must therefore rely on the immutable
    accepted records and their matching QC report, not that stale request.
    """
    qc_path = stage_root / "post_generation_qc.json"
    if not qc_path.is_file():
        return False
    qc = load_json(qc_path)
    accepted_count = len(read_jsonl(stage_root / "accepted_prompts.jsonl"))
    return (
        accepted_count > 0
        and qc.get("record_count") == accepted_count
        and qc.get("passed_count") == accepted_count
        and qc.get("failed_count") == 0
        and not qc.get("manual_fail_reason")
    )


def render(template: str, key: str, value: Any) -> str:
    return template.replace(
        "{" + key + "}", json.dumps(value, ensure_ascii=False, indent=2)
    )


def sampling(config: dict[str, Any], key: str) -> SamplingConfig:
    section = config[key]
    return SamplingConfig(
        temperature=float(section["temperature"]),
        top_p=float(section["top_p"]),
        max_tokens=int(section["max_tokens"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a complete prompt batch before user approval and Oscar."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/generation_config.json"
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Finalize only the selected accepted prompt IDs (repeatable).",
    )
    parser.add_argument(
        "--source-prompts",
        type=Path,
        help="Optional JSONL source relative to the prompt stage directory.",
    )
    parser.add_argument(
        "--force-refinalize",
        action="store_true",
        help="Rebuild final prompts after an approved batch when review rules changed.",
    )
    return parser.parse_args()


def completed_checkpoints(log_path: Path) -> dict[str, dict[str, Any]]:
    """Return the latest valid completed finalization record for each prompt."""
    checkpoints: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(log_path):
        prompt_id = str(record.get("id", ""))
        final_prompt = record.get("final_prompt")
        audit = record.get("audit")
        if (
            prompt_id
            and record.get("status") == "completed"
            and isinstance(final_prompt, str)
            and final_prompt.strip()
            and isinstance(audit, dict)
            and audit.get("pass") is True
        ):
            checkpoints[prompt_id] = record
    return checkpoints


def main() -> int:
    args = parse_args()
    run_root = args.runs_root / args.run_id
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Unknown run: {run_root}")
    manifest = load_json(manifest_path)
    prompt_stage = manifest["stages"]["01_prompt_generation"]
    stage_root = run_root / "01_prompt_generation"
    finalization_retry = (
        prompt_stage.get("status") == "failed"
        and prompt_stage.get("finalization_failed_id")
        and post_generation_qc_passed(stage_root)
    )
    if (
        prompt_stage.get("status") not in {"completed", "awaiting_approval"}
        and not finalization_retry
    ):
        raise SystemExit("Prompt generation must complete before batch finalization")
    if batch_is_approved(stage_root) and not args.force_refinalize:
        raise SystemExit(
            "The batch is already approved; use --force-refinalize to rebuild it "
            "and require fresh approval"
        )
    source_path = (
        stage_root / args.source_prompts
        if args.source_prompts is not None
        else stage_root / "accepted_prompts.jsonl"
    )
    accepted = read_jsonl(source_path)
    if args.sample_id:
        wanted = set(args.sample_id)
        accepted = [record for record in accepted if str(record.get("id")) in wanted]
        found = {str(record.get("id")) for record in accepted}
        missing = wanted - found
        if missing:
            raise SystemExit(f"Unknown sample IDs: {sorted(missing)}")
        # Preserve the caller's order so a hand-reviewed pilot is reproducible.
        order = {sample_id: index for index, sample_id in enumerate(args.sample_id)}
        accepted.sort(key=lambda record: order[str(record.get("id"))])
    config = load_json(args.config)
    finalizer = (ROOT / "prompts/finalize_batch.txt").read_text(encoding="utf-8")
    audit_template = (ROOT / "prompts/audit.txt").read_text(encoding="utf-8")
    client = ChatClient(config["api_base"], config["model"])
    final_records: list[dict[str, Any]] = []
    log_path = stage_root / "finalization.jsonl"
    checkpoints = (
        {} if args.force_refinalize else completed_checkpoints(log_path)
    )

    for source in accepted:
        prompt_id = str(source["id"])
        checkpoint = checkpoints.get(prompt_id)
        if (
            checkpoint is not None
            and checkpoint.get("source_prompt") == source["prompt"]
            and prompt_language_matches(
                str(checkpoint.get("final_prompt", "")), source.get("language")
            )
        ):
            finalized = str(checkpoint["final_prompt"])
            final_records.append(
                {
                    **source,
                    "prompt": finalized,
                    "prompt_sha256": sha256_json(finalized),
                    "pipeline_stage": "final_prompt_awaiting_user_approval",
                    "finalized_at": checkpoint.get("created_at", utc_now()),
                    "source_accepted_prompt_sha256": source["prompt_sha256"],
                }
            )
            continue
        finalized_response = client.json_completion(
            render(
                finalizer,
                "candidate_json",
                {
                    "prompt": source["prompt"],
                    "language": source.get("language"),
                },
            ),
            sampling(config, "rewrite"),
        )
        finalized = prompt_from_response(finalized_response)
        audit: dict[str, Any] = {}
        finalization_attempts = []
        passed = False
        for pass_index in range(1, 4):
            audit = client.json_completion(
                render(audit_template, "candidate_json", {"prompt": finalized}),
                sampling(config, "audit"),
            )
            finalization_attempts.append(
                {
                    "pass_index": pass_index,
                    "stage": "llm_audio_only_audit",
                    "result": audit,
                    "prompt": finalized,
                }
            )
            if audit.get("pass") is True:
                passed = True
                break
            try:
                corrected = prompt_from_response(audit, "corrected_prompt")
            except ValueError:
                break
            finalized = corrected

        if not passed:
            append_jsonl(
                log_path,
                {
                    "id": prompt_id,
                    "status": "failed",
                    "created_at": utc_now(),
                    "source_prompt": source["prompt"],
                    "finalized_response": finalized_response,
                    "attempts": finalization_attempts,
                },
            )
            update_run_stage(
                run_root,
                "01_prompt_generation",
                "failed",
                finalization_failed_id=prompt_id,
                finalization_log="01_prompt_generation/finalization.jsonl",
            )
            return 1

        if not prompt_language_matches(finalized, source.get("language")):
            append_jsonl(
                log_path,
                {
                    "id": prompt_id,
                    "status": "failed",
                    "created_at": utc_now(),
                    "source_prompt": source["prompt"],
                    "finalized_response": finalized_response,
                    "attempts": finalization_attempts,
                    "reason": "final_prompt_language_mismatch",
                    "expected_language": source.get("language"),
                    "final_prompt": finalized,
                },
            )
            update_run_stage(
                run_root,
                "01_prompt_generation",
                "failed",
                finalization_failed_id=prompt_id,
                finalization_failure_reason="final_prompt_language_mismatch",
                finalization_log="01_prompt_generation/finalization.jsonl",
            )
            return 1

        record = {
            **source,
            "prompt": finalized,
            "prompt_sha256": sha256_json(finalized),
            "pipeline_stage": "final_prompt_awaiting_user_approval",
            "finalized_at": utc_now(),
            "source_accepted_prompt_sha256": source["prompt_sha256"],
        }
        final_records.append(record)
        append_jsonl(
            log_path,
            {
                "id": prompt_id,
                "status": "completed",
                "created_at": utc_now(),
                "source_prompt": source["prompt"],
                "finalized_response": finalized_response,
                "final_prompt": finalized,
                "audit": audit,
                "attempts": finalization_attempts,
            },
        )

    final_path = stage_root / "final_prompts.jsonl"
    atomic_write_jsonl(final_path, final_records)
    final_hash = sha256_file(final_path)
    atomic_write_json(
        stage_root / "batch_approval.json",
        {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "approved": False,
            "approved_by": None,
            "approved_at": None,
            "final_prompt_count": len(final_records),
            "final_prompts_sha256": final_hash,
            "reason": "Awaiting explicit user review and approval.",
        },
    )
    update_run_stage(
        run_root,
        "01_prompt_generation",
        "awaiting_approval",
        final_prompt_count=len(final_records),
        final_prompts="01_prompt_generation/final_prompts.jsonl",
        final_prompts_sha256=final_hash,
        batch_approval="01_prompt_generation/batch_approval.json",
    )
    print(
        f"run_id={args.run_id} finalized={len(final_records)} awaiting_user_approval"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
