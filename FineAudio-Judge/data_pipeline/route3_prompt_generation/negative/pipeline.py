from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
STAGES = CODE_ROOT / "stages"
POSITIVE_STAGES = CODE_ROOT.parent / "positive/stages"
DEFAULT_RUN_DIR = Path("outputs/route3/negative")


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route 3 controlled negative construction. Run positive construction first."
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    def add_run_dir(stage_parser: argparse.ArgumentParser) -> None:
        stage_parser.add_argument(
            "--run-dir", type=Path, default=DEFAULT_RUN_DIR,
            help="Run directory for plans, checkpoints, generated audio, and manifests.",
        )

    plan = sub.add_parser("plan", help="Select 1,000 unique positives and split prompt/audio branches.")
    add_run_dir(plan)
    plan.add_argument(
        "--positive-with-questions", type=Path,
        help="Legacy combined positive/rubric manifest.",
    )
    plan.add_argument(
        "--positive-manifest", type=Path,
        help="Route 3 positive pairs; use together with --rubric-manifest.",
    )
    plan.add_argument(
        "--rubric-manifest", type=Path,
        help="Independent label-blind rubric-decomposition output.",
    )
    plan.add_argument("--positive-review", type=Path, required=True)

    prompt_prepare = sub.add_parser("prompt-prepare", help="Build dependency-closed prompt edit targets and span groups.")
    add_run_dir(prompt_prepare)
    prompt_run = sub.add_parser("prompt-run-shard", help="Generate one local-model prompt-negative shard.")
    add_run_dir(prompt_run)
    prompt_run.add_argument("--shard-id", type=int, required=True)
    prompt_run.add_argument("--num-shards", type=int, default=2)
    prompt_run.add_argument("--model-path", type=Path, required=True)
    prompt_run.add_argument("--batch-size", type=int, default=32)
    prompt_run.add_argument("--audit-batch-size", type=int, default=16)
    prompt_merge = sub.add_parser("prompt-merge", help="Merge completed prompt-side shards.")
    add_run_dir(prompt_merge)
    prompt_merge.add_argument("--num-shards", type=int, default=2)
    rubric_export = sub.add_parser(
        "rubric-export", help="Export P-prime prompts for canonical rubric decomposition."
    )
    add_run_dir(rubric_export)

    audio_prompts = sub.add_parser("audio-prompts", help="Generate exact counterfactual prompts for audio regeneration.")
    add_run_dir(audio_prompts)
    audio_prompts.add_argument("--endpoint", action="append", required=True)
    audio_prompts.add_argument("--model", required=True)
    audio_prompts.add_argument("--concurrency", type=int, default=4)

    audio_generate = sub.add_parser("audio-generate", help="Regenerate A-prime with the Route 3 Audio-Oscar runner.")
    add_run_dir(audio_generate)
    audio_generate.add_argument("--max-sample-attempts", type=int, default=3)
    audio_assemble = sub.add_parser("audio-assemble", help="Attach completed A-prime audio to the pending pair manifest.")
    add_run_dir(audio_assemble)
    audit = sub.add_parser("audit", help="Audit the 500 prompt-side and 500 audio-side pairs.")
    add_run_dir(audit)
    audit.add_argument(
        "--prompt-rubrics", type=Path, required=True,
        help="results.jsonl from canonical decomposition of the exported P-prime prompts.",
    )
    args = parser.parse_args()

    run_root = args.run_dir.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    stage_env = os.environ.copy()
    stage_env["FINEAUDIO_ROUTE3_NEGATIVE_RUN"] = str(run_root)
    common = ["--root", str(run_root)]
    if args.stage == "plan":
        legacy = args.positive_with_questions
        split = args.positive_manifest or args.rubric_manifest
        if legacy and split:
            raise SystemExit(
                "use either --positive-with-questions or "
                "--positive-manifest with --rubric-manifest"
            )
        if legacy:
            source = legacy
            rubric_source = None
        elif args.positive_manifest and args.rubric_manifest:
            source = args.positive_manifest
            rubric_source = args.rubric_manifest
        else:
            raise SystemExit(
                "provide --positive-with-questions, or both --positive-manifest "
                "and --rubric-manifest"
            )
        command = [
            sys.executable, str(STAGES / "plan_negatives.py"), *common,
            "--source", str(source),
            "--positive-review", str(args.positive_review), "prepare",
        ]
        if rubric_source is not None:
            command[4:4] = ["--rubric-source", str(rubric_source)]
        run(command, env=stage_env)
    elif args.stage == "prompt-prepare":
        run([sys.executable, str(STAGES / "prompt_negative_planner.py"), "prepare"], env=stage_env)
        run([sys.executable, str(STAGES / "generate_prompt_negatives.py"), "prepare-groups"], env=stage_env)
    elif args.stage == "prompt-run-shard":
        run([
            sys.executable, str(STAGES / "generate_prompt_negatives.py"), "run-shard",
            "--shard-id", str(args.shard_id), "--num-shards", str(args.num_shards),
            "--model-path", str(args.model_path), "--batch-size", str(args.batch_size),
            "--audit-batch-size", str(args.audit_batch_size),
        ], env=stage_env)
    elif args.stage == "prompt-merge":
        run([
            sys.executable, str(STAGES / "generate_prompt_negatives.py"), "merge",
            "--num-shards", str(args.num_shards),
        ], env=stage_env)
    elif args.stage == "rubric-export":
        run([
            sys.executable, str(STAGES / "export_prompt_rubric_input.py"),
            "--pairs", str(run_root / "prompt_side/manifests/prompt_side_pairs.jsonl"),
            "--output", str(run_root / "manifests/prompt_side_rubric_input.jsonl"),
        ], env=stage_env)
    elif args.stage == "audio-prompts":
        command = [
            sys.executable, str(STAGES / "generate_audio_negative_prompts.py"),
            "--plan", str(run_root / "plan/audio_side_plan.jsonl"),
            "--output", str(run_root / "patches/audio_side_patch_proposals.jsonl"),
            "--pprime-output", str(run_root / "manifests/audio_side_prompt_preview.jsonl"),
            "--model", args.model, "--concurrency", str(args.concurrency),
        ]
        for endpoint in args.endpoint:
            command.extend(["--endpoint", endpoint])
        run(command, env=stage_env)
        run([
            sys.executable, str(STAGES / "plan_negatives.py"), *common,
            "finalize-audio", "--proposals", str(run_root / "patches/audio_side_patch_proposals.jsonl"),
        ], env=stage_env)
    elif args.stage == "audio-generate":
        env = stage_env.copy()
        env["AUDIO_OSCAR_RUN_FINAL_EDIT"] = "false"
        run([
            sys.executable, str(POSITIVE_STAGES / "run_audio_oscar.py"),
            "--prompts", str(run_root / "manifests/audio_generation_prompts.jsonl"),
            "--output-root", str(run_root / "audio_side/oscar_outputs"),
            "--skip-complete", "--max-sample-attempts", str(args.max_sample_attempts),
        ], env=env)
    elif args.stage == "audio-assemble":
        run([sys.executable, str(STAGES / "plan_negatives.py"), *common, "assemble-audio"], env=stage_env)
    else:
        run([
            sys.executable, str(STAGES / "audit_outputs.py"),
            "--root", str(run_root), "--prompt-rubrics", str(args.prompt_rubrics),
        ], env=stage_env)


if __name__ == "__main__":
    main()
