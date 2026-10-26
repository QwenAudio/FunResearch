from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = ROOT / "stages"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def require_audio_environment() -> dict[str, str]:
    required = [
        "AUDIO_OSCAR_ROOT", "AUDIO_OSCAR_PLANNER_URL", "AUDIO_OSCAR_PLANNER_MODEL",
        "AUDIO_OSCAR_OMNI_URL", "AUDIO_OSCAR_OMNI_MODEL", "AUDIO_OSCAR_QWEN_TTS_URL",
        "AUDIO_OSCAR_COSYVOICE_URL", "AUDIO_OSCAR_VOXCPM_URL", "AUDIO_OSCAR_FA_URL",
        "AUDIO_OSCAR_VOICE_DESIGN_MODEL", "AUDIO_OSCAR_BASE_TTS_MODEL",
        "AUDIO_OSCAR_STABLE_AUDIO_URL", "AUDIO_OSCAR_ACESTEP_URL",
        "AUDIO_OSCAR_DEMUCS_HOST", "AUDIO_OSCAR_DEMUCS_PORT",
        "AUDIO_OSCAR_TTA_CONFIG_PATH", "AUDIO_OSCAR_MUSIC_CONFIG_PATH",
        "AUDIO_OSCAR_SONG_CONFIG_PATH",
    ]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(
            "Missing reproducer-supplied Audio-Oscar settings: " + ", ".join(missing)
            + ". See audio_oscar.env.example."
        )
    return os.environ.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route 3 positive construction: prompt creation, Audio-Oscar generation, and QC."
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    prompts = sub.add_parser("prompts", help="Generate, rewrite, audit, and deduplicate prompts.")
    prompts.add_argument("--count", type=int, required=True)
    prompts.add_argument("--chinese-count", type=int, required=True)
    prompts.add_argument("--english-count", type=int, required=True)
    prompts.add_argument("--run-id", required=True)
    prompts.add_argument("--runs-root", type=Path, required=True)
    prompts.add_argument("--config", type=Path, default=ROOT / "config.json")
    prompts.add_argument("--external-dedup-prompts", type=Path)

    finalize = sub.add_parser("finalize", help="Apply the frozen final prompt rewrite and audit.")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--runs-root", type=Path, required=True)
    finalize.add_argument("--config", type=Path, default=ROOT / "config.json")

    audio = sub.add_parser("audio", help="Generate audio with the pinned Audio-Oscar process.")
    audio.add_argument("--prompts", type=Path, required=True)
    audio.add_argument("--output-root", type=Path, required=True)
    audio.add_argument("--sample-id", action="append", default=[])
    audio.add_argument("--max-sample-attempts", type=int, default=3)

    audit = sub.add_parser("audit", help="Validate completed audio and materialize pairs.jsonl.")
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--expected-total", type=int)
    audit.add_argument("--expected-chinese", type=int)
    audit.add_argument("--expected-english", type=int)

    args = parser.parse_args()
    if args.stage == "prompts":
        command = [
            sys.executable, str(STAGES / "generate_prompts.py"),
            "--count", str(args.count), "--chinese-count", str(args.chinese_count),
            "--english-count", str(args.english_count), "--run-id", args.run_id,
            "--runs-root", str(args.runs_root), "--config", str(args.config),
            "--prompts-dir", str(ROOT / "prompts"),
        ]
        if args.external_dedup_prompts:
            command.extend(["--external-dedup-prompts", str(args.external_dedup_prompts)])
        run(command)
    elif args.stage == "finalize":
        run([
            sys.executable, str(STAGES / "finalize_prompt_batch.py"),
            "--run-id", args.run_id, "--runs-root", str(args.runs_root),
            "--config", str(args.config),
        ])
    elif args.stage == "audio":
        command = [
            sys.executable, str(STAGES / "run_audio_oscar.py"),
            "--prompts", str(args.prompts), "--output-root", str(args.output_root),
            "--skip-complete", "--max-sample-attempts", str(args.max_sample_attempts),
        ]
        for sample_id in args.sample_id:
            command.extend(["--sample-id", sample_id])
        run(command, env=require_audio_environment())
    else:
        command = [sys.executable, str(STAGES / "audit_outputs.py"), "--output-root", str(args.output_root)]
        for name in ("expected_total", "expected_chinese", "expected_english"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        run(command)


if __name__ == "__main__":
    main()
