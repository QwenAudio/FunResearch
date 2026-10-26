#!/usr/bin/env python3
"""Build Route 2 controlled negatives from a completed Route 2 positive run."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
STAGES = HERE / "stages"
POSITIVE_STAGES = HERE.parent / "positive" / "stages"
STAGE_ORDER = (
    "construction_labels",
    "plan",
    "relation_plan",
    "negative_pools",
    "prepare",
    "audio_scenes",
    "prompt_generation",
    "prompt_materialization",
    "audio_mix",
    "audio_qc",
    "audio_materialization",
    "prompt_construction_labels",
    "audit",
)


def command(directory: Path, script: str, *args: object) -> list[str]:
    return [sys.executable, str(directory / script), *(str(value) for value in args)]


def stage_slice(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGE_ORDER.index(start_at)
    stop = STAGE_ORDER.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    return STAGE_ORDER[start : stop + 1]


def require(path: Path | None, flag: str, dry_run: bool) -> Path:
    if path is None:
        raise ValueError(f"{flag} is required for the selected stages")
    if not dry_run and not path.exists():
        raise FileNotFoundError(f"{flag} does not exist: {path}")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_recipe(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Route 2 recipe must declare schema_version=1")
    return payload


def recipe_value(recipe: dict, section: str, key: str, fallback: int) -> int:
    return int(recipe.get(section, {}).get(key, fallback))


class Driver:
    def __init__(self, root: Path, dry_run: bool) -> None:
        self.root = root
        self.dry_run = dry_run

    def run(self, label: str, argv: Sequence[str], append_log: bool = False) -> None:
        print(f"[{label}] {shlex.join(argv)}", flush=True)
        if self.dry_run:
            return
        log_path = self.root / "logs" / f"{label}.log"
        with log_path.open("a" if append_log else "w", encoding="utf-8") as log:
            completed = subprocess.run(
                list(argv), stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode:
            raise RuntimeError(
                f"Stage {label!r} failed with exit code {completed.returncode}. "
                f"See {log_path}."
            )


def valid_audio_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(row.get("valid") is True for row in read_jsonl(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path,
        help="Optional Route 2 recipe; reads the negative_pool section when present.",
    )
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audioset-shards", type=Path)
    parser.add_argument("--zh-labels", type=Path)
    parser.add_argument("--zh-audio-root", type=Path)
    parser.add_argument("--paraspeechcaps-parquet", type=Path)
    parser.add_argument("--ears-audio-root", type=Path)
    parser.add_argument("--expresso-audio-root", type=Path)
    parser.add_argument(
        "--prepared-speech-pool", type=Path,
        help="Optional canonical speech JSONL; bypasses dataset-specific speech selection.",
    )
    parser.add_argument("--start-at", choices=STAGE_ORDER, default=STAGE_ORDER[0])
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default=STAGE_ORDER[-1])
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--mix-workers", type=int, default=16)
    parser.add_argument("--negative-effect-count", type=int)
    parser.add_argument("--negative-ambience-count", type=int)
    parser.add_argument("--negative-music-count", type=int)
    parser.add_argument("--negative-zh-speech-count", type=int)
    parser.add_argument("--negative-en-speech-count", type=int)
    parser.add_argument(
        "--exclude-source-manifest", type=Path, action="append", default=[],
        help="Optional JSONL denylists applied to the negative replacement speech pool.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    args.negative_effect_count = (
        args.negative_effect_count if args.negative_effect_count is not None
        else recipe_value(recipe, "negative_pool", "effect_count", 1400)
    )
    args.negative_ambience_count = (
        args.negative_ambience_count if args.negative_ambience_count is not None
        else recipe_value(recipe, "negative_pool", "ambience_count", 250)
    )
    args.negative_music_count = (
        args.negative_music_count if args.negative_music_count is not None
        else recipe_value(recipe, "negative_pool", "music_count", 250)
    )
    args.negative_zh_speech_count = (
        args.negative_zh_speech_count if args.negative_zh_speech_count is not None
        else recipe_value(recipe, "negative_pool", "zh_speech_count", 800)
    )
    args.negative_en_speech_count = (
        args.negative_en_speech_count if args.negative_en_speech_count is not None
        else recipe_value(recipe, "negative_pool", "en_speech_count", 450)
    )
    for name in (
        "negative_effect_count", "negative_ambience_count", "negative_music_count",
        "negative_zh_speech_count", "negative_en_speech_count",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    selected = stage_slice(args.start_at, args.stop_after)
    root = args.output_dir.resolve()
    positive = args.positive_dir.resolve()
    for name in (
        "assets/audio", "assets/extract_parts", "prepared", "prompt_side",
        "audio_side", "logs",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)

    positive_pairs = positive / "pairs.jsonl"
    positive_speech = positive / "speech_manifest.jsonl"
    positive_assets = positive / "assets/source_assets.jsonl"
    for path, flag in (
        (positive_pairs, "--positive-dir/pairs.jsonl"),
        (positive_speech, "--positive-dir/speech_manifest.jsonl"),
        (positive_assets, "--positive-dir/assets/source_assets.jsonl"),
    ):
        if not args.dry_run and not path.exists():
            raise FileNotFoundError(f"{flag} does not exist: {path}")

    if "negative_pools" in selected:
        require(args.audioset_shards, "--audioset-shards", args.dry_run)
        if args.prepared_speech_pool is not None:
            require(args.prepared_speech_pool, "--prepared-speech-pool", args.dry_run)
        else:
            for path, flag in (
                (args.zh_labels, "--zh-labels"),
                (args.zh_audio_root, "--zh-audio-root"),
                (args.paraspeechcaps_parquet, "--paraspeechcaps-parquet"),
                (args.ears_audio_root, "--ears-audio-root"),
                (args.expresso_audio_root, "--expresso-audio-root"),
            ):
                require(path, flag, args.dry_run)
    labels = root / "positive_construction_labels.jsonl"
    initial_plan = root / "negative_plan.jsonl"
    plan = root / "materializable_negative_plan.jsonl"
    selection = root / "assets/nonspeech_selection.jsonl"
    nonspeech = root / "assets/nonspeech_pool.jsonl"
    speech = root / "assets/speech_pool.jsonl"
    prompt_jobs = root / "prepared/prompt_jobs.jsonl"
    audio_recipes = root / "prepared/audio_recipes.jsonl"
    audio_scenes = root / "audio_side/scene_graphs.jsonl"
    generated = root / "prompt_side/generated.jsonl"
    mix = root / "audio_side/mix"
    driver = Driver(root, args.dry_run)

    if "construction_labels" in selected:
        driver.run("build_positive_construction_labels", command(
            STAGES, "build_construction_labels.py",
            "--input", positive_pairs, "--output", labels,
        ))

    if "plan" in selected:
        driver.run("plan_negatives", command(
            STAGES, "plan_negatives.py",
            "--construction-labels", labels,
            "--output", initial_plan,
            "--seed", args.seed,
        ))

    if "relation_plan" in selected:
        driver.run("make_relation_plan_materializable", command(
            STAGES, "make_relation_plan_materializable.py",
            "--plan", initial_plan,
            "--positive-pairs", positive_pairs,
            "--construction-labels", labels,
            "--output", plan,
        ))

    if "negative_pools" in selected:
        driver.run("select_negative_nonspeech_pool", command(
            STAGES, "build_nonspeech_pool.py",
            "--positive-pairs", positive_pairs,
            "--shards", args.audioset_shards,
            "--output", selection,
            "--effect-count", args.negative_effect_count,
            "--ambience-count", args.negative_ambience_count,
            "--music-count", args.negative_music_count,
            "--seed", args.seed,
        ))
        driver.run("extract_negative_nonspeech_pool", command(
            STAGES, "extract_nonspeech_pool.py",
            "--selection", selection,
            "--output-dir", root / "assets/audio",
            "--manifest", nonspeech,
            "--parts-dir", root / "assets/extract_parts",
        ))
        if args.prepared_speech_pool is not None:
            print(f"[use_prepared_speech_pool] {args.prepared_speech_pool} -> {speech}", flush=True)
            if not args.dry_run:
                rows = read_jsonl(args.prepared_speech_pool)
                expected = args.negative_zh_speech_count + args.negative_en_speech_count
                if len(rows) != expected:
                    raise ValueError(
                        f"prepared speech pool has {len(rows)} rows; recipe requests {expected}"
                    )
                if any(not Path(row["audio_path"]).exists() for row in rows):
                    raise FileNotFoundError("prepared speech pool contains missing audio paths")
                shutil.copyfile(args.prepared_speech_pool, speech)
        else:
            speech_pool_command = command(
                STAGES, "build_speech_pool.py",
                "--positive-speech", positive_speech,
                "--zh-labels", args.zh_labels,
                "--zh-audio-root", args.zh_audio_root,
                "--paraspeechcaps-parquet", args.paraspeechcaps_parquet,
                "--ears-audio-root", args.ears_audio_root,
                "--expresso-audio-root", args.expresso_audio_root,
                "--output", speech,
                "--zh-count", args.negative_zh_speech_count,
                "--en-count", args.negative_en_speech_count,
                "--seed", args.seed,
            )
            for manifest in args.exclude_source_manifest:
                speech_pool_command.extend(("--exclude-source-manifest", str(manifest)))
            driver.run("build_negative_speech_pool", speech_pool_command)

    if "prepare" in selected:
        driver.run("prepare_materialization", command(
            STAGES, "prepare_materialization.py",
            "--plan", plan,
            "--pairs", positive_pairs,
            "--construction-labels", labels,
            "--assets", positive_assets,
            "--speech-pool", speech,
            "--output-dir", root / "prepared",
        ))

    if "audio_scenes" in selected:
        driver.run("build_audio_negative_scenes", command(
            STAGES, "build_audio_negative_scenes.py",
            "--recipes", audio_recipes,
            "--nonspeech-pool", nonspeech,
            "--speech-pool", speech,
            "--output", audio_scenes,
            "--seed", args.seed,
        ))

    if "prompt_generation" in selected:
        # Public Route 2 negatives are exact atomic edits.  The authoritative
        # materializer already rebuilds every non-relation row from P, while
        # relation-only rows use a deterministic full-order reversal.  An LLM
        # proposal here was discarded downstream, so invoking one added cost
        # and a failure mode without changing P'.
        driver.run("build_relation_prompt_negatives", command(
            STAGES, "build_relation_prompt_negatives.py",
            "--jobs", prompt_jobs,
            "--pairs", positive_pairs,
            "--output", generated,
        ))

    if "prompt_materialization" in selected:
        driver.run("materialize_prompt_negatives", command(
            STAGES, "materialize_prompt_negatives.py",
            "--jobs", prompt_jobs,
            "--generated", generated,
            "--output-dir", root / "prompt_side",
            "--positive-pairs", positive_pairs,
            "--speech-pool", speech,
        ))

    if "audio_mix" in selected:
        driver.run("compose_audio_negatives", command(
            POSITIVE_STAGES, "compose_scene_graphs.py",
            "--scene-graphs", audio_scenes,
            "--output-dir", mix,
            "--workers", args.mix_workers,
            "--resume",
        ))

    if "audio_qc" in selected:
        expected_audio_count = len(read_jsonl(audio_recipes)) if not args.dry_run else None
        driver.run("repair_audio_peaks", command(
            STAGES, "repair_audio_peaks.py", "--mix-dir", mix,
        ))
        validation = mix / "validation_report.jsonl"
        if args.dry_run or valid_audio_count(validation) != expected_audio_count:
            repair_root = root / "audio_side/gap_repair"
            repaired_scenes = repair_root / "scene_graphs.jsonl"
            repaired_mix = repair_root / "mix"
            driver.run("repair_audio_gaps", command(
                STAGES, "repair_audio_gaps.py",
                "--scenes", audio_scenes,
                "--recipes", audio_recipes,
                "--validation", validation,
                "--output", repaired_scenes,
            ))
            driver.run("compose_gap_repairs", command(
                POSITIVE_STAGES, "compose_scene_graphs.py",
                "--scene-graphs", repaired_scenes,
                "--output-dir", repaired_mix,
                "--workers", min(args.mix_workers, 8),
            ))
            driver.run("repair_gap_audio_peaks", command(
                STAGES, "repair_audio_peaks.py", "--mix-dir", repaired_mix,
            ))
            driver.run("merge_audio_repairs", command(
                STAGES, "merge_audio_repair.py",
                "--canonical", mix, "--repair", repaired_mix,
            ))
        if not args.dry_run and valid_audio_count(validation) != expected_audio_count:
            raise RuntimeError(
                f"Audio QC did not validate all {expected_audio_count} audio-side negatives"
            )

    if "audio_materialization" in selected:
        driver.run("materialize_audio_negatives", command(
            STAGES, "materialize_audio_negatives.py",
            "--recipes", audio_recipes,
            "--composed", mix / "pairs.jsonl",
            "--validation", mix / "validation_report.jsonl",
            "--output-dir", root / "audio_side",
        ))

    if "prompt_construction_labels" in selected:
        driver.run("rebuild_prompt_construction_labels", command(
            STAGES, "rebuild_prompt_construction_labels.py",
            "--run", root,
            "--positive-pairs", positive_pairs,
            "--positive-construction-labels", labels,
            "--output", root / "prompt_side/construction_labels.jsonl",
        ))

    if "audit" in selected:
        driver.run("audit_outputs", command(
            STAGES, "audit_outputs.py", "--run", root, "--plan", plan,
        ))

    summary = {
        "stages": list(selected),
        "positive_run": str(positive),
        "prompt_pairs": str(root / "prompt_side/pairs.jsonl"),
        "audio_pairs": str(root / "audio_side/pairs.jsonl"),
        "final_audit": str(root / "final_audit.json"),
        "dry_run": args.dry_run,
        "prompt_negative_method": "deterministic_atomic_edit_no_llm",
        "prompt_llm_calls": 0,
        "negative_pool": {
            "effect_count": args.negative_effect_count,
            "ambience_count": args.negative_ambience_count,
            "music_count": args.negative_music_count,
            "zh_speech_count": args.negative_zh_speech_count,
            "en_speech_count": args.negative_en_speech_count,
        },
    }
    (root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
