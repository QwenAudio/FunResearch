"""Build controlled Route 2 positive pairs from short user-supplied assets."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
STAGES = HERE / "stages"
STAGE_ORDER = (
    "assets",
    "speech",
    "scenes",
    "mix",
    "mix_qc",
    "timeline_captions",
    "natural_captions",
    "validate",
)


def load_recipe(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Route 2 recipe must declare schema_version=1")
    return payload


def recipe_value(recipe: dict, section: str, key: str, fallback: object) -> object:
    return recipe.get(section, {}).get(key, fallback)


def validate_recipe(recipe: dict) -> None:
    if not recipe:
        return
    assets = recipe.get("asset_pool", {})
    speech = recipe.get("speech_pool", {})
    scenes = recipe.get("scenes", {})
    required = {
        "asset_pool": {"effect_count", "ambience_count", "music_count"},
        "speech_pool": {"zh_count", "ears_count", "expresso_count"},
        "scenes": {
            "count", "zh_speech_count", "en_speech_count",
            "zh_no_speech_count", "en_no_speech_count",
            "overlap_zero_count", "overlap_one_count", "overlap_two_count",
            "continuous_background_count", "music_count", "co_onset_count",
        },
    }
    sections = {"asset_pool": assets, "speech_pool": speech, "scenes": scenes}
    for name, keys in required.items():
        missing = sorted(keys - set(sections[name]))
        if missing:
            raise ValueError(f"Recipe section {name!r} is missing: {', '.join(missing)}")
    count = int(scenes["count"])
    if sum(int(scenes[key]) for key in (
        "zh_speech_count", "en_speech_count", "zh_no_speech_count", "en_no_speech_count"
    )) != count:
        raise ValueError("Scene language/speech quotas must sum to scenes.count")
    if sum(int(scenes[key]) for key in (
        "overlap_zero_count", "overlap_one_count", "overlap_two_count"
    )) != count:
        raise ValueError("Scene overlap quotas must sum to scenes.count")
    if int(speech["zh_count"]) < int(scenes["zh_speech_count"]):
        raise ValueError("speech_pool.zh_count is smaller than scenes.zh_speech_count")
    if int(speech["ears_count"]) + int(speech["expresso_count"]) < int(scenes["en_speech_count"]):
        raise ValueError("English speech pool is smaller than scenes.en_speech_count")
    for key in ("continuous_background_count", "music_count"):
        if not 0 <= int(scenes[key]) <= count:
            raise ValueError(f"scenes.{key} must be in [0, scenes.count]")
    if not 0 <= int(scenes["co_onset_count"]) <= int(scenes["overlap_one_count"]) + int(scenes["overlap_two_count"]):
        raise ValueError("co-onsets require scenes with at least one overlap")
    if scenes.get("max_effect_slug_reuse") is not None and int(scenes["max_effect_slug_reuse"]) < 1:
        raise ValueError("scenes.max_effect_slug_reuse must be at least 1")
    if scenes.get("max_effect_family_reuse") is not None and int(scenes["max_effect_family_reuse"]) < 1:
        raise ValueError("scenes.max_effect_family_reuse must be at least 1")
    for prefix in ("foreground_no_background", "foreground_with_background"):
        minimum = int(scenes.get(f"{prefix}_min", 2))
        maximum = int(scenes.get(f"{prefix}_max", minimum))
        if minimum < 2 or maximum < minimum:
            raise ValueError(f"invalid scenes.{prefix} range: {minimum}..{maximum}")
    duration_min = int(scenes.get("duration_min", 42))
    duration_max = int(scenes.get("duration_max", 58))
    if duration_min < 30 or duration_max > 60 or duration_max < duration_min:
        raise ValueError("scenes duration range must satisfy 30 <= min <= max <= 60")


def command(script: str, *args: object) -> list[str]:
    return [sys.executable, str(STAGES / script), *(str(value) for value in args)]


def stage_slice(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGE_ORDER.index(start_at)
    stop = STAGE_ORDER.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    return STAGE_ORDER[start : stop + 1]


def required_path(path: Path | None, flag: str, dry_run: bool) -> Path:
    if path is None:
        raise ValueError(f"{flag} is required for the selected stages")
    if not dry_run and not path.exists():
        raise FileNotFoundError(f"{flag} does not exist: {path}")
    return path


class Driver:
    def __init__(self, root: Path, dry_run: bool) -> None:
        self.root = root
        self.dry_run = dry_run

    def run(self, label: str, argv: Sequence[str]) -> None:
        print(f"[{label}] {shlex.join(argv)}", flush=True)
        if self.dry_run:
            return
        log_path = self.root / "logs" / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                list(argv),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Stage {label!r} failed with exit code {completed.returncode}. "
                f"See {log_path}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path,
        help="JSON recipe for a demo or user-owned custom run. CLI values override it.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audioset-shards", type=Path)
    parser.add_argument("--zh-labels", type=Path)
    parser.add_argument("--zh-audio-root", type=Path)
    parser.add_argument("--paraspeechcaps-parquet", type=Path)
    parser.add_argument("--ears-audio-root", type=Path)
    parser.add_argument("--expresso-audio-root", type=Path)
    parser.add_argument(
        "--caption-base-url",
        help="OpenAI-compatible endpoint selected by the reproducer.",
    )
    parser.add_argument(
        "--caption-model",
        help="Model name exposed by the reproducer's caption endpoint.",
    )
    parser.add_argument("--start-at", choices=STAGE_ORDER, default="assets")
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default="validate")
    parser.add_argument("--asset-seed", type=int)
    parser.add_argument("--scene-seed", type=int)
    parser.add_argument(
        "--effect-count",
        type=int,
        help="Number of sound-effect candidates.",
    )
    parser.add_argument(
        "--ambience-count",
        type=int,
        help="Number of ambience candidates.",
    )
    parser.add_argument(
        "--music-count",
        type=int,
        help="Number of music candidates.",
    )
    parser.add_argument(
        "--scene-count",
        type=int,
        help="Number of scenes to construct.",
    )
    parser.add_argument("--zh-speech-count", type=int)
    parser.add_argument("--ears-speech-count", type=int)
    parser.add_argument("--expresso-speech-count", type=int)
    parser.add_argument("--zh-scene-speech-count", type=int)
    parser.add_argument("--en-scene-speech-count", type=int)
    parser.add_argument("--zh-scene-no-speech-count", type=int)
    parser.add_argument("--en-scene-no-speech-count", type=int)
    parser.add_argument("--overlap-zero-count", type=int)
    parser.add_argument("--overlap-one-count", type=int)
    parser.add_argument("--overlap-two-count", type=int)
    parser.add_argument("--continuous-background-count", type=int)
    parser.add_argument("--music-scene-count", type=int)
    parser.add_argument("--co-onset-count", type=int)
    parser.add_argument("--max-effect-slug-reuse", type=int)
    parser.add_argument("--max-effect-family-reuse", type=int)
    parser.add_argument("--foreground-no-background-min", type=int)
    parser.add_argument("--foreground-no-background-max", type=int)
    parser.add_argument("--foreground-with-background-min", type=int)
    parser.add_argument("--foreground-with-background-max", type=int)
    parser.add_argument("--duration-min", type=int)
    parser.add_argument("--duration-max", type=int)
    parser.add_argument("--id-prefix")
    parser.add_argument(
        "--exclude-source-manifest", type=Path, action="append", default=[],
        help="Optional JSONL denylist; may be repeated. Matching source IDs are never selected.",
    )
    parser.add_argument("--mix-workers", type=int, default=16)
    parser.add_argument("--caption-workers", type=int, default=40)
    parser.add_argument("--caption-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    validate_recipe(recipe)
    args.asset_seed = args.asset_seed if args.asset_seed is not None else int(recipe_value(recipe, "seeds", "asset", 17))
    args.scene_seed = args.scene_seed if args.scene_seed is not None else int(recipe_value(recipe, "seeds", "scene", 29))
    args.effect_count = args.effect_count if args.effect_count is not None else int(recipe_value(recipe, "asset_pool", "effect_count", 80))
    args.ambience_count = args.ambience_count if args.ambience_count is not None else int(recipe_value(recipe, "asset_pool", "ambience_count", 8))
    args.music_count = args.music_count if args.music_count is not None else int(recipe_value(recipe, "asset_pool", "music_count", 8))
    args.scene_count = args.scene_count if args.scene_count is not None else int(recipe_value(recipe, "scenes", "count", 10))
    args.zh_speech_count = args.zh_speech_count if args.zh_speech_count is not None else int(recipe_value(recipe, "speech_pool", "zh_count", 4))
    args.ears_speech_count = args.ears_speech_count if args.ears_speech_count is not None else int(recipe_value(recipe, "speech_pool", "ears_count", 3))
    args.expresso_speech_count = args.expresso_speech_count if args.expresso_speech_count is not None else int(recipe_value(recipe, "speech_pool", "expresso_count", 1))
    args.zh_scene_speech_count = args.zh_scene_speech_count if args.zh_scene_speech_count is not None else int(recipe_value(recipe, "scenes", "zh_speech_count", 4))
    args.en_scene_speech_count = args.en_scene_speech_count if args.en_scene_speech_count is not None else int(recipe_value(recipe, "scenes", "en_speech_count", 4))
    args.zh_scene_no_speech_count = args.zh_scene_no_speech_count if args.zh_scene_no_speech_count is not None else int(recipe_value(recipe, "scenes", "zh_no_speech_count", 1))
    args.en_scene_no_speech_count = args.en_scene_no_speech_count if args.en_scene_no_speech_count is not None else int(recipe_value(recipe, "scenes", "en_no_speech_count", 1))
    args.overlap_zero_count = args.overlap_zero_count if args.overlap_zero_count is not None else int(recipe_value(recipe, "scenes", "overlap_zero_count", 3))
    args.overlap_one_count = args.overlap_one_count if args.overlap_one_count is not None else int(recipe_value(recipe, "scenes", "overlap_one_count", 5))
    args.overlap_two_count = args.overlap_two_count if args.overlap_two_count is not None else int(recipe_value(recipe, "scenes", "overlap_two_count", 2))
    args.continuous_background_count = args.continuous_background_count if args.continuous_background_count is not None else int(recipe_value(recipe, "scenes", "continuous_background_count", 5))
    args.music_scene_count = args.music_scene_count if args.music_scene_count is not None else int(recipe_value(recipe, "scenes", "music_count", 4))
    args.co_onset_count = args.co_onset_count if args.co_onset_count is not None else int(recipe_value(recipe, "scenes", "co_onset_count", 1))
    args.max_effect_slug_reuse = args.max_effect_slug_reuse if args.max_effect_slug_reuse is not None else recipe_value(recipe, "scenes", "max_effect_slug_reuse", None)
    if args.max_effect_slug_reuse is not None:
        args.max_effect_slug_reuse = int(args.max_effect_slug_reuse)
    args.max_effect_family_reuse = args.max_effect_family_reuse if args.max_effect_family_reuse is not None else recipe_value(recipe, "scenes", "max_effect_family_reuse", None)
    if args.max_effect_family_reuse is not None:
        args.max_effect_family_reuse = int(args.max_effect_family_reuse)
    args.foreground_no_background_min = args.foreground_no_background_min if args.foreground_no_background_min is not None else int(recipe_value(recipe, "scenes", "foreground_no_background_min", 6))
    args.foreground_no_background_max = args.foreground_no_background_max if args.foreground_no_background_max is not None else int(recipe_value(recipe, "scenes", "foreground_no_background_max", 8))
    args.foreground_with_background_min = args.foreground_with_background_min if args.foreground_with_background_min is not None else int(recipe_value(recipe, "scenes", "foreground_with_background_min", 5))
    args.foreground_with_background_max = args.foreground_with_background_max if args.foreground_with_background_max is not None else int(recipe_value(recipe, "scenes", "foreground_with_background_max", 7))
    args.duration_min = args.duration_min if args.duration_min is not None else int(recipe_value(recipe, "scenes", "duration_min", 42))
    args.duration_max = args.duration_max if args.duration_max is not None else int(recipe_value(recipe, "scenes", "duration_max", 58))
    args.id_prefix = args.id_prefix or str(recipe_value(recipe, "output", "id_prefix", "route2_demo"))

    configured = {
        "schema_version": 1,
        "name": recipe.get("name", "route2_custom"),
        "seeds": {"asset": args.asset_seed, "scene": args.scene_seed},
        "asset_pool": {
            "effect_count": args.effect_count, "ambience_count": args.ambience_count,
            "music_count": args.music_count,
        },
        "speech_pool": {
            "zh_count": args.zh_speech_count, "ears_count": args.ears_speech_count,
            "expresso_count": args.expresso_speech_count,
        },
        "scenes": {
            "count": args.scene_count,
            "zh_speech_count": args.zh_scene_speech_count,
            "en_speech_count": args.en_scene_speech_count,
            "zh_no_speech_count": args.zh_scene_no_speech_count,
            "en_no_speech_count": args.en_scene_no_speech_count,
            "overlap_zero_count": args.overlap_zero_count,
            "overlap_one_count": args.overlap_one_count,
            "overlap_two_count": args.overlap_two_count,
            "continuous_background_count": args.continuous_background_count,
            "music_count": args.music_scene_count,
            "co_onset_count": args.co_onset_count,
            "max_effect_slug_reuse": args.max_effect_slug_reuse,
            "max_effect_family_reuse": args.max_effect_family_reuse,
            "foreground_no_background_min": args.foreground_no_background_min,
            "foreground_no_background_max": args.foreground_no_background_max,
            "foreground_with_background_min": args.foreground_with_background_min,
            "foreground_with_background_max": args.foreground_with_background_max,
            "duration_min": args.duration_min,
            "duration_max": args.duration_max,
        },
        "output": {"id_prefix": args.id_prefix},
    }
    validate_recipe(configured)

    stages = stage_slice(args.start_at, args.stop_after)
    root = args.output_dir.resolve()
    for name in ("assets", "logs", "mix"):
        (root / name).mkdir(parents=True, exist_ok=True)

    if "assets" in stages:
        audioset_shards = required_path(
            args.audioset_shards, "--audioset-shards", args.dry_run
        )
    else:
        audioset_shards = args.audioset_shards
    speech_inputs: dict[str, Path | None] = {
        "--zh-labels": args.zh_labels,
        "--zh-audio-root": args.zh_audio_root,
        "--paraspeechcaps-parquet": args.paraspeechcaps_parquet,
        "--ears-audio-root": args.ears_audio_root,
        "--expresso-audio-root": args.expresso_audio_root,
    }
    if "speech" in stages:
        speech_inputs = {
            flag: required_path(path, flag, args.dry_run)
            for flag, path in speech_inputs.items()
        }
    if "natural_captions" in stages:
        if not args.caption_base_url:
            raise ValueError("--caption-base-url is required for natural caption rewriting")
        if not args.caption_model:
            raise ValueError("--caption-model is required for natural caption rewriting")

    assets = root / "assets" / "source_assets.jsonl"
    asset_selection = root / "assets" / "source_assets_selection.jsonl"
    speech = root / "speech_manifest.jsonl"
    scenes = root / "scene_graphs.jsonl"
    timeline_pairs = root / "pairs_timeline_captions.jsonl"
    caption_jobs = root / "caption_rewrite_jobs.jsonl"
    rewritten_pairs = root / "pairs_natural_captions.jsonl"
    final_pairs = root / "pairs.jsonl"
    driver = Driver(root, args.dry_run)

    if "assets" in stages:
        driver.run(
            "build_asset_pool",
            command(
                "build_asset_pool.py",
                "--shards",
                audioset_shards,
                "--output-dir",
                root / "assets" / "audio",
                "--manifest",
                assets,
                "--selection-manifest",
                asset_selection,
                "--effect-count",
                args.effect_count,
                "--ambience-count",
                args.ambience_count,
                "--music-count",
                args.music_count,
                "--seed",
                args.asset_seed,
                *(sum((["--exclude-source-manifest", path] for path in args.exclude_source_manifest), [])),
            ),
        )

    if "speech" in stages:
        driver.run(
            "build_speech_pool",
            command(
                "build_speech_pool.py",
                "--zh-labels",
                speech_inputs["--zh-labels"],
                "--zh-audio-root",
                speech_inputs["--zh-audio-root"],
                "--paraspeechcaps-parquet",
                speech_inputs["--paraspeechcaps-parquet"],
                "--ears-audio-root",
                speech_inputs["--ears-audio-root"],
                "--expresso-audio-root",
                speech_inputs["--expresso-audio-root"],
                "--output",
                speech,
                "--summary",
                root / "speech_manifest_summary.json",
                "--seed",
                args.scene_seed,
                "--zh-count",
                args.zh_speech_count,
                "--ears-count",
                args.ears_speech_count,
                "--expresso-count",
                args.expresso_speech_count,
                *(sum((["--exclude-source-manifest", path] for path in args.exclude_source_manifest), [])),
            ),
        )

    if "scenes" in stages:
        driver.run(
            "generate_scene_graphs",
            command(
                "generate_scene_graphs.py",
                "--asset-manifest",
                assets,
                "--speech-manifest",
                speech,
                "--output",
                scenes,
                "--usage-report",
                root / "source_usage_report.json",
                "--seed",
                args.scene_seed,
                "--count",
                args.scene_count,
                "--zh-speech-count", args.zh_scene_speech_count,
                "--en-speech-count", args.en_scene_speech_count,
                "--zh-no-speech-count", args.zh_scene_no_speech_count,
                "--en-no-speech-count", args.en_scene_no_speech_count,
                "--overlap-zero-count", args.overlap_zero_count,
                "--overlap-one-count", args.overlap_one_count,
                "--overlap-two-count", args.overlap_two_count,
                "--continuous-background-count", args.continuous_background_count,
                "--music-count", args.music_scene_count,
                "--co-onset-count", args.co_onset_count,
                *([] if args.max_effect_slug_reuse is None else ["--max-effect-slug-reuse", args.max_effect_slug_reuse]),
                *([] if args.max_effect_family_reuse is None else ["--max-effect-family-reuse", args.max_effect_family_reuse]),
                "--foreground-no-background-min", args.foreground_no_background_min,
                "--foreground-no-background-max", args.foreground_no_background_max,
                "--foreground-with-background-min", args.foreground_with_background_min,
                "--foreground-with-background-max", args.foreground_with_background_max,
                "--duration-min", args.duration_min,
                "--duration-max", args.duration_max,
                "--id-prefix", args.id_prefix,
            ),
        )

    if "mix" in stages:
        driver.run(
            "deterministic_mix",
            command(
                "compose_scene_graphs.py",
                "--scene-graphs",
                scenes,
                "--output-dir",
                root / "mix",
                "--resume",
                "--workers",
                args.mix_workers,
                "--fast",
            ),
        )

    if "mix_qc" in stages:
        driver.run(
            "mix_qc_repair",
            command(
                "repair_mix_qc.py",
                "--scene-graphs",
                scenes,
                "--asset-manifest",
                assets,
                "--output-dir",
                root / "mix",
                "--workers",
                args.mix_workers,
            ),
        )

    if "timeline_captions" in stages:
        driver.run(
            "prepare_timeline_captions",
            command(
                "prepare_timeline_captions.py",
                "--input",
                root / "mix" / "pairs.jsonl",
                "--scene-graphs",
                scenes,
                "--output",
                timeline_pairs,
            ),
        )

    if "natural_captions" in stages:
        driver.run(
            "rewrite_natural_captions",
            command(
                "rewrite_natural_captions.py",
                "--input",
                timeline_pairs,
                "--jobs-output",
                caption_jobs,
                "--output",
                rewritten_pairs,
                "--base-url",
                args.caption_base_url,
                "--model",
                args.caption_model,
                "--workers",
                args.caption_workers,
                "--attempts",
                args.caption_attempts,
                "--resume",
            ),
        )

    if "validate" in stages:
        driver.run(
            "validate_and_finalize",
            command(
                "validate_and_finalize.py",
                "--input",
                rewritten_pairs,
                "--output",
                final_pairs,
                "--report",
                root / "validation_report.json",
                "--expected-count",
                args.scene_count,
                "--expected-zh-count",
                args.zh_scene_speech_count + args.zh_scene_no_speech_count,
                "--expected-en-count",
                args.en_scene_speech_count + args.en_scene_no_speech_count,
                "--expected-zh-speech-count",
                args.zh_scene_speech_count,
                "--expected-en-speech-count",
                args.en_scene_speech_count,
                "--expected-foreground-min",
                min(args.foreground_no_background_min, args.foreground_with_background_min),
                "--expected-foreground-max",
                max(args.foreground_no_background_max, args.foreground_with_background_max, 5 if args.overlap_two_count else 0),
                *([] if args.max_effect_slug_reuse is None else ["--max-effect-slug-reuse", args.max_effect_slug_reuse]),
                *([] if args.max_effect_family_reuse is None else ["--max-effect-family-reuse", args.max_effect_family_reuse]),
            ),
        )

    summary = {
        "stages": list(stages),
        "final_manifest": str(final_pairs),
        "scene_graphs": str(scenes),
        "mix_directory": str(root / "mix"),
        "scene_count": args.scene_count,
        "recipe": configured,
        "dry_run": args.dry_run,
    }
    (root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
