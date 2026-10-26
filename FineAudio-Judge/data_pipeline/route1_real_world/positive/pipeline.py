"""Build Route 1 positive audio--caption pairs from real-world media.

The driver accepts the built-in Vript/FineVideo adapters or user-provided
JSONL manifests containing arbitrary video and/or audio datasets.  It runs the
same audio-grounded selection and captioning protocol for every input dataset,
then optionally applies bounded speaker-alignment patches.  Exactly one final
audio--caption manifest is published; intermediate variants remain auditable
but are not separate release products.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
STAGES = HERE / "stages"
STAGE_ORDER = (
    "ingest",
    "preflight",
    "language",
    "screen",
    "verify",
    "select",
    "detective",
    "finalize",
    "speaker_align",
    "publish",
)


def selected_stages(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGE_ORDER.index(start_at)
    stop = STAGE_ORDER.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    return STAGE_ORDER[start : stop + 1]


def requested_datasets(value: str) -> tuple[str, ...] | None:
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    if normalized == "both":
        return ("vript", "finevideo")
    values = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    if not values:
        raise ValueError("--dataset must be 'all', 'both', or a comma-separated dataset list")
    return values


def manifest_datasets(paths: Sequence[Path]) -> tuple[str, ...]:
    values: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"--input-manifest does not exist: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                dataset = str(json.loads(line).get("dataset") or "").strip().lower()
                if not dataset:
                    raise ValueError(f"every input-manifest row needs dataset: {path}")
                values.append(dataset)
    return tuple(dict.fromkeys(values))


def resolve_datasets(args: argparse.Namespace, root: Path, stages: tuple[str, ...]) -> tuple[str, ...]:
    available: list[str] = []
    if "ingest" in stages:
        if args.vript_root:
            available.append("vript")
        if args.finevideo_data_dir:
            available.append("finevideo")
        available.extend(manifest_datasets(args.input_manifest))
    else:
        available.extend(
            path.parent.name
            for path in sorted((root / "data").glob("*/source_manifest.jsonl"))
        )
    available = list(dict.fromkeys(available))
    requested = requested_datasets(args.dataset)
    selected = available if requested is None else list(requested)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"selected datasets have no input/source manifest: {missing}")
    if not selected:
        raise ValueError("no datasets selected; provide a built-in root or --input-manifest")
    return tuple(selected)


def require(value: str | Path | None, flag: str, *, dry_run: bool = False) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{flag} is required for the selected stages")
    if isinstance(value, Path) and not dry_run and not value.exists():
        raise FileNotFoundError(f"{flag} does not exist: {value}")
    return str(value)


class Driver:
    def __init__(self, root: Path, dry_run: bool, gpu_sets: dict[str, str]) -> None:
        self.root = root
        self.dry_run = dry_run
        self.gpu_sets = gpu_sets
        self.logs = root / "logs" / "audio_grounded"
        if not dry_run:
            self.logs.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        label: str,
        argv: Sequence[str],
        *,
        model_job: str | None = None,
    ) -> None:
        print(f"[{label}] {shlex.join(argv)}", flush=True)
        if self.dry_run:
            return
        env = dict(os.environ)
        env["AUDIOFLOW_PIPELINE_CONFIG"] = str(
            self.root / "resolved_config.json"
        )
        if model_job and self.gpu_sets.get(model_job):
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_sets[model_job]
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        log_path = self.logs / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                list(argv), env=env, stdout=log, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"Stage {label!r} failed with exit code {result.returncode}; "
                f"inspect {log_path}"
            )


def write_resolved_config(
    args: argparse.Namespace, root: Path, datasets: tuple[str, ...]
) -> tuple[Path, dict]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bounds = config["selection"]["duration_seconds"]
    if args.min_duration is not None:
        bounds["min"] = args.min_duration
    if args.max_duration is not None:
        bounds["max"] = args.max_duration
    if float(bounds["min"]) < 0 or float(bounds["min"]) > float(bounds["max"]):
        raise ValueError("duration bounds must satisfy 0 <= min <= max")
    config["selection"]["buffer_per_dataset"] = args.buffer_per_dataset
    config["selection"]["target_per_dataset"] = args.target_per_dataset
    config["datasets"] = list(datasets)
    config["language_gate"]["whisper_model"] = args.whisper_model
    config["language_gate"]["voxlingua_model"] = args.voxlingua_model
    config["generation"]["omni_model"] = args.omni_model or ""
    config["generation"]["text_model"] = args.text_model or ""
    config["generation"]["omni_tensor_parallel"] = args.omni_tp
    config["generation"]["text_tensor_parallel"] = args.text_tp
    config["speaker_alignment"]["whisper_word_model"] = args.whisper_model
    config["speaker_alignment"]["speaker_embedding_model"] = args.speaker_embedding_model
    config["speaker_alignment"]["enabled"] = args.with_speaker_align
    config["executables"].update(
        {
            "default_python": args.default_python,
            "language_python": args.language_python,
            "omni_python": args.omni_python,
            "text_python": args.text_python,
            "speaker_python": args.speaker_python,
            "demucs_python": args.demucs_python,
        }
    )
    resolved = root / "resolved_config.json"
    if not args.dry_run:
        resolved.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return resolved, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vript-root", type=Path)
    parser.add_argument("--finevideo-data-dir", type=Path)
    parser.add_argument(
        "--input-manifest", action="append", type=Path, default=[],
        help="JSONL manifest(s) with id, dataset, and audio_path and/or video_path.",
    )
    parser.add_argument("--omni-model", default=os.environ.get("FINEAUDIO_OMNI_MODEL"))
    parser.add_argument("--text-model", default=os.environ.get("FINEAUDIO_TEXT_MODEL"))
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument(
        "--voxlingua-model",
        default="speechbrain/lang-id-voxlingua107-ecapa",
    )
    parser.add_argument(
        "--speaker-embedding-model",
        default="iic/speech_ecapa-tdnn_sv_en_voxceleb_16k",
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--start-at", choices=STAGE_ORDER, default="ingest")
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default="publish")
    parser.add_argument(
        "--dataset", default="all",
        help="all, both, one dataset name, or a comma-separated dataset list.",
    )
    parser.add_argument("--candidate-limit", type=int, default=2500)
    parser.add_argument(
        "--min-duration", type=float,
        help="Override selection.duration_seconds.min from the config.",
    )
    parser.add_argument(
        "--max-duration", type=float,
        help="Override selection.duration_seconds.max from the config.",
    )
    parser.add_argument("--buffer-per-dataset", type=int, default=1300)
    parser.add_argument("--target-per-dataset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--omni-tp", type=int, default=2)
    parser.add_argument("--text-tp", type=int, default=2)
    parser.add_argument("--omni-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--text-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--speaker-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--default-python", default=sys.executable)
    parser.add_argument("--language-python", default=sys.executable)
    parser.add_argument("--omni-python", default=sys.executable)
    parser.add_argument("--text-python", default=sys.executable)
    parser.add_argument("--speaker-python", default=sys.executable)
    parser.add_argument("--demucs-python", default=sys.executable)
    parser.add_argument("--speaker-ids-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--with-speaker-align",
        dest="with_speaker_align",
        action="store_true",
        default=True,
        help="Run speaker alignment (enabled by default).",
    )
    parser.add_argument(
        "--without-speaker-align",
        dest="with_speaker_align",
        action="store_false",
        help="Skip speaker alignment and publish only the base audio-grounded captions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.config = args.config.resolve()
    if not args.config.is_file():
        raise FileNotFoundError(f"configuration does not exist: {args.config}")
    root = args.output_dir.resolve()
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        for directory in ("work", "logs", "outputs", "data", "combined", "reports"):
            (root / directory).mkdir(parents=True, exist_ok=True)
    stages = selected_stages(args.start_at, args.stop_after)
    datasets = resolve_datasets(args, root, stages)
    if {"screen", "verify", "detective", "speaker_align"}.intersection(stages):
        require(args.omni_model, "--omni-model")
    if "finalize" in stages:
        require(args.text_model, "--text-model")

    config_path, resolved_config = write_resolved_config(args, root, datasets)
    duration_bounds = resolved_config["selection"]["duration_seconds"]
    minimum_duration = str(duration_bounds["min"])
    maximum_duration = str(duration_bounds["max"])
    driver = Driver(
        root,
        args.dry_run,
        {"omni": args.omni_gpus, "text": args.text_gpus, "speaker": args.speaker_gpus},
    )

    if "ingest" in stages:
        if "vript" in datasets and args.vript_root:
            driver.run("discover_vript", [
                args.default_python, str(STAGES / "discover_vript_candidates.py"),
                "--vript-root", str(args.vript_root),
                "--output", str(root / "work" / "vript_candidates.jsonl"),
                "--audit-output", str(root / "work" / "vript_candidates_audit.jsonl"),
                "--limit", str(args.candidate_limit), "--min-duration", minimum_duration,
                "--max-duration", maximum_duration, "--max-per-source", "1",
            ])
            driver.run("prepare_vript", [
                args.default_python, str(STAGES / "prepare_samples.py"),
                "--root", str(root), "--dataset", "vript", "--vript-selection",
                str(root / "work" / "vript_candidates.jsonl"),
            ])
        if "finevideo" in datasets and args.finevideo_data_dir:
            driver.run("discover_finevideo", [
                args.default_python, str(STAGES / "discover_finevideo_candidates.py"),
                "--data-dir", str(args.finevideo_data_dir),
                "--output", str(root / "work" / "finevideo_candidates_all.jsonl"),
                "--min-duration", minimum_duration, "--max-duration", maximum_duration,
                "--max-per-video", "1",
            ])
            driver.run("sample_finevideo", [
                args.default_python, str(STAGES / "sample_candidate_manifest.py"),
                "--input", str(root / "work" / "finevideo_candidates_all.jsonl"),
                "--output", str(root / "work" / "finevideo_candidates.jsonl"),
                "--limit", str(args.candidate_limit), "--seed", str(args.seed),
            ])
            driver.run("prepare_finevideo", [
                args.default_python, str(STAGES / "prepare_samples.py"),
                "--root", str(root), "--dataset", "finevideo", "--finevideo-selection",
                str(root / "work" / "finevideo_candidates.jsonl"),
            ])
        for index, manifest in enumerate(args.input_manifest):
            driver.run(f"ingest_manifest_{index:02d}", [
                args.default_python, str(STAGES / "ingest_manifest.py"),
                "--run-root", str(root), "--manifest", str(manifest),
                "--datasets", ",".join(datasets),
            ])

    if "preflight" in stages:
        driver.run(
            "preflight",
            [args.default_python, str(STAGES / "preflight.py"), "--run-root", str(root),
             "--config", str(config_path), "--dataset", ",".join(datasets)],
        )

    for dataset in datasets:
        if "language" in stages:
            command = [
                args.language_python, str(STAGES / "language_gate.py"),
                "--run-root", str(root), "--config", str(config_path),
                "--dataset", dataset, "--device", args.device,
            ]
            if args.overwrite:
                command.append("--overwrite")
            driver.run(f"language_{dataset}", command, model_job="speaker")

        for stage in ("screen", "verify"):
            if stage in stages:
                command = [
                    args.omni_python, str(STAGES / "screen_verify.py"),
                    "--run-root", str(root), "--config", str(config_path),
                    "--dataset", dataset, "--stage", stage,
                ]
                if args.overwrite:
                    command.append("--overwrite")
                driver.run(f"{stage}_{dataset}", command, model_job="omni")

        if "select" in stages:
            driver.run(
                f"select_{dataset}",
                [
                    args.default_python, str(STAGES / "select_candidates.py"),
                    "--root", str(root), "--config", str(config_path),
                    "--dataset", dataset, "--target", str(args.buffer_per_dataset),
                ],
            )

        if "detective" in stages:
            command = [
                args.omni_python, str(STAGES / "detective.py"),
                "--run-root", str(root), "--config", str(config_path),
                "--dataset", dataset,
            ]
            if args.overwrite:
                command.append("--overwrite")
            driver.run(f"detective_{dataset}", command, model_job="omni")

        if "finalize" in stages:
            command = [
                args.text_python, str(STAGES / "finalize.py"),
                "--run-root", str(root), "--config", str(config_path),
                "--dataset", dataset,
            ]
            if args.overwrite:
                command.append("--overwrite")
            driver.run(f"finalize_{dataset}", command, model_job="text")

        if "speaker_align" in stages and args.with_speaker_align:
            command = [
                args.speaker_python, str(STAGES / "speaker_align.py"),
                "--run-root", str(root), "--config", str(config_path),
                "--dataset", dataset, "--step", "all", "--device", args.device,
            ]
            if args.speaker_ids_file:
                command.extend(["--ids-file", str(args.speaker_ids_file)])
            if args.overwrite:
                command.append("--overwrite")
            driver.run(f"speaker_align_{dataset}", command, model_job="speaker")

    if "publish" in stages:
        driver.run(
            "publish",
            [args.default_python, str(STAGES / "publish.py"), "--run-root", str(root),
             "--config", str(config_path)],
        )


if __name__ == "__main__":
    main()
