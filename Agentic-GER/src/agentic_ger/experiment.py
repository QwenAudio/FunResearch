#!/usr/bin/env python3
"""Run, evaluate, and render one configured GigaSpeechBench experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentic_ger.config import load_prompt_pack


from agentic_ger.utils import PACKAGE_ROOT, REPO_ROOT


ROOT = REPO_ROOT
BATCH_RUNNER = PACKAGE_ROOT / "runner.py"
AGENT = PACKAGE_ROOT / "agent.py"
AGENT_CONFIG = PACKAGE_ROOT / "config.py"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"configuration must contain one JSON object: {path}")
    return value


def object_value(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"configuration field {key!r} must be an object")
    return value


def string_value(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"configuration field {key!r} must be a non-empty string")
    return value


def integer_value(config: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SystemExit(
            f"configuration field {key!r} must be an integer >= {minimum}"
        )
    return value


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_repo_path(value: str, field: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise SystemExit(f"{field} must stay inside the repository: {value}")
    return path


def validate_config(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("format_version") != 1:
        raise SystemExit("configuration format_version must be 1")
    experiment_id = string_value(config, "id")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment_id) is None:
        raise SystemExit("configuration id must use lowercase letters, digits, _ or -")
    language = string_value(config, "language")
    if language not in {"CH", "EN"}:
        raise SystemExit("configuration language must be CH or EN")

    data = object_value(config, "data")
    models = object_value(config, "models")
    services = object_value(config, "services")
    flow = object_value(config, "flow")
    inference = object_value(config, "inference")
    runtime = object_value(config, "runtime")
    correction_urls = services.get("correction_urls")
    if (
        not isinstance(correction_urls, list)
        or not correction_urls
        or any(not isinstance(value, str) or not value for value in correction_urls)
    ):
        raise SystemExit("services.correction_urls must contain at least one URL")
    if not isinstance(inference.get("thinking"), bool):
        raise SystemExit("inference.thinking must be true or false")
    reasoning_effort = inference.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort not in {
        "low",
        "medium",
        "xhigh",
    }:
        raise SystemExit(
            "inference.reasoning_effort must be null, low, medium, or xhigh"
        )
    sampling_profiles = object_value(inference, "sampling")
    sampling_keys = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    }
    for mode in ("nonthinking", "thinking"):
        profile = object_value(sampling_profiles, mode)
        unsupported = set(profile) - sampling_keys
        if unsupported:
            raise SystemExit(
                f"inference.sampling.{mode} has unsupported fields: "
                f"{sorted(unsupported)}"
            )
        if not profile or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in profile.values()
        ):
            raise SystemExit(
                f"inference.sampling.{mode} must contain numeric parameters"
            )

    prompt_pack = resolve_repo_path(
        string_value(config, "prompt_pack"), "prompt_pack"
    )
    loaded_pack = load_prompt_pack(prompt_pack)
    data_root = Path(string_value(data, "root")).resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data.root does not exist: {data_root}")
    string_value(data, "baseline_system")
    string_value(models, "correction")
    string_value(models, "relisten")
    string_value(services, "relisten_url")
    integer_value(flow, "candidates_per_scan", 1)
    integer_value(flow, "max_loops", 1)
    integer_value(flow, "max_patches", 1)
    integer_value(inference, "thinking_max_tokens", 1)
    integer_value(runtime, "workers_per_correction_service", 1)
    integer_value(runtime, "required_context", 1)
    integer_value(runtime, "llm_timeout_seconds", 1)
    integer_value(runtime, "recording_timeout_seconds", 1)
    integer_value(runtime, "retry_rounds", 0)
    return {
        "path": path,
        "id": experiment_id,
        "language": language,
        "data": data,
        "models": models,
        "services": services,
        "flow": flow,
        "inference": inference,
        "runtime": runtime,
        "prompt_pack": prompt_pack,
        "prompt_pack_id": loaded_pack.pack_id,
        "prompt_pack_sha256": loaded_pack.sha256,
        "config_sha256": semantic_sha256(config),
    }


def default_run_root(config: dict[str, Any]) -> Path:
    revision = semantic_sha256(
        {
            "config_sha256": config["config_sha256"],
            "prompt_pack_sha256": config["prompt_pack_sha256"],
            "agent_sha256": file_sha256(AGENT),
            "agent_config_sha256": file_sha256(AGENT_CONFIG),
            "batch_runner_sha256": file_sha256(BATCH_RUNNER),
        }
    )[:12]
    return ROOT / "runs" / config["id"] / revision


def command_text(command: list[str]) -> str:
    return " ".join(json.dumps(value) if " " in value else value for value in command)


def terminal_batch(run_root: Path) -> bool:
    path = run_root / "batch_summary.json"
    if not path.is_file():
        return False
    batch = load_json(path)
    tasks = int(batch.get("tasks") or 0)
    complete = int(batch.get("complete") or 0)
    failed = int(batch.get("failed") or 0)
    return tasks > 0 and complete + failed == tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="tracked experiment JSON")
    parser.add_argument("--run-root", type=Path, help="override revisioned run root")
    parser.add_argument("--data-root", type=Path, help="override configured data root")
    parser.add_argument(
        "--prompt-pack", type=Path, help="override the configured prompt/schema pack"
    )
    parser.add_argument(
        "--llm-url", action="append", default=[], help="override correction URLs"
    )
    parser.add_argument("--asr-url", help="override the Qwen3-ASR URL")
    parser.add_argument("--baseline-system", help="override the configured ASR baseline")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking_override",
        action="store_const",
        const=True,
        help="enable thinking for this run",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking_override",
        action="store_const",
        const=False,
        help="disable thinking for this run",
    )
    parser.set_defaults(thinking_override=None)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--report-only", action="store_true", help="skip inference and rebuild reports"
    )
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    raw_config = load_json(config_path)
    effective_config = copy.deepcopy(raw_config)
    if args.prompt_pack:
        prompt_pack = args.prompt_pack.resolve()
        if not prompt_pack.is_relative_to(ROOT):
            raise SystemExit(f"prompt pack must stay inside the repository: {prompt_pack}")
        effective_config["prompt_pack"] = prompt_pack.relative_to(ROOT).as_posix()
    if args.data_root:
        object_value(effective_config, "data")["root"] = str(args.data_root.resolve())
    if args.baseline_system:
        object_value(effective_config, "data")["baseline_system"] = (
            args.baseline_system
        )
    if args.llm_url:
        object_value(effective_config, "services")["correction_urls"] = args.llm_url
    if args.asr_url:
        object_value(effective_config, "services")["relisten_url"] = args.asr_url
    if args.thinking_override is not None:
        object_value(effective_config, "inference")["thinking"] = (
            args.thinking_override
        )
    config = validate_config(config_path, effective_config)
    run_root = (args.run_root or default_run_root(config)).resolve()
    data_root = Path(config["data"]["root"]).resolve()
    llm_urls = list(config["services"]["correction_urls"])
    asr_url = str(config["services"]["relisten_url"])
    effective_config_path = run_root / "effective_experiment.json"

    batch_command = [
        sys.executable,
        "-u",
        "-m",
        "agentic_ger.runner",
        "--experiment-config",
        str(effective_config_path),
        "--run-root",
        str(run_root),
        "--data-root",
        str(data_root),
        "--language-suffix",
        config["language"],
        "--baseline-system",
        str(config["data"]["baseline_system"]),
        "--llm-model",
        str(config["models"]["correction"]),
        "--asr-model",
        str(config["models"]["relisten"]),
        "--asr-url",
        asr_url,
        "--prompt-pack",
        str(config["prompt_pack"]),
        "--k",
        str(config["flow"]["candidates_per_scan"]),
        "--max-loops",
        str(config["flow"]["max_loops"]),
        "--max-patches",
        str(config["flow"]["max_patches"]),
        "--thinking-max-tokens",
        str(config["inference"]["thinking_max_tokens"]),
        "--workers-per-llm",
        str(config["runtime"]["workers_per_correction_service"]),
        "--required-llm-context",
        str(config["runtime"]["required_context"]),
        "--llm-timeout",
        str(config["runtime"]["llm_timeout_seconds"]),
        "--timeout",
        str(config["runtime"]["recording_timeout_seconds"]),
        "--retry-rounds",
        str(config["runtime"]["retry_rounds"]),
    ]
    reasoning_effort = config["inference"].get("reasoning_effort")
    if reasoning_effort:
        batch_command.extend(("--reasoning-effort", str(reasoning_effort)))
    sampling_mode = (
        "thinking" if config["inference"]["thinking"] else "nonthinking"
    )
    batch_command.extend(
        (
            "--sampling-json",
            json.dumps(
                config["inference"]["sampling"][sampling_mode],
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    for url in llm_urls:
        batch_command.extend(("--llm-url", url))
    if not config["inference"]["thinking"]:
        batch_command.append("--disable-thinking")
    for domain in args.domain:
        batch_command.extend(("--domain", domain))
    for run_id in args.run_id:
        batch_command.extend(("--run-id", run_id))
    if args.limit:
        batch_command.extend(("--limit", str(args.limit)))
    evaluator = (
        "agentic_ger.evaluation.evaluate_zh"
        if config["language"] == "CH" else "agentic_ger.evaluation.evaluate_en"
    )
    evaluation_command = [
        sys.executable,
        "-m",
        evaluator,
        "--run-root",
        str(run_root),
        "--data-root",
        str(data_root),
        "--baseline-system",
        str(config["data"]["baseline_system"]),
        "--allow-incomplete",
    ]
    dashboard_command = [
        sys.executable,
        "-m",
        "agentic_ger.reporting.dashboard",
        "--run-root",
        str(run_root),
        "--output-root",
        str(run_root / "cases"),
        "--language",
        config["language"],
    ]

    print(f"experiment: {config['id']} ({config['language']})")
    print(f"prompt: {config['prompt_pack_id']} {config['prompt_pack_sha256'][:12]}")
    print(
        f"baseline: {config['data']['baseline_system']}; "
        f"thinking: {'on' if config['inference']['thinking'] else 'off'}"
    )
    print(f"run root: {run_root}")
    if args.dry_run:
        if not args.report_only:
            print(command_text(batch_command))
        print(command_text(evaluation_command))
        if not args.no_dashboard:
            print(command_text(dashboard_command))
        return

    if not args.report_only:
        run_root.mkdir(parents=True, exist_ok=True)
        if effective_config_path.is_file():
            existing_config = load_json(effective_config_path)
            if existing_config != effective_config:
                raise SystemExit(
                    "run root belongs to a different effective experiment config: "
                    f"{effective_config_path}"
                )
        else:
            effective_config_path.write_text(
                json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        result = subprocess.run(batch_command, check=False)
        if result.returncode and not terminal_batch(run_root):
            raise SystemExit(result.returncode)
    elif not terminal_batch(run_root):
        raise SystemExit(f"run is not ready for reporting: {run_root}")

    subprocess.run(evaluation_command, check=True)
    if not args.no_dashboard:
        subprocess.run(dashboard_command, check=True)
    print(f"evaluation: {run_root / 'evaluation/summary.md'}")
    if not args.no_dashboard:
        print(f"cases: {run_root / 'cases/index.html'}")
        print(
            "serve: "
            f"{sys.executable} -m agentic_ger.reporting.server "
            f"--directory {run_root / 'cases'} --port 12398"
        )


if __name__ == "__main__":
    main()
