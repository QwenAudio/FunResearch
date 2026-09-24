#!/usr/bin/env python3
"""Run agent.py over one GigaSpeechBench Vertical-Domain language split.

Each LLM endpoint receives a configurable number of deterministic workers. A bad
edit never stops the batch because references are not read here. Infrastructure
failures are recorded, retried, and the remaining recordings continue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from agentic_ger.config import load_prompt_pack
from agentic_ger.utils import PACKAGE_ROOT, REPO_ROOT, service_headers


ROOT = REPO_ROOT
AGENT = PACKAGE_ROOT / "agent.py"
AGENT_CONFIG = PACKAGE_ROOT / "config.py"
DEFAULT_PROMPT_PACK = ROOT / "prompts/zh"
DATA_ROOT = Path(
    "data/prepared/Vertical-Domain"
)
DEFAULT_RUN_ROOT = ROOT / "runs/zh"
DEFAULT_LLM_URLS = (
    "http://127.0.0.1:8002/v1/chat/completions",
)
DEFAULT_ASR_URL = "http://127.0.0.1:7879/v1/chat/completions"
REQUIRED_LLM_CONTEXT = 65_536
WORKERS_PER_LLM = 10
EXPECTED_RECORDINGS = {"CH": 524, "EN": 387}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fingerprint(inference_config: dict[str, Any]) -> str:
    """Hash every setting that can affect a completed recording."""
    payload = json.dumps(
        inference_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_full_context_services(
    llm_urls: list[str],
    required_context: int = REQUIRED_LLM_CONTEXT,
    expected_model: str = "qwen3.8-27b",
) -> None:
    for chat_url in llm_urls:
        models_url = chat_url.split("/v1/chat/completions", 1)[0] + "/v1/models"
        response = requests.get(models_url, headers=service_headers("correction"), timeout=10)
        response.raise_for_status()
        models = response.json().get("data") or []
        model_ids = {str(model.get("id") or "") for model in models}
        if expected_model not in model_ids:
            raise SystemExit(
                f"{models_url} does not expose configured model {expected_model!r}; "
                f"available IDs: {sorted(model_ids)}"
            )
        available = max(
            (int(model.get("max_model_len") or 0) for model in models), default=0
        )
        if available < required_context:
            raise SystemExit(
                f"{models_url} exposes max_model_len={available}; the full-transcript "
                f"batch requires at least {required_context}"
            )


def require_asr_service(asr_url: str, expected_model: str) -> None:
    models_url = asr_url.split("/v1/chat/completions", 1)[0] + "/v1/models"
    response = requests.get(models_url, headers=service_headers("asr"), timeout=10)
    response.raise_for_status()
    models = response.json().get("data") or []
    model_ids = {str(model.get("id") or "") for model in models}
    if expected_model not in model_ids:
        raise SystemExit(
            f"{models_url} does not expose configured model {expected_model!r}; "
            f"available IDs: {sorted(model_ids)}"
        )


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def discover_manifest(
    data_root: Path,
    language_suffix: str = "CH",
    baseline_system: str = "FunASR-Realtime",
) -> list[dict[str, str]]:
    rows = []
    domains = sorted(
        path.name
        for path in data_root.glob(f"???-{language_suffix}")
        if path.is_dir()
    )
    if len(domains) != 12:
        raise SystemExit(
            f"expected 12 {language_suffix} domains, found {len(domains)}"
        )
    for domain in domains:
        model_root = data_root / domain / baseline_system
        for baseline in sorted((model_root / "hyp").glob(f"{domain}#*.json")):
            run_id = baseline.stem
            audio = model_root / "audio" / f"{run_id}.wav"
            rows.append(
                {"run_id": run_id, "audio": str(audio), "baseline": str(baseline)}
            )
    expected = EXPECTED_RECORDINGS[language_suffix]
    if len(rows) != expected or len({row["run_id"] for row in rows}) != expected:
        raise SystemExit(
            f"expected {expected} unique {language_suffix} recordings, "
            f"found {len(rows)}"
        )
    return rows


def load_explicit_manifest(path: Path) -> list[dict[str, str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    result = []
    for index, row in enumerate(rows, 1):
        run_id = str(row.get("run_id") or "")
        audio = Path(str(row.get("audio") or ""))
        baseline = Path(str(row.get("baseline") or ""))
        if not run_id:
            raise SystemExit(f"{path}:{index}: missing run_id")
        if run_id in seen:
            raise SystemExit(f"{path}:{index}: duplicate run_id {run_id}")
        if not audio.is_file():
            raise SystemExit(f"{path}:{index}: missing audio {audio}")
        if not baseline.is_file():
            raise SystemExit(f"{path}:{index}: missing baseline {baseline}")
        seen.add(run_id)
        result.append(
            {
                "run_id": run_id,
                "audio": str(audio.resolve()),
                "baseline": str(baseline.resolve()),
            }
        )
    if not result:
        raise SystemExit(f"explicit manifest is empty: {path}")
    return result


def audio_seconds(baseline: Path) -> float:
    data = json.loads(baseline.read_text(encoding="utf-8"))
    return max(
        (float(row.get("end") or 0) for row in data.get("segments") or []),
        default=0.0,
    )


def completed(run_dir: Path, expected_fingerprint: str) -> bool:
    summary_path = run_dir / "runner_summary.json"
    output = run_dir / "workspace/output"
    required = [
        summary_path,
        output / "baseline_transcript.json",
        output / "final_transcript.json",
        output / "trace.jsonl",
        output / "result.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (
            summary.get("required_outputs_ready") is True
            and summary.get("pi_return_code") == 0
            and summary.get("run_fingerprint") == expected_fingerprint
        )
    except (OSError, json.JSONDecodeError):
        return False


def attempted(run_dir: Path, expected_fingerprint: str) -> bool:
    """Return whether this revision already produced a terminal attempt."""
    summary_path = run_dir / "runner_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (
            summary.get("runner_status") in {"complete", "failed"}
            and summary.get("run_fingerprint") == expected_fingerprint
        )
    except (OSError, json.JSONDecodeError):
        return False


def run_one(
    row: dict[str, str],
    llm_url: str,
    args: argparse.Namespace,
    agent_hash: str,
    agent_config_hash: str,
    prompt_pack_hash: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    run_id = row["run_id"]
    run_dir = args.run_root / run_id
    if completed(run_dir, expected_fingerprint):
        return {"run_id": run_id, "status": "skipped_complete"}
    if sha256(AGENT) != agent_hash:
        raise RuntimeError("agent.py changed while the batch was running")
    if sha256(AGENT_CONFIG) != agent_config_hash:
        raise RuntimeError("config.py changed while the batch was running")
    if load_prompt_pack(args.prompt_pack).sha256 != prompt_pack_hash:
        raise RuntimeError("prompt pack changed while the batch was running")
    output = run_dir / "workspace/output"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "agentic_ger.agent",
        row["baseline"],
        row["audio"],
        str(output),
        "--llm-url",
        llm_url,
        "--llm-model",
        args.llm_model,
        "--asr-url",
        args.asr_url,
        "--asr-model",
        args.asr_model,
        "--max-loops",
        str(args.max_loops),
        "--max-patches",
        str(args.max_patches),
        "--k",
        str(args.k),
        "--thinking-max-tokens",
        str(args.thinking_max_tokens),
        "--llm-timeout",
        str(args.llm_timeout),
        "--prompt-pack",
        str(args.prompt_pack),
        "--expected-prompt-pack-sha256",
        prompt_pack_hash,
        "--expected-agent-sha256",
        agent_hash,
    ]
    if args.reasoning_effort:
        command.extend(("--reasoning-effort", args.reasoning_effort))
    if args.sampling_json:
        command.extend(("--sampling-json", args.sampling_json))
    if args.disable_thinking:
        command.append("--disable-thinking")
    started = time.monotonic()
    return_code, error, console = 1, "", ""
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        return_code = result.returncode
        console = (result.stdout + result.stderr)[-8000:]
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        error = f"timeout after {args.timeout}s"
        console = ((exc.stdout or "") + (exc.stderr or ""))[-8000:]
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    result_path = output / "result.json"
    agent_result: dict[str, Any] = {}
    if return_code == 0 and result_path.is_file():
        try:
            agent_result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            error = f"bad result.json: {exc}"
    ready = (
        return_code == 0
        and agent_result.get("status") == "complete"
        and (output / "baseline_transcript.json").is_file()
        and (output / "final_transcript.json").is_file()
        and (output / "trace.jsonl").is_file()
    )
    summary = {
        "run_id": run_id,
        "runner_status": "complete" if ready else "failed",
        "pi_return_code": 0 if ready else return_code,
        "required_outputs_ready": ready,
        "agent_sha256": agent_hash,
        "agent_config_sha256": agent_config_hash,
        "prompt_pack_id": args.prompt_pack_id,
        "prompt_pack_sha256": prompt_pack_hash,
        "run_fingerprint": expected_fingerprint,
        "llm_url": llm_url,
        "llm_model": args.llm_model,
        "thinking_enabled": not args.disable_thinking,
        "thinking_max_tokens": args.thinking_max_tokens,
        "reasoning_effort": (
            args.reasoning_effort if not args.disable_thinking else None
        ),
        "llm_timeout_seconds": args.llm_timeout,
        "asr_model": args.asr_model,
        "total_seconds": time.monotonic() - started,
        "audio_seconds": audio_seconds(Path(row["baseline"])),
        "counters": agent_result.get("counters") or {},
        "error": error,
        "console_tail": "" if ready else console,
    }
    write_json(run_dir / "runner_summary.json", summary)
    return {
        "run_id": run_id,
        "status": "complete" if ready else "failed",
        "accepted": int((agent_result.get("counters") or {}).get("accepted") or 0),
        "seconds": summary["total_seconds"],
        "error": error or (console[-500:] if not ready else ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        help="tracked high-level experiment config recorded with the run",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="explicit JSONL rows with run_id/audio/baseline; bypass discovery",
    )
    parser.add_argument(
        "--language-suffix",
        choices=tuple(EXPECTED_RECORDINGS),
        default="CH",
        help="Vertical-Domain language split; prompt and schema are unchanged",
    )
    parser.add_argument("--baseline-system", default="FunASR-Realtime")
    parser.add_argument("--llm-model", default="qwen3.8-27b")
    parser.add_argument(
        "--llm-url",
        action="append",
        default=[],
        help="OpenAI-compatible chat-completions URL; repeat for replicas",
    )
    parser.add_argument("--workers-per-llm", type=int, default=WORKERS_PER_LLM)
    parser.add_argument("--asr-model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    parser.add_argument("--max-loops", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=48)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=10_800)
    parser.add_argument("--retry-rounds", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="run one complete domain, for example ECM-CH; repeat for more than one",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="run only this recording; repeat the option for more than one",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--thinking-max-tokens", type=int, default=32_768)
    parser.add_argument(
        "--reasoning-effort",
        choices=("xhigh", "medium", "low"),
        default=None,
        help="optional model-native reasoning-effort control",
    )
    parser.add_argument(
        "--sampling-json",
        default="",
        help="model-specific sampling object forwarded to agent.py",
    )
    parser.add_argument("--llm-timeout", type=int, default=7_200)
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=DEFAULT_PROMPT_PACK,
        help="versioned directory containing prompts, schemas, and stage budgets",
    )
    parser.add_argument(
        "--required-llm-context", type=int, default=REQUIRED_LLM_CONTEXT
    )
    parser.add_argument(
        "--expected-agent-sha256",
        default="",
        help="refuse to run unless agent.py exactly matches this frozen version",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config_path = (
        args.experiment_config.resolve() if args.experiment_config else None
    )
    experiment_config = (
        json.loads(experiment_config_path.read_text(encoding="utf-8"))
        if experiment_config_path
        else None
    )
    experiment_config_sha256 = (
        run_fingerprint(experiment_config) if experiment_config else None
    )
    args.prompt_pack = args.prompt_pack.resolve()
    prompt_pack = load_prompt_pack(args.prompt_pack)
    args.prompt_pack_id = prompt_pack.pack_id
    llm_urls = args.llm_url or list(DEFAULT_LLM_URLS)
    if args.workers_per_llm < 1:
        raise SystemExit("--workers-per-llm must be at least 1")
    require_full_context_services(
        llm_urls, args.required_llm_context, args.llm_model
    )
    require_asr_service(args.asr_url, args.asr_model)
    args.run_root = args.run_root.resolve()
    args.data_root = args.data_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    explicit_manifest = args.manifest.resolve() if args.manifest else None
    rows = (
        load_explicit_manifest(explicit_manifest)
        if explicit_manifest
        else discover_manifest(
            args.data_root,
            language_suffix=args.language_suffix,
            baseline_system=args.baseline_system,
        )
    )
    if args.domain:
        wanted_domains = set(args.domain)
        known_domains = {row["run_id"].split("#", 1)[0] for row in rows}
        unknown_domains = wanted_domains - known_domains
        if unknown_domains:
            raise SystemExit(f"unknown domains: {sorted(unknown_domains)}")
        rows = [
            row
            for row in rows
            if row["run_id"].split("#", 1)[0] in wanted_domains
        ]
    if args.run_id:
        wanted = set(args.run_id)
        rows = [row for row in rows if row["run_id"] in wanted]
        missing = wanted - {row["run_id"] for row in rows}
        if missing:
            raise SystemExit(f"unknown run ids: {sorted(missing)}")
    if args.limit > 0:
        rows = rows[: args.limit]
    for row in rows:
        if not Path(row["audio"]).is_file():
            raise SystemExit(f"missing selected recording audio: {row['audio']}")
    agent_hash = sha256(AGENT)
    agent_config_hash = sha256(AGENT_CONFIG)
    prompt_pack_hash = prompt_pack.sha256
    inference_config = {
        "experiment_config_sha256": experiment_config_sha256,
        "agent_sha256": agent_hash,
        "agent_config_sha256": agent_config_hash,
        "prompt_pack_id": prompt_pack.pack_id,
        "prompt_pack_sha256": prompt_pack_hash,
        "llm_urls": llm_urls,
        "llm_model": args.llm_model,
        "thinking_enabled": not args.disable_thinking,
        "thinking_max_tokens": (
            args.thinking_max_tokens if not args.disable_thinking else None
        ),
        "reasoning_effort": (
            args.reasoning_effort if not args.disable_thinking else None
        ),
        "sampling": json.loads(args.sampling_json) if args.sampling_json else None,
        "llm_timeout_seconds": args.llm_timeout,
        "required_llm_context": args.required_llm_context,
        "workers_per_llm": args.workers_per_llm,
        "asr_url": args.asr_url,
        "asr_model": args.asr_model,
        "max_loops": args.max_loops,
        "max_patches": args.max_patches,
        "k": args.k,
        "task_timeout_seconds": args.timeout,
        "language_suffix": args.language_suffix,
        "baseline_system": args.baseline_system,
    }
    expected_fingerprint = run_fingerprint(inference_config)
    if args.expected_agent_sha256 and agent_hash != args.expected_agent_sha256:
        raise SystemExit(
            f"agent.py SHA-256 is {agent_hash}, expected "
            f"{args.expected_agent_sha256}"
        )
    existing_config_path = args.run_root / "run_config.json"
    if existing_config_path.is_file():
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        existing_fingerprint = existing_config.get("run_fingerprint")
        if existing_fingerprint and existing_fingerprint != expected_fingerprint:
            raise SystemExit(
                "run root belongs to a different code/config/prompt revision; "
                "use a new --run-root"
            )
    reproduction_root = args.run_root / "repro"
    source_snapshot = reproduction_root / "source"
    source_snapshot.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "src").rglob("*.py")):
        destination = source_snapshot / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for source in (
        ROOT / "requirements.txt",
        ROOT / "requirements-metrics.txt",
        ROOT / "requirements-data.txt",
        ROOT / "requirements.lock",
        ROOT / "pyproject.toml",
    ):
        shutil.copy2(source, source_snapshot / source.name)
    vendor_snapshot = source_snapshot / "vendor"
    vendor_snapshot.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "vendor").glob("*.py")):
        shutil.copy2(source, vendor_snapshot / source.name)
    shutil.copy2(ROOT / "vendor/README.md", vendor_snapshot / "README.md")
    shutil.copytree(ROOT / "prompts", source_snapshot / "prompts", dirs_exist_ok=True)
    configs_snapshot = source_snapshot / "configs"
    configs_snapshot.mkdir(exist_ok=True)
    for language in ("zh", "en"):
        shutil.copy2(ROOT / "configs" / f"{language}.json", configs_snapshot / f"{language}.json")
    prompt_snapshot = reproduction_root / "prompt_pack"
    shutil.copytree(args.prompt_pack, prompt_snapshot, dirs_exist_ok=True)
    if experiment_config_path:
        shutil.copy2(experiment_config_path, reproduction_root / "experiment.json")
    manifest_path = args.run_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(
        args.run_root / "run_config.json",
        {
            "experiment_config": (
                str(experiment_config_path) if experiment_config_path else None
            ),
            "experiment_config_sha256": experiment_config_sha256,
            "experiment_config_snapshot": experiment_config,
            "reproduction_snapshot": str(reproduction_root),
            "agent": str(AGENT),
            "agent_sha256": agent_hash,
            "agent_config": str(AGENT_CONFIG),
            "agent_config_sha256": agent_config_hash,
            **prompt_pack.metadata(),
            "run_fingerprint": expected_fingerprint,
            "inference_config": inference_config,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "tasks": len(rows),
            "domains": sorted({row["run_id"].split("#", 1)[0] for row in rows}),
            "data_root": str(args.data_root),
            "input_manifest": str(explicit_manifest) if explicit_manifest else None,
            "input_manifest_sha256": (
                sha256(explicit_manifest) if explicit_manifest else None
            ),
            "language_suffix": args.language_suffix,
            "baseline_system": args.baseline_system,
            "llm_urls": llm_urls,
            "llm_model": args.llm_model,
            "thinking_enabled": not args.disable_thinking,
            "thinking_max_tokens": args.thinking_max_tokens,
            "reasoning_effort": (
                args.reasoning_effort if not args.disable_thinking else None
            ),
            "llm_timeout_seconds": args.llm_timeout,
            "required_llm_context": args.required_llm_context,
            "workers_per_llm": args.workers_per_llm,
            "total_workers": len(llm_urls) * args.workers_per_llm,
            "asr_url": args.asr_url,
            "asr_model": args.asr_model,
            "max_loops": args.max_loops,
            "max_patches": args.max_patches,
            "k": args.k,
            "task_timeout_seconds": args.timeout,
            "retry_rounds": args.retry_rounds,
            "quality_stop": False,
            "reference_available_to_agent": False,
        },
    )

    lock = threading.Lock()
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()

    def progress() -> None:
        complete_count = sum(
            completed(args.run_root / row["run_id"], expected_fingerprint)
            for row in rows
        )
        attempted_count = sum(
            attempted(args.run_root / row["run_id"], expected_fingerprint)
            for row in rows
        )
        failed_rows = [
            value for value in results.values() if value.get("status") == "failed"
        ]
        write_json(
            args.run_root / "progress.json",
            {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "tasks": len(rows),
                "complete": complete_count,
                "failed": attempted_count - complete_count,
                "remaining": len(rows) - attempted_count,
                "failures_in_latest_attempt": len(failed_rows),
                "accepted_patches_in_latest_attempts": sum(
                    int(value.get("accepted") or 0) for value in results.values()
                ),
                "quality_stop": False,
            },
        )

    def run_shard(shard: list[dict[str, str]], url: str) -> None:
        for row in shard:
            try:
                result = run_one(
                    row,
                    url,
                    args,
                    agent_hash,
                    agent_config_hash,
                    prompt_pack_hash,
                    expected_fingerprint,
                )
            except Exception as exc:  # noqa: BLE001
                result = {"run_id": row["run_id"], "status": "failed", "error": str(exc)}
            with lock:
                results[row["run_id"]] = result
                progress()
                print(json.dumps(result, ensure_ascii=False), flush=True)

    progress()
    for round_index in range(args.retry_rounds + 1):
        pending = [
            row
            for row in rows
            if not (
                completed(args.run_root / row["run_id"], expected_fingerprint)
                or (
                    args.retry_rounds == 0
                    and attempted(
                        args.run_root / row["run_id"], expected_fingerprint
                    )
                )
            )
        ]
        if not pending:
            break
        print(
            json.dumps(
                {"round": round_index + 1, "pending": len(pending), "quality_stop": False},
                ensure_ascii=False,
            ),
            flush=True,
        )
        worker_urls = llm_urls * args.workers_per_llm
        shards = [
            pending[index :: len(worker_urls)] for index in range(len(worker_urls))
        ]
        with ThreadPoolExecutor(max_workers=len(worker_urls)) as pool:
            futures = [
                pool.submit(run_shard, shard, url)
                for shard, url in zip(shards, worker_urls, strict=True)
            ]
            for future in futures:
                future.result()

    final_rows = []
    for row in rows:
        summary_path = args.run_root / row["run_id"] / "runner_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {"run_id": row["run_id"], "runner_status": "missing"}
        )
        final_rows.append(summary)
    complete_count = sum(
        row.get("required_outputs_ready") is True
        and row.get("run_fingerprint") == expected_fingerprint
        for row in final_rows
    )
    summary = {
        "status": "complete" if complete_count == len(rows) else "incomplete",
        "tasks": len(rows),
        "complete": complete_count,
        "failed": len(rows) - complete_count,
        "batch_wall_seconds": time.monotonic() - started,
        "agent_sha256": agent_hash,
        "agent_config_sha256": agent_config_hash,
        "prompt_pack_id": prompt_pack.pack_id,
        "prompt_pack_sha256": prompt_pack_hash,
        "run_fingerprint": expected_fingerprint,
        "quality_stop": False,
        "results": final_rows,
    }
    write_json(args.run_root / "batch_summary.json", summary)
    progress()
    print(json.dumps({key: summary[key] for key in ("status", "tasks", "complete", "failed")}, ensure_ascii=False))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
