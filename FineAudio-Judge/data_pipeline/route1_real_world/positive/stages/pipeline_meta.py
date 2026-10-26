from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(
    os.environ.get("AUDIOFLOW_PIPELINE_CONFIG", SCRIPT_DIR.parent / "config.json")
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prompt_hashes(names: Iterable[str]) -> dict[str, str]:
    from prompts import PROMPT_BUILDERS

    values: dict[str, str] = {}
    for name in names:
        builder = PROMPT_BUILDERS[name]
        try:
            payload = inspect.getsource(builder).encode("utf-8")
        except (OSError, TypeError):
            payload = repr(builder).encode("utf-8")
        values[name] = sha256_bytes(payload)
    return values


def stage_provenance(
    *,
    stage: str,
    model: str,
    backend: str,
    prompt_names: Iterable[str],
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "run_id": os.environ.get("ASG_RUN_ID", "untracked"),
        "pipeline_name": config["pipeline_name"],
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prompts_sha256": prompt_hashes(prompt_names),
        "model": model,
        "backend": backend,
    }
