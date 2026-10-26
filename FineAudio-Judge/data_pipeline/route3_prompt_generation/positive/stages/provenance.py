from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    return run_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid JSON object at {path}:{line_number}")
            records.append(value)
    return records


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _snapshot_files(pipeline_root: Path, snapshot_root: Path) -> dict[str, str]:
    patterns = ("stages/*.py", "prompts/*.txt", "config/*.json", "config.json")
    checksums: dict[str, str] = {}
    for pattern in patterns:
        for source in sorted(pipeline_root.glob(pattern)):
            relative = source.relative_to(pipeline_root)
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            checksums[relative.as_posix()] = sha256_file(destination)
    atomic_write_json(snapshot_root / "checksums.json", checksums)
    return checksums


def init_run(
    pipeline_root: Path,
    runs_root: Path,
    run_id: str | None = None,
    audio_oscar_root: Path | None = None,
) -> tuple[str, Path]:
    run_id = validate_run_id(run_id or new_run_id())
    run_root = runs_root / run_id
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id:
            raise ValueError(f"Run manifest mismatch in {manifest_path}")
        return run_id, run_root

    # Directory creation is intentionally idempotent: an interrupted
    # initialization can leave these folders behind before run_manifest.json
    # is committed, and the supervisor must be able to resume safely.
    (run_root / "01_prompt_generation").mkdir(parents=True, exist_ok=True)
    (run_root / "02_oscar_generation/samples").mkdir(parents=True, exist_ok=True)
    (run_root / "03_dataset/audio").mkdir(parents=True, exist_ok=True)
    # A persistent launcher may create the log directory before the run
    # manifest is initialized. Treat that empty/pre-existing directory as a
    # valid resumable run state instead of failing initialization.
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    snapshot_root = run_root / "snapshots"
    checksums = _snapshot_files(pipeline_root, snapshot_root)

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "initialized",
        "origin": {
            "type": "qwen3.5_natural_scene_generation",
            "external_dataset": None,
            "description": (
                "Qwen3.5 creates new audio-centered scene prompts from the "
                "snapshotted generation instruction."
            ),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "software": {
            "path3_pipeline_root": str(pipeline_root.resolve()),
            "path3_git_commit": git_commit(pipeline_root),
            "audio_oscar_root": (
                str(audio_oscar_root.resolve()) if audio_oscar_root else None
            ),
            "audio_oscar_git_commit": (
                git_commit(audio_oscar_root) if audio_oscar_root else None
            ),
        },
        "snapshot": {
            "path": "snapshots",
            "checksums": checksums,
        },
        "stages": {
            "01_prompt_generation": {"status": "pending"},
            "02_oscar_generation": {"status": "pending"},
            "03_dataset": {"status": "pending"},
        },
        "paths": {
            "accepted_prompts": "01_prompt_generation/accepted_prompts.jsonl",
            "oscar_jobs": "02_oscar_generation/jobs.jsonl",
            "dataset": "03_dataset/dataset.jsonl",
        },
    }
    atomic_write_json(manifest_path, manifest)
    return run_id, run_root


def update_run_stage(
    run_root: Path,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_value = dict(manifest["stages"].get(stage, {}))
    stage_value.update(details)
    stage_value["status"] = status
    stage_value["updated_at"] = utc_now()
    manifest["stages"][stage] = stage_value
    manifest["status"] = "failed" if status == "failed" else "running"
    if all(
        value.get("status") == "completed"
        for value in manifest["stages"].values()
    ):
        manifest["status"] = "completed"
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
