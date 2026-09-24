#!/usr/bin/env python3
"""Load and fingerprint versioned prompt/schema packs for the ASR agent."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    prompt: str
    schema: dict[str, Any]
    max_tokens: int


@dataclass(frozen=True)
class PromptPack:
    pack_id: str
    root: Path
    sha256: str
    summary: StageSpec
    scan: StageSpec
    check: StageSpec
    json_format_prompt: str

    def scan_schema(self, k: int) -> dict[str, Any]:
        """Bind the pipeline's per-loop candidate count into the scan schema."""
        schema = copy.deepcopy(self.scan.schema)
        schema["properties"]["suspects"]["maxItems"] = k
        return schema

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt_pack_id": self.pack_id,
            "prompt_pack_path": str(self.root),
            "prompt_pack_sha256": self.sha256,
            "stage_max_tokens": {
                "summary": self.summary.max_tokens,
                "scan": self.scan.max_tokens,
                "check": self.check.max_tokens,
            },
        }


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"prompt pack is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _pack_file(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must name a file")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} escapes prompt pack: {value}")
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _prompt(root: Path, value: Any, field: str) -> str:
    text = _pack_file(root, value, field).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{field} is empty")
    return text


def _schema(root: Path, value: Any, field: str) -> dict[str, Any]:
    schema = json.loads(_pack_file(root, value, field).read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError(f"{field} must contain an object JSON schema")
    return schema


def load_prompt_pack(path: Path) -> PromptPack:
    root = path.resolve()
    manifest_path = root / "pack.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("prompt pack format_version must be 1")
    pack_id = manifest.get("id")
    if not isinstance(pack_id, str) or not pack_id:
        raise ValueError("prompt pack id must be a non-empty string")
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("prompt pack stages must be an object")

    loaded: dict[str, StageSpec] = {}
    for name in ("summary", "scan", "check"):
        stage = stages.get(name)
        if not isinstance(stage, dict):
            raise ValueError(f"prompt pack is missing stage {name}")
        max_tokens = stage.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"stage {name} max_tokens must be a positive integer")
        loaded[name] = StageSpec(
            prompt=_prompt(root, stage.get("prompt"), f"stages.{name}.prompt"),
            schema=_schema(root, stage.get("schema"), f"stages.{name}.schema"),
            max_tokens=max_tokens,
        )

    scan_items = (
        loaded["scan"].schema.get("properties", {})
        .get("suspects", {})
        .get("items")
    )
    if not isinstance(scan_items, dict):
        raise ValueError("scan schema must define properties.suspects.items")

    return PromptPack(
        pack_id=pack_id,
        root=root,
        sha256=tree_sha256(root),
        summary=loaded["summary"],
        scan=loaded["scan"],
        check=loaded["check"],
        json_format_prompt=(
            _prompt(root, manifest["json_format_prompt"], "json_format_prompt")
            if "json_format_prompt" in manifest
            else ""
        ),
    )
