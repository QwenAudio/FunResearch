#!/usr/bin/env python3
"""Export the three audited HotBias artifacts without training/private paths.

Only run this on checkpoints you trust. Input hashes pin the paper's evaluated
versions; output hashes are written to SHA256SUMS after serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_SHA256 = {
    "source": "486b9a36e5d8f594a79343364b2ed04f542121664eda9f08f1b20b75d3a3e70a",
    "nano": "9d5f9f5cf7cf4769fd9dca72604abab68178ae5b741fc90a890e653731be4a23",
    "firered": "b73e7619e1765833b72e37c7e33bd4d218365a94d3a8c9d5c784785a19977aee",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audited_load(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != EXPECTED_SHA256[role]:
        raise ValueError(
            f"{role} SHA256 mismatch: expected {EXPECTED_SHA256[role]}, got {actual}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{role} checkpoint must be a dict")
    return payload


def require_state(payload: dict[str, Any], role: str) -> dict[str, torch.Tensor]:
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{role} checkpoint has no model_state")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise TypeError(f"{role} model_state must contain tensor values")
    return state


def build_release_payloads(
    source: dict[str, Any], nano: dict[str, Any], firered: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if source.get("format") != "hotword_attention_fusion_transfer":
        raise ValueError("source is not a HotBias transfer package")
    shared = source.get("shared_core_state")
    source_adapter = source.get("source_adapter")
    if not isinstance(shared, dict) or not isinstance(source_adapter, dict):
        raise ValueError("source transfer package is incomplete")
    if len(shared) != 130 or len(source_adapter.get("state", {})) != 18:
        raise ValueError("unexpected source core/projection tensor count")

    nano_state = require_state(nano, "nano")
    firered_state = require_state(firered, "firered")
    if bool(nano.get("initialization_only")) or not bool(
        nano.get("ready_for_target_inference")
    ):
        raise ValueError("Nano checkpoint is not an aligned inference checkpoint")
    if bool(firered.get("initialization_only")) or not bool(
        firered.get("alignment_completed")
    ):
        raise ValueError("FireRed checkpoint is not a completed alignment")
    for target_name, target_state in (("nano", nano_state), ("firered", firered_state)):
        if any(
            key not in target_state or not torch.equal(value, target_state[key])
            for key, value in shared.items()
        ):
            raise ValueError(f"{target_name} does not contain the frozen source core")

    safe_source = {
        "format": source["format"],
        "format_version": source["format_version"],
        "model_class": source["model_class"],
        "shared_config": source["shared_config"],
        "shared_core_state": shared,
        "source_adapter": {
            "name": "Qwen3-based source ASR",
            "audio_dim": source_adapter["audio_dim"],
            "text_dim": source_adapter["text_dim"],
            "state": source_adapter["state"],
        },
        "adapter_prefixes": source["adapter_prefixes"],
        "usage": source["usage"],
    }
    safe_nano = {
        "config": nano["config"],
        "model_state": nano_state,
        "target_name": "Fun-ASR-Nano-2512",
        "training_scope": "six_backbone_adapters_only",
        "initialization_only": False,
        "ready_for_target_inference": True,
    }
    safe_firered = {
        "config": firered["config"],
        "model_state": firered_state,
        "target_name": "FireRedASR2-AED",
        "trainable_scope": "six_backbone_adapters_only",
        "initialization_only": False,
        "alignment_completed": True,
    }
    return {
        "source_transfer.pt": safe_source,
        "nano_aligned.pt": safe_nano,
        "firered_aligned.pt": safe_firered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-transfer", type=Path, required=True)
    parser.add_argument("--nano-checkpoint", type=Path, required=True)
    parser.add_argument("--firered-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = build_release_payloads(
        audited_load(args.source_transfer, "source"),
        audited_load(args.nano_checkpoint, "nano"),
        audited_load(args.firered_checkpoint, "firered"),
    )
    summary = {
        "core_tensors": len(payloads["source_transfer.pt"]["shared_core_state"]),
        "core_identical_in_both_targets": True,
        "input_sha256": EXPECTED_SHA256,
    }
    if args.verify_only:
        print(json.dumps(summary, indent=2))
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = [*payloads, "SHA256SUMS", "manifest.json"]
    existing = [name for name in output_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"output files already exist: {existing}")

    output_hashes: dict[str, str] = {}
    for name, payload in payloads.items():
        path = output_dir / name
        torch.save(payload, path)
        output_hashes[name] = sha256(path)
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in output_hashes.items()),
        encoding="utf-8",
    )
    summary["output_sha256"] = output_hashes
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
