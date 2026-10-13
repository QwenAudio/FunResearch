"""Initialization and weight audits for frozen-core backbone alignment."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Mapping

import torch

from .transfer import BACKBONE_ADAPTER_PREFIXES, TRANSFER_FORMAT, is_backbone_adapter_key


def overlay_shared_core(model, source: Mapping[str, Any]) -> dict[str, Any]:
    """Keep target projections intact and copy an exactly compatible source core."""
    if source.get("format") == TRANSFER_FORMAT:
        source_config = dict(source["shared_config"])
        source_state = dict(source["shared_core_state"])
        provenance = dict(source.get("provenance", {}))
    else:
        source_config = dict(source["config"])
        source_config.pop("audio_dim", None)
        source_config.pop("text_dim", None)
        source_state = {
            key: value for key, value in source["model_state"].items()
            if not is_backbone_adapter_key(key)
        }
        provenance = {
            key: source[key] for key in
            ("optimizer_step", "selection_manifest", "selected_hours", "note")
            if key in source
        }
    target_config = asdict(model.config)
    target_config.pop("audio_dim")
    target_config.pop("text_dim")
    if source_config != target_config:
        raise ValueError("source and target shared Attention Fusion configs differ")
    target_state = model.state_dict()
    expected = {key for key in target_state if not is_backbone_adapter_key(key)}
    if set(source_state) != expected:
        raise ValueError("source shared core keys do not exactly match the target core")
    for key, value in source_state.items():
        if value.shape != target_state[key].shape:
            raise ValueError(f"source shared core shape mismatch: {key}")
    target_state.update(source_state)
    model.load_state_dict(target_state, strict=True)
    return {
        "shared_core_tensors_loaded": len(source_state),
        "shared_core_parameters_loaded": sum(value.numel() for value in source_state.values()),
        "target_projection_tensors_preserved": sum(
            is_backbone_adapter_key(key) for key in target_state
        ),
        "source_provenance": provenance,
    }


def set_trainable_scope(model, scope: str) -> None:
    if scope not in ("all", "bias_head", "backbone_adapters"):
        raise ValueError(f"unknown trainable scope: {scope}")
    model.requires_grad_(scope == "all")
    if scope == "bias_head":
        model.bias_head.requires_grad_(True)
    elif scope == "backbone_adapters":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(is_backbone_adapter_key(name))
        prefixes = {
            name.split(".", 1)[0] + "." for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if prefixes != set(BACKBONE_ADAPTER_PREFIXES):
            raise RuntimeError("six-projection alignment scope is incomplete")


def state_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, value in state.items()
    }


def alignment_weight_audit(model, initial_hashes: Mapping[str, str]) -> dict[str, Any]:
    current = state_hashes(model.state_dict())
    if set(current) != set(initial_hashes):
        raise RuntimeError("alignment model state keys changed")
    changed = [key for key in current if current[key] != initial_hashes[key]]
    frozen_changed = [
        key for key, parameter in model.named_parameters()
        if not parameter.requires_grad and key in changed
    ]
    if frozen_changed:
        raise RuntimeError(f"frozen Fusion parameters changed: {frozen_changed}")
    projection_names = [key for key in current if is_backbone_adapter_key(key)]
    core_names = [key for key in current if not is_backbone_adapter_key(key)]
    return {
        "frozen_parameters_unchanged": not frozen_changed,
        "shared_core_unchanged": all(key not in changed for key in core_names),
        "shared_core_tensors": len(core_names),
        "projection_tensors": len(projection_names),
        "projection_tensors_changed": [key for key in changed if is_backbone_adapter_key(key)],
        "projection_modules_changed": sorted({
            key.split(".", 1)[0] for key in changed if is_backbone_adapter_key(key)
        }),
        "all_six_projection_modules_changed": all(
            any(key.startswith(prefix) for key in changed)
            for prefix in BACKBONE_ADAPTER_PREFIXES
        ),
        "initial_state_sha256": dict(initial_hashes),
        "current_state_sha256": current,
    }
