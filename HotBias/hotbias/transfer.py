from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .attention_fusion import AttentionFusionConfig, FrozenFeatureAttentionFusion


TRANSFER_FORMAT = "hotword_attention_fusion_transfer"
TRANSFER_VERSION = 1

# These modules consume coordinates produced by a specific frozen ASR base.
# Everything else operates in the shared hidden space and is portable once a
# target base has learned these six projections.
BACKBONE_ADAPTER_PREFIXES = (
    "audio_projection.",
    "history_projection.",
    "decoder_projection.",
    "hotword_projection.",
    "current_token_projection.",
    "prefix_projection.",
)


def is_backbone_adapter_key(name: str) -> bool:
    return name.startswith(BACKBONE_ADAPTER_PREFIXES)


def split_attention_fusion_state(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a Fusion state dict into portable core and base-specific adapters."""

    shared_core: dict[str, torch.Tensor] = {}
    backbone_adapter: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        destination = backbone_adapter if is_backbone_adapter_key(name) else shared_core
        destination[name] = value.detach().cpu()
    return shared_core, backbone_adapter


def build_transferable_checkpoint(
    source_checkpoint: Mapping[str, Any],
    *,
    source_name: str,
    source_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a training checkpoint into a base-independent transfer package."""

    if "config" not in source_checkpoint or "model_state" not in source_checkpoint:
        raise ValueError("source checkpoint must contain config and model_state")
    config = AttentionFusionConfig(**dict(source_checkpoint["config"]))
    model = FrozenFeatureAttentionFusion(config)
    expected = model.state_dict()
    source_state = source_checkpoint["model_state"]
    if set(source_state) != set(expected):
        missing = sorted(set(expected) - set(source_state))
        unexpected = sorted(set(source_state) - set(expected))
        raise ValueError(
            f"source state mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    shape_errors = {
        name: (tuple(source_state[name].shape), tuple(expected[name].shape))
        for name in expected
        if source_state[name].shape != expected[name].shape
    }
    if shape_errors:
        raise ValueError(f"source state shape mismatch: {shape_errors}")

    shared_core, source_adapter = split_attention_fusion_state(source_state)
    shared_config = asdict(config)
    audio_dim = int(shared_config.pop("audio_dim"))
    text_dim = int(shared_config.pop("text_dim"))
    provenance_keys = (
        "optimizer_step",
        "next_global_sample",
        "selected_hours",
        "selection_manifest",
        "note",
        "frozen_dependencies",
    )
    provenance = {
        key: source_checkpoint[key]
        for key in provenance_keys
        if key in source_checkpoint
    }
    if source_checkpoint_path is not None:
        provenance["source_checkpoint_path"] = str(source_checkpoint_path)

    return {
        "format": TRANSFER_FORMAT,
        "format_version": TRANSFER_VERSION,
        "model_class": "FrozenFeatureAttentionFusion",
        "shared_config": shared_config,
        "shared_core_state": shared_core,
        "source_adapter": {
            "name": source_name,
            "audio_dim": audio_dim,
            "text_dim": text_dim,
            "state": source_adapter,
        },
        "adapter_prefixes": list(BACKBONE_ADAPTER_PREFIXES),
        "provenance": provenance,
        "usage": {
            "ready_for_source_inference": True,
            "ready_for_new_backbone_inference": False,
            "target_requirement": (
                "Initialize target-dimension adapters, rebuild tokenizer/Trie, "
                "and run target-base alignment/SFT before inference."
            ),
        },
    }


def load_transferable_checkpoint(
    checkpoint: Mapping[str, Any] | str | Path,
    *,
    audio_dim: int,
    text_dim: int,
    reuse_shape_compatible_source_adapter: bool = True,
) -> tuple[FrozenFeatureAttentionFusion, dict[str, Any]]:
    """Initialize a target-dimension Fusion model from a transfer package.

    Shape-compatible adapter values (biases and RMSNorm scales) can be reused,
    but dimension-dependent projection matrices remain freshly initialized.
    The returned model is a warm start and requires target-base training.
    """

    payload = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, (str, Path))
        else dict(checkpoint)
    )
    if payload.get("format") != TRANSFER_FORMAT:
        raise ValueError(f"unsupported transfer format: {payload.get('format')!r}")
    if int(payload.get("format_version", -1)) != TRANSFER_VERSION:
        raise ValueError(
            f"unsupported transfer version: {payload.get('format_version')!r}"
        )

    config_values = dict(payload["shared_config"])
    config_values.update(audio_dim=int(audio_dim), text_dim=int(text_dim))
    config = AttentionFusionConfig(**config_values)
    model = FrozenFeatureAttentionFusion(config)
    target_state = model.state_dict()

    load_state = dict(payload["shared_core_state"])
    reused_adapter_names: list[str] = []
    skipped_adapter_names: list[str] = []
    for name, value in payload["source_adapter"]["state"].items():
        if (
            reuse_shape_compatible_source_adapter
            and name in target_state
            and value.shape == target_state[name].shape
        ):
            load_state[name] = value
            reused_adapter_names.append(name)
        else:
            skipped_adapter_names.append(name)

    result = model.load_state_dict(load_state, strict=False)
    non_adapter_missing = [
        name for name in result.missing_keys if not is_backbone_adapter_key(name)
    ]
    if non_adapter_missing or result.unexpected_keys:
        raise ValueError(
            "transfer package does not match model: "
            f"missing={non_adapter_missing} unexpected={result.unexpected_keys}"
        )

    loaded_parameters = sum(target_state[name].numel() for name in load_state)
    total_parameters = sum(value.numel() for value in target_state.values())
    report = {
        "source_name": payload["source_adapter"]["name"],
        "source_audio_dim": int(payload["source_adapter"]["audio_dim"]),
        "source_text_dim": int(payload["source_adapter"]["text_dim"]),
        "target_audio_dim": int(audio_dim),
        "target_text_dim": int(text_dim),
        "loaded_tensors": len(load_state),
        "total_tensors": len(target_state),
        "loaded_parameters": loaded_parameters,
        "total_parameters": total_parameters,
        "loaded_parameter_ratio": loaded_parameters / max(total_parameters, 1),
        "reused_adapter_tensors": sorted(reused_adapter_names),
        "fresh_adapter_tensors": sorted(skipped_adapter_names),
        "missing_keys": sorted(result.missing_keys),
        "requires_target_training": True,
    }
    return model, report


def build_target_initialization_checkpoint(
    transfer_checkpoint: Mapping[str, Any] | str | Path,
    *,
    audio_dim: int,
    text_dim: int,
    target_name: str,
    reuse_shape_compatible_source_adapter: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a regular model-only checkpoint for target-base SFT startup."""

    model, report = load_transferable_checkpoint(
        transfer_checkpoint,
        audio_dim=audio_dim,
        text_dim=text_dim,
        reuse_shape_compatible_source_adapter=reuse_shape_compatible_source_adapter,
    )
    checkpoint = {
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_step": 0,
        "next_global_sample": 0,
        "note": (
            f"Initialization-only Attention Fusion checkpoint for {target_name}; "
            "target-base alignment/SFT is required before inference."
        ),
        "transfer_report": report,
        "target_name": target_name,
        "initialization_only": True,
    }
    return checkpoint, report
