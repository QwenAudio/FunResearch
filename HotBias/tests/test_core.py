"""Small, backbone-independent checks for the released HotBias core."""

from dataclasses import asdict

import torch

from hotbias import AttentionFusionConfig, FrozenFeatureAttentionFusion
from hotbias.transfer import (
    build_target_initialization_checkpoint,
    build_transferable_checkpoint,
    is_backbone_adapter_key,
    load_transferable_checkpoint,
)
from hotbias.transfer_alignment import set_trainable_scope


def tiny_checkpoint() -> dict:
    config = AttentionFusionConfig(
        audio_dim=8,
        text_dim=8,
        hidden_dim=16,
        num_heads=4,
        ffn_dim=32,
        audio_cross_layers=1,
        text_cross_layers=1,
        candidate_layers=1,
        dropout=0.0,
    )
    model = FrozenFeatureAttentionFusion(config)
    return {"config": asdict(config), "model_state": model.state_dict()}


def test_transfer_preserves_source_weights() -> None:
    source = tiny_checkpoint()
    package = build_transferable_checkpoint(source, source_name="test source")
    restored, report = load_transferable_checkpoint(package, audio_dim=8, text_dim=8)
    assert report["loaded_tensors"] == report["total_tensors"]
    for name, value in source["model_state"].items():
        assert torch.equal(restored.state_dict()[name], value)


def test_new_backbone_trains_only_six_projection_modules() -> None:
    package = build_transferable_checkpoint(tiny_checkpoint(), source_name="test source")
    initialized, report = build_target_initialization_checkpoint(
        package, audio_dim=4, text_dim=4, target_name="test target"
    )
    assert initialized["initialization_only"] is True
    assert report["requires_target_training"] is True
    model = FrozenFeatureAttentionFusion(
        AttentionFusionConfig(**initialized["config"])
    )
    model.load_state_dict(initialized["model_state"], strict=True)
    set_trainable_scope(model, "backbone_adapters")
    assert all(
        parameter.requires_grad == is_backbone_adapter_key(name)
        for name, parameter in model.named_parameters()
    )
