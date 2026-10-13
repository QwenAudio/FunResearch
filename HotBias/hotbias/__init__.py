"""HotBias fusion core and cross-backbone alignment utilities."""

from .attention_fusion import (
    AttentionFusionConfig,
    FrozenFeatureAttentionFusion,
    trainable_parameter_summary,
)
from .binary_attention_fusion import (
    candidate_input_scalar_features,
    explicit_relation_scalar_features,
)
from .sparse_logits import coalesce_candidate_bias, sparse_biased_nll
from .transfer import (
    BACKBONE_ADAPTER_PREFIXES,
    build_target_initialization_checkpoint,
    build_transferable_checkpoint,
    load_transferable_checkpoint,
)

__all__ = [
    "AttentionFusionConfig",
    "BACKBONE_ADAPTER_PREFIXES",
    "FrozenFeatureAttentionFusion",
    "build_target_initialization_checkpoint",
    "build_transferable_checkpoint",
    "candidate_input_scalar_features",
    "coalesce_candidate_bias",
    "explicit_relation_scalar_features",
    "load_transferable_checkpoint",
    "sparse_biased_nll",
    "trainable_parameter_summary",
]
