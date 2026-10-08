"""Compatibility layer for Microsoft's WavLM-Large + ECAPA checkpoint.

The ECAPA blocks below preserve the module names and arithmetic used by the
official UniSpeech speaker-verification implementation so its checkpoint can
be loaded without an approximate head conversion.  WavLM parameters are
translated to Hugging Face's equivalent names by :func:`convert_official_checkpoint`.

Upstream implementation (CC BY-SA 3.0; see the vendored ``UPSTREAM_LICENSE``):
https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
"""

from __future__ import annotations

import re
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Res2Conv1dReluBn(nn.Module):
    """The Res2Net temporal block used by Microsoft's ECAPA implementation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if channels % scale:
            raise ValueError(f"channels ({channels}) must be divisible by scale ({scale})")
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=bias,
            )
            for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(self.nums))

    def forward(self, x: Tensor) -> Tensor:
        pieces = torch.split(x, self.width, dim=1)
        outputs: list[Tensor] = []
        current: Tensor | None = None
        for index in range(self.nums):
            current = pieces[index] if index == 0 else current + pieces[index]  # type: ignore[operator]
            current = self.bns[index](F.relu(self.convs[index](current)))
            outputs.append(current)
        if self.scale != 1:
            outputs.append(pieces[self.nums])
        return torch.cat(outputs, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """Keep the upstream spelling because it is part of the state-dict ABI."""

    def __init__(self, channels: int, se_bottleneck_dim: int = 128) -> None:
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean(dim=2)
        scale = F.relu(self.linear1(scale))
        scale = torch.sigmoid(self.linear2(scale))
        return x * scale.unsqueeze(2)


class SE_Res2Block(nn.Module):
    """SE-Res2 block with official attribute names for exact key loading."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
        se_bottleneck_dim: int,
    ) -> None:
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            scale=scale,
        )
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x) if self.shortcut is not None else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(
        self,
        in_dim: int,
        attention_channels: int = 128,
        global_context_att: bool = False,
    ) -> None:
        super().__init__()
        self.global_context_att = global_context_att
        attention_input_dim = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(attention_input_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if self.global_context_att:
            context_mean = x.mean(dim=-1, keepdim=True).expand_as(x)
            # Deliberately retain torch.var's upstream default (unbiased=True).
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            attention_input = torch.cat((x, context_mean, context_std), dim=1)
        else:
            attention_input = x
        alpha = torch.tanh(self.linear1(attention_input))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x.square(), dim=2) - mean.square()
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat((mean, std), dim=1)


class UniSpeechECAPAHead(nn.Module):
    """The exact inference head of ``ECAPA_TDNN_SMALL`` (without S3PRL)."""

    def __init__(
        self,
        feat_dim: int = 1024,
        channels: int = 512,
        embedding_dim: int = 256,
        *,
        scale: int = 8,
        se_bottleneck_dim: int = 128,
        aggregation_channels: int = 1536,
        attention_channels: int = 128,
        global_context_att: bool = False,
    ) -> None:
        super().__init__()
        if channels % scale:
            raise ValueError("channels must be divisible by scale")
        self.feat_dim = feat_dim
        self.embedding_dim = embedding_dim
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels, channels, 3, 1, 2, 2, scale, se_bottleneck_dim
        )
        self.layer3 = SE_Res2Block(
            channels, channels, 3, 1, 3, 3, scale, se_bottleneck_dim
        )
        self.layer4 = SE_Res2Block(
            channels, channels, 3, 1, 4, 4, scale, se_bottleneck_dim
        )
        self.conv = nn.Conv1d(channels * 3, aggregation_channels, kernel_size=1)
        self.pooling = AttentiveStatsPool(
            aggregation_channels,
            attention_channels=attention_channels,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(aggregation_channels * 2)
        self.linear = nn.Linear(aggregation_channels * 2, embedding_dim)

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 3 or frames.shape[-1] != self.feat_dim:
            raise ValueError(
                f"expected frames [batch, time, {self.feat_dim}], got {tuple(frames.shape)}"
            )
        # +1e-6 and this exact operation order are inherited from the official code.
        x = self.instance_norm(frames.transpose(1, 2) + 1e-6)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat((out2, out3, out4), dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        return self.linear(out)


_CONV_LAYER_RE = re.compile(r"^feature_extractor\.conv_layers\.(\d+)\.(0|2\.1)\.(.+)$")
_ENCODER_LAYER_RE = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")
_HEAD_ROOTS = frozenset({"layer1", "layer2", "layer3", "layer4", "conv", "pooling", "bn", "linear"})
_CLASSIFIER_KEY = "loss_calculator.projection.weight"


def validate_official_wavlm_large_config(config: Any) -> None:
    """Reject shape-compatible configs whose WavLM semantics differ upstream."""

    expected = {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "conv_dim": (512, 512, 512, 512, 512, 512, 512),
        "conv_kernel": (10, 3, 3, 3, 3, 2, 2),
        "conv_stride": (5, 2, 2, 2, 2, 2, 2),
        "conv_bias": False,
        "feat_extract_norm": "layer",
        "do_stable_layer_norm": True,
        "num_conv_pos_embeddings": 128,
        "num_conv_pos_embedding_groups": 16,
        "num_buckets": 320,
        "max_bucket_distance": 800,
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    for name, wanted in expected.items():
        actual = getattr(config, name, None)
        comparable = tuple(actual) if isinstance(wanted, tuple) and actual is not None else actual
        if comparable != wanted:
            mismatches[name] = (actual, wanted)
    if mismatches:
        raise ValueError(f"config is not the audited Microsoft WavLM-Large architecture: {mismatches}")


def map_official_state_key(
    source_key: str,
    target_state_keys: Collection[str] | None = None,
) -> str | None:
    """Map one official key to this wrapper; return ``None`` for the AAM classifier.

    ``target_state_keys`` selects the positional-convolution weight-normalization
    spelling used by the installed PyTorch/Transformers version.  Modern PyTorch
    uses ``parametrizations.weight.original{0,1}``; old versions use
    ``weight_{g,v}``.
    """

    if source_key == _CLASSIFIER_KEY:
        return None
    if source_key == "feature_weight":
        return source_key
    root = source_key.split(".", 1)[0]
    if root in _HEAD_ROOTS:
        return f"ecapa.{source_key}"
    prefix = "feature_extract.model."
    if not source_key.startswith(prefix):
        raise KeyError(f"unsupported official checkpoint key: {source_key}")
    key = source_key[len(prefix) :]

    if key == "mask_emb":
        mapped = "masked_spec_embed"
    elif (match := _CONV_LAYER_RE.fullmatch(key)) is not None:
        index, component, tail = match.groups()
        component_name = "conv" if component == "0" else "layer_norm"
        mapped = f"feature_extractor.conv_layers.{index}.{component_name}.{tail}"
    elif key.startswith("post_extract_proj."):
        mapped = key.replace("post_extract_proj.", "feature_projection.projection.", 1)
    elif key.startswith("layer_norm."):
        mapped = key.replace("layer_norm.", "feature_projection.layer_norm.", 1)
    elif key == "encoder.pos_conv.0.bias":
        mapped = "encoder.pos_conv_embed.conv.bias"
    elif key in {"encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v"}:
        suffix = key.rsplit(".", 1)[-1]
        legacy = f"wavlm.encoder.pos_conv_embed.conv.{suffix}"
        modern_index = "0" if suffix == "weight_g" else "1"
        modern = (
            "wavlm.encoder.pos_conv_embed.conv.parametrizations.weight."
            f"original{modern_index}"
        )
        if target_state_keys is not None and legacy in target_state_keys:
            return legacy
        if target_state_keys is not None and modern not in target_state_keys:
            raise KeyError(
                f"target state has neither modern nor legacy destination for {source_key}"
            )
        return modern
    elif key.startswith("encoder.layer_norm."):
        mapped = key
    elif (match := _ENCODER_LAYER_RE.fullmatch(key)) is not None:
        index, tail = match.groups()
        if tail.startswith("self_attn.relative_attention_bias."):
            tail = tail.replace(
                "self_attn.relative_attention_bias.", "attention.rel_attn_embed.", 1
            )
        elif tail == "self_attn.grep_a":
            tail = "attention.gru_rel_pos_const"
        elif tail.startswith("self_attn.grep_linear."):
            tail = tail.replace("self_attn.grep_linear.", "attention.gru_rel_pos_linear.", 1)
        elif re.match(r"^self_attn\.(k_proj|v_proj|q_proj|out_proj)\.", tail):
            tail = tail.replace("self_attn.", "attention.", 1)
        elif tail.startswith("self_attn_layer_norm."):
            tail = tail.replace("self_attn_layer_norm.", "layer_norm.", 1)
        elif tail.startswith("fc1."):
            tail = tail.replace("fc1.", "feed_forward.intermediate_dense.", 1)
        elif tail.startswith("fc2."):
            tail = tail.replace("fc2.", "feed_forward.output_dense.", 1)
        elif not tail.startswith("final_layer_norm."):
            raise KeyError(f"unsupported WavLM encoder-layer key: {source_key}")
        mapped = f"encoder.layers.{index}.{tail}"
    else:
        raise KeyError(f"unsupported WavLM key: {source_key}")

    target_key = f"wavlm.{mapped}"
    if target_state_keys is not None and target_key not in target_state_keys:
        raise KeyError(f"mapped destination is absent from target state: {target_key}")
    return target_key


@dataclass(frozen=True)
class ConvertedUniSpeechCheckpoint:
    """The inference encoder plus training-only metadata split from the source."""

    encoder_state: OrderedDict[str, Tensor]
    source_classifier_weight: Tensor
    best_valid_eer: float | None


def convert_official_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    target_state: Mapping[str, Tensor] | None = None,
) -> ConvertedUniSpeechCheckpoint:
    """Convert Microsoft's 711-key checkpoint into 710 encoder + 1 classifier keys.

    When ``target_state`` is supplied, this performs strict key and shape checks
    before returning, preventing a silently partial ``strict=False`` load.
    """

    raw_state = checkpoint.get("model")
    if not isinstance(raw_state, Mapping):
        raise ValueError("official checkpoint must contain a mapping at key 'model'")
    target_keys = set(target_state) if target_state is not None else None
    converted: OrderedDict[str, Tensor] = OrderedDict()
    classifier: Tensor | None = None
    for source_key, value in raw_state.items():
        if not isinstance(source_key, str) or not isinstance(value, Tensor):
            raise TypeError("official model state must map string keys to tensors")
        target_key = map_official_state_key(source_key, target_keys)
        if target_key is None:
            if classifier is not None:
                raise ValueError("duplicate source classifier weight")
            classifier = value
            continue
        if target_key in converted:
            raise ValueError(f"multiple source keys map to {target_key}")
        converted[target_key] = value

    if classifier is None:
        raise ValueError(f"official checkpoint is missing {_CLASSIFIER_KEY}")
    if "feature_weight" not in converted:
        raise ValueError("official checkpoint is missing feature_weight")

    if target_state is not None:
        missing = set(target_state) - set(converted)
        unexpected = set(converted) - set(target_state)
        if missing or unexpected:
            raise ValueError(
                "converted state does not exactly match target; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        bad_shapes = {
            key: (tuple(converted[key].shape), tuple(target_state[key].shape))
            for key in target_state
            if converted[key].shape != target_state[key].shape
        }
        if bad_shapes:
            raise ValueError(f"converted tensor shapes do not match target: {bad_shapes}")

    best_valid_eer = checkpoint.get("best_valid_eer")
    if best_valid_eer is not None:
        best_valid_eer = float(best_valid_eer)
    return ConvertedUniSpeechCheckpoint(converted, classifier, best_valid_eer)


class MicrosoftUniSpeechWavLMECAPA(nn.Module):
    """Hugging Face WavLM with the official Microsoft ECAPA speaker head."""

    def __init__(
        self,
        wavlm: nn.Module,
        *,
        embedding_dim: int = 256,
        channels: int = 512,
        normalize_waveforms: bool = False,
        normalize_embeddings: bool = False,
        last_n_layers: int = 0,
        ecapa_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(wavlm, "config"):
            raise TypeError("wavlm must expose a Hugging Face-style config")
        config = wavlm.config
        if not hasattr(config, "hidden_size") or not hasattr(config, "num_hidden_layers"):
            raise TypeError("wavlm.config must define hidden_size and num_hidden_layers")
        self.wavlm = wavlm
        if hasattr(config, "apply_spec_augment"):
            config.apply_spec_augment = False
        if hasattr(config, "layerdrop"):
            config.layerdrop = 0.0
        # In the audited upstream config, dropout_input/dropout_features are 0.
        # Hugging Face names the corresponding projection dropout
        # ``feat_proj_dropout`` and materializes it inside feature_projection.
        if hasattr(config, "feat_proj_dropout"):
            config.feat_proj_dropout = 0.0
        projection_dropout = getattr(getattr(wavlm, "feature_projection", None), "dropout", None)
        if isinstance(projection_dropout, nn.Dropout):
            projection_dropout.p = 0.0
        self.feature_weight = nn.Parameter(torch.zeros(config.num_hidden_layers + 1))
        self.ecapa = UniSpeechECAPAHead(
            feat_dim=config.hidden_size,
            channels=channels,
            embedding_dim=embedding_dim,
            **dict(ecapa_kwargs or {}),
        )
        self.embedding_dim = embedding_dim
        # The pinned S3PRL expert conditionally applies layer_norm only when the
        # upstream checkpoint's cfg.normalize is true.  Microsoft's released
        # WavLM-Large speaker checkpoint has normalize=false, hence this default.
        self.normalize_waveforms = normalize_waveforms
        self.normalize_embeddings = normalize_embeddings
        self.configure_trainable(last_n_layers)

    @classmethod
    def from_official_checkpoint(
        cls,
        checkpoint_path: str | Path,
        wavlm_config_name_or_path: str | Path,
        **wrapper_kwargs: Any,
    ) -> tuple["MicrosoftUniSpeechWavLMECAPA", ConvertedUniSpeechCheckpoint]:
        """Construct and strictly load the official checkpoint on CPU."""

        try:
            from transformers import WavLMConfig, WavLMModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers is required for WavLM conversion") from exc
        config = WavLMConfig.from_pretrained(str(wavlm_config_name_or_path), local_files_only=True)
        validate_official_wavlm_large_config(config)
        # Construct on meta and assign checkpoint tensors into the module.  This
        # avoids temporarily holding initialized WavLM plus the 1.3 GB source.
        with torch.device("meta"):
            model = cls(WavLMModel(config), **wrapper_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        converted = convert_official_checkpoint(checkpoint, target_state=model.state_dict())
        model.load_state_dict(converted.encoder_state, strict=True, assign=True)
        return model, converted

    def _transformer_layers(self) -> Sequence[nn.Module]:
        try:
            return self.wavlm.encoder.layers
        except AttributeError as exc:
            raise TypeError("wavlm must expose encoder.layers") from exc

    @property
    def backbone_fully_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.wavlm.parameters())

    def configure_trainable(self, last_n_layers: int) -> None:
        layers = self._transformer_layers()
        if not 0 <= last_n_layers <= len(layers):
            raise ValueError(f"last_n_layers must be between 0 and {len(layers)}")
        self.wavlm.requires_grad_(False)
        if last_n_layers:
            for layer in layers[-last_n_layers:]:
                layer.requires_grad_(True)
        if self.backbone_fully_frozen:
            self.wavlm.eval()

    def train(self, mode: bool = True) -> "MicrosoftUniSpeechWavLMECAPA":
        super().train(mode)
        if self.backbone_fully_frozen:
            self.wavlm.eval()
        elif mode:
            # Frozen feature layers must not change via dropout/layer dropping.
            # Explicit waveform augmentation is kept identical across ablations.
            self.wavlm.config.layerdrop = 0.0
            self.wavlm.config.mask_time_prob = 0.0
            self.wavlm.config.mask_feature_prob = 0.0
            self.wavlm.feature_extractor.eval()
            self.wavlm.feature_projection.eval()
            self.wavlm.encoder.pos_conv_embed.eval()
            if hasattr(self.wavlm.encoder, 'layer_norm'):
                self.wavlm.encoder.layer_norm.eval()
            for layer in self._transformer_layers():
                if not any(p.requires_grad for p in layer.parameters()):
                    layer.eval()
        return self

    def _encode_equal_length(self, waveforms: Tensor) -> Tensor:
        if self.normalize_waveforms:
            waveforms = F.layer_norm(waveforms, (waveforms.shape[-1],))
        context = torch.no_grad() if self.backbone_fully_frozen else nullcontext()
        with context:
            output = self.wavlm(
                waveforms,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != self.feature_weight.numel():
            count = 0 if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"expected {self.feature_weight.numel()} WavLM hidden states, got {count}"
            )
        stacked = torch.stack(tuple(hidden_states), dim=0)
        weights = self.feature_weight.softmax(dim=0).to(dtype=stacked.dtype)
        frames = torch.sum(stacked * weights[:, None, None, None], dim=0)
        embeddings = self.ecapa(frames)
        return F.normalize(embeddings, dim=-1) if self.normalize_embeddings else embeddings

    def forward(self, waveforms: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if waveforms.ndim != 2:
            raise ValueError(f"expected waveforms [batch, samples], got {tuple(waveforms.shape)}")
        batch, samples = waveforms.shape
        if batch == 0 or samples == 0:
            raise ValueError("waveform batch and sample dimensions must be non-empty")
        if attention_mask is None:
            lengths = torch.full((batch,), samples, device=waveforms.device, dtype=torch.long)
        else:
            if attention_mask.shape != waveforms.shape:
                raise ValueError("attention_mask must have the same shape as waveforms")
            valid = attention_mask.to(device=waveforms.device, dtype=torch.bool)
            lengths = valid.sum(dim=1)
            if bool((lengths == 0).any()):
                raise ValueError("each waveform must contain at least one valid sample")
            expected = torch.arange(samples, device=waveforms.device)[None, :] < lengths[:, None]
            if not torch.equal(valid, expected):
                raise ValueError("attention_mask must be left-aligned (valid samples then padding)")

        unique_lengths = torch.unique(lengths, sorted=True)
        if unique_lengths.numel() == 1:
            length = int(unique_lengths[0].item())
            return self._encode_equal_length(waveforms[:, :length])

        pieces: list[Tensor] = []
        indices: list[Tensor] = []
        for length_tensor in unique_lengths:
            length = int(length_tensor.item())
            group_indices = torch.nonzero(lengths == length_tensor, as_tuple=False).squeeze(1)
            pieces.append(self._encode_equal_length(waveforms.index_select(0, group_indices)[:, :length]))
            indices.append(group_indices)
        concatenated = torch.cat(pieces, dim=0)
        concatenated_indices = torch.cat(indices, dim=0)
        return concatenated.index_select(0, torch.argsort(concatenated_indices))


__all__ = [
    "AttentiveStatsPool",
    "ConvertedUniSpeechCheckpoint",
    "MicrosoftUniSpeechWavLMECAPA",
    "UniSpeechECAPAHead",
    "convert_official_checkpoint",
    "map_official_state_key",
    "validate_official_wavlm_large_config",
]
