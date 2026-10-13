from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AttentionFusionConfig:
    """Configuration for the frozen-base Attention hotword branch."""

    audio_dim: int = 2048
    text_dim: int = 2048
    hidden_dim: int = 256
    num_heads: int = 4
    ffn_dim: int = 1024
    audio_cross_layers: int = 2
    text_cross_layers: int = 2
    candidate_layers: int = 4
    scalar_dim: int = 6
    dropout: float = 0.1
    layer_scale_init: float = 0.1
    max_gain: float = 1.5
    max_bias: float = 10.0
    logit_margin: float = 0.3


class RMSNorm(nn.Module):
    """RMSNorm compatible with the older PyTorch used by the ASR runtime."""

    def __init__(self, hidden_dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(inputs.dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(hidden_dim, ffn_dim * 2)
        self.down = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(inputs).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention plus SwiGLU with independent layer scales."""

    def __init__(self, config: AttentionFusionConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.query_norm = RMSNorm(hidden)
        self.memory_norm = RMSNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn_norm = RMSNorm(hidden)
        self.ffn = SwiGLU(hidden, config.ffn_dim, config.dropout)
        self.attention_scale = nn.Parameter(
            torch.full((hidden,), config.layer_scale_init)
        )
        self.ffn_scale = nn.Parameter(torch.full((hidden,), config.layer_scale_init))

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        query = query + self.attention_scale * attended
        return query + self.ffn_scale * self.ffn(self.ffn_norm(query))


class CandidateSelfAttentionBlock(nn.Module):
    """Permutation-equivariant relation block over the candidate set."""

    def __init__(self, config: AttentionFusionConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.attention_norm = RMSNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn_norm = RMSNorm(hidden)
        self.ffn = SwiGLU(hidden, config.ffn_dim, config.dropout)
        self.attention_scale = nn.Parameter(
            torch.full((hidden,), config.layer_scale_init)
        )
        self.ffn_scale = nn.Parameter(torch.full((hidden,), config.layer_scale_init))

    def forward(
        self,
        candidates: torch.Tensor,
        *,
        candidate_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(candidates)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )
        candidates = candidates + self.attention_scale * attended
        return candidates + self.ffn_scale * self.ffn(self.ffn_norm(candidates))


class FrozenFeatureAttentionFusion(nn.Module):
    """Hotword-conditioned audio/text evidence model with dynamic gap bias."""

    scalar_names = (
        "base_probability",
        "base_logit_gap",
        "base_rank_normalized",
        "progress",
        "is_start",
        "is_end",
    )

    def __init__(self, config: AttentionFusionConfig | None = None):
        super().__init__()
        self.config = config or AttentionFusionConfig()
        config = self.config
        hidden = config.hidden_dim

        self.audio_projection = nn.Sequential(
            nn.Linear(config.audio_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.history_projection = nn.Sequential(
            nn.Linear(config.text_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.decoder_projection = nn.Sequential(
            nn.Linear(config.text_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.hotword_projection = nn.Sequential(
            nn.Linear(config.text_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.current_token_projection = nn.Sequential(
            nn.Linear(config.text_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.prefix_projection = nn.Sequential(
            nn.Linear(config.text_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(config.scalar_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.query_builder = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            RMSNorm(hidden),
        )

        self.history_segment_embedding = nn.Embedding(2, hidden)
        self.audio_null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.text_null = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.audio_null, std=hidden**-0.5)
        nn.init.normal_(self.text_null, std=hidden**-0.5)

        self.audio_cross_attention = nn.ModuleList(
            CrossAttentionBlock(config) for _ in range(config.audio_cross_layers)
        )
        self.text_cross_attention = nn.ModuleList(
            CrossAttentionBlock(config) for _ in range(config.text_cross_layers)
        )
        self.text_mix = nn.Linear(hidden * 3, hidden)
        self.fusion_norm = RMSNorm(hidden)
        self.fusion_ffn_norm = RMSNorm(hidden)
        self.fusion_ffn = SwiGLU(hidden, config.ffn_dim, config.dropout)
        self.fusion_scale = nn.Parameter(
            torch.full((hidden,), config.layer_scale_init)
        )

        self.candidate_attention = nn.ModuleList(
            CandidateSelfAttentionBlock(config)
            for _ in range(config.candidate_layers)
        )
        self.direct_relation_merge = nn.Sequential(
            nn.Linear(hidden * 2, hidden), RMSNorm(hidden), nn.SiLU()
        )
        self.bias_head = nn.Sequential(
            RMSNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        final = self.bias_head[-1]
        nn.init.normal_(final.weight, std=1e-3)
        nn.init.constant_(final.bias, -5.0)

    @staticmethod
    def _append_null(
        memory: torch.Tensor,
        null: torch.Tensor,
        padding_mask: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        memory = torch.cat((null.expand(memory.shape[0], -1, -1), memory), dim=1)
        if padding_mask is not None:
            null_padding = torch.zeros(
                padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device
            )
            padding_mask = torch.cat((null_padding, padding_mask.bool()), dim=1)
        if attention_mask is not None:
            null_attention = torch.zeros(
                attention_mask.shape[0], 1, dtype=torch.bool, device=attention_mask.device
            )
            attention_mask = torch.cat(
                (null_attention, attention_mask.bool()), dim=1
            )
        return memory, padding_mask, attention_mask

    def _forward_sample(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        decoder_states = sample["decoder_states"]
        hotword_means = sample["hotword_means"]
        current_token_embeddings = sample["current_token_embeddings"]
        prefix_embeddings = sample["prefix_embeddings"]
        scalar_features = sample["scalar_features"]
        steps, candidates = current_token_embeddings.shape[:2]

        decoder = self.decoder_projection(decoder_states)[:, None, :].expand(
            -1, candidates, -1
        )
        projected_word = sample.get("projected_hotword_means")
        if projected_word is None:
            projected_word = self.hotword_projection(hotword_means)
        word = projected_word[None, :, :].expand(steps, -1, -1)
        current = sample.get("projected_current_token_embeddings")
        if current is None:
            current = self.current_token_projection(current_token_embeddings)
        prefix = sample.get("projected_prefix_embeddings")
        if prefix is None:
            prefix = self.prefix_projection(prefix_embeddings)
        scalars = self.scalar_projection(scalar_features)
        query = self.query_builder(
            torch.cat((decoder, word, current, prefix, scalars), dim=-1)
        )
        flat_query = query.reshape(1, steps * candidates, -1)

        projected_audio = sample.get("projected_audio_features")
        if projected_audio is None:
            projected_audio = self.audio_projection(sample["audio_features"])
        audio = projected_audio[None, :, :]
        audio_padding = sample.get("audio_padding_mask")
        if audio_padding is not None:
            audio_padding = audio_padding[None, :]
        audio, audio_padding, _ = self._append_null(
            audio, self.audio_null, audio_padding
        )
        audio_state = flat_query
        for block in self.audio_cross_attention:
            audio_state = block(audio_state, audio, memory_padding_mask=audio_padding)

        history = self.history_projection(sample["history_features"])
        history = history + self.history_segment_embedding(
            sample["history_segment_ids"]
        )
        history = history[None, :, :]
        history_padding = sample.get("history_padding_mask")
        if history_padding is not None:
            history_padding = history_padding[None, :]
        text_attention_mask = sample["history_attention_mask"].repeat_interleave(
            candidates, dim=0
        )
        history, history_padding, text_attention_mask = self._append_null(
            history, self.text_null, history_padding, text_attention_mask
        )
        text_state = flat_query
        for block in self.text_cross_attention:
            text_state = block(
                text_state,
                history,
                memory_padding_mask=history_padding,
                attention_mask=text_attention_mask,
            )

        text_weight = torch.sigmoid(
            self.text_mix(torch.cat((flat_query, audio_state, text_state), dim=-1))
        )
        independent = self.fusion_norm(
            flat_query + audio_state + text_weight * text_state
        )
        independent = independent + self.fusion_scale * self.fusion_ffn(
            self.fusion_ffn_norm(independent)
        )
        independent = independent.reshape(steps, candidates, -1)

        related = independent
        candidate_padding = sample.get("candidate_padding_mask")
        if candidate_padding is not None:
            candidate_padding = candidate_padding.expand(steps, -1)
        for block in self.candidate_attention:
            related = block(related, candidate_padding_mask=candidate_padding)
        final_state = self.direct_relation_merge(
            torch.cat((independent, related), dim=-1)
        )

        raw_score = self.bias_head(final_state).squeeze(-1)
        gain = self.config.max_gain * torch.sigmoid(raw_score)
        base_logits = sample.get("base_logits")
        candidate_token_ids = sample["candidate_token_ids"]
        if "base_gap" in sample:
            base_gap = sample["base_gap"].detach().float()
        else:
            if base_logits is None:
                raise ValueError("base_logits or base_gap is required")
            candidate_logits = base_logits.gather(1, candidate_token_ids)
            top1_logits = base_logits.max(dim=-1, keepdim=True).values
            base_gap = (top1_logits - candidate_logits).clamp_min(0.0).detach()
        required_bias = base_gap + self.config.logit_margin
        candidate_bias = (gain.float() * required_bias.float()).clamp(
            min=0.0, max=self.config.max_bias
        )
        if candidate_padding is not None:
            candidate_bias = candidate_bias.masked_fill(candidate_padding, 0.0)

        output = {
            "candidate_bias": candidate_bias,
            "gain": gain,
            "raw_score": raw_score,
            "base_gap": base_gap,
            "required_bias": required_bias,
            "independent_state": independent,
            "related_state": related,
        }
        if sample.get("return_full_logits", True):
            if base_logits is None:
                raise ValueError("full vocabulary output requires base_logits")
            delta_logits = torch.zeros_like(base_logits)
            delta_logits.scatter_reduce_(
                1,
                candidate_token_ids,
                candidate_bias.to(base_logits.dtype),
                reduce="amax",
                include_self=True,
            )
            output["delta_logits"] = delta_logits
            output["final_logits"] = base_logits + delta_logits
        return output

    def forward_online_batch(
        self,
        *,
        decoder_states: torch.Tensor,
        hotword_means: torch.Tensor,
        current_token_embeddings: torch.Tensor,
        prefix_embeddings: torch.Tensor,
        scalar_features: torch.Tensor,
        history_features: torch.Tensor,
        history_segment_ids: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        base_logits: torch.Tensor,
        audio_features: torch.Tensor | None = None,
        projected_audio_features: torch.Tensor | None = None,
        audio_padding_mask: torch.Tensor | None = None,
        history_padding_mask: torch.Tensor | None = None,
        candidate_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Vectorized one-step inference over live rollouts of one utterance."""

        batch, candidates = current_token_embeddings.shape[:2]
        decoder = self.decoder_projection(decoder_states)[:, None, :].expand(
            -1, candidates, -1
        )
        word = self.hotword_projection(hotword_means)
        current = self.current_token_projection(current_token_embeddings)
        prefix = self.prefix_projection(prefix_embeddings)
        scalars = self.scalar_projection(scalar_features)
        query = self.query_builder(
            torch.cat((decoder, word, current, prefix, scalars), dim=-1)
        )

        if projected_audio_features is None:
            if audio_features is None:
                raise ValueError("audio features are required")
            projected_audio_features = self.audio_projection(audio_features)
        if projected_audio_features.ndim == 2:
            audio = projected_audio_features[None].expand(batch, -1, -1)
        else:
            audio = projected_audio_features.expand(batch, -1, -1)
        audio, audio_padding_mask, _ = self._append_null(
            audio, self.audio_null, audio_padding_mask
        )
        audio_state = query
        for block in self.audio_cross_attention:
            audio_state = block(
                audio_state, audio, memory_padding_mask=audio_padding_mask
            )

        history = self.history_projection(history_features)
        history = history + self.history_segment_embedding(history_segment_ids)
        history, history_padding_mask, _ = self._append_null(
            history, self.text_null, history_padding_mask
        )
        text_state = query
        for block in self.text_cross_attention:
            text_state = block(
                text_state, history, memory_padding_mask=history_padding_mask
            )

        text_weight = torch.sigmoid(
            self.text_mix(torch.cat((query, audio_state, text_state), dim=-1))
        )
        independent = self.fusion_norm(query + audio_state + text_weight * text_state)
        independent = independent + self.fusion_scale * self.fusion_ffn(
            self.fusion_ffn_norm(independent)
        )
        related = independent
        for block in self.candidate_attention:
            related = block(
                related, candidate_padding_mask=candidate_padding_mask
            )
        final_state = self.direct_relation_merge(
            torch.cat((independent, related), dim=-1)
        )
        raw_score = self.bias_head(final_state).squeeze(-1)
        gain = self.config.max_gain * torch.sigmoid(raw_score)
        candidate_logits = base_logits.gather(1, candidate_token_ids)
        top1_logits = base_logits.max(dim=-1, keepdim=True).values
        base_gap = (top1_logits - candidate_logits).clamp_min(0.0).detach()
        required_bias = base_gap + self.config.logit_margin
        candidate_bias = (gain.float() * required_bias.float()).clamp(
            min=0.0, max=self.config.max_bias
        )
        if candidate_padding_mask is not None:
            candidate_bias = candidate_bias.masked_fill(
                candidate_padding_mask, 0.0
            )
        return {
            "candidate_bias": candidate_bias,
            "gain": gain,
            "raw_score": raw_score,
            "base_gap": base_gap,
            "required_bias": required_bias,
            "independent_state": independent,
            "related_state": related,
        }

    def _forward_equal_length_batch(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """Batch full rollout sequences that share one utterance and length."""

        batch = len(samples)
        steps, candidates = samples[0]["current_token_embeddings"].shape[:2]
        decoder_states = torch.stack([sample["decoder_states"] for sample in samples])
        hotword_means = torch.stack([sample["hotword_means"] for sample in samples])
        current_tokens = torch.stack(
            [sample["current_token_embeddings"] for sample in samples]
        )
        prefixes = torch.stack([sample["prefix_embeddings"] for sample in samples])
        scalars = torch.stack([sample["scalar_features"] for sample in samples])

        decoder = self.decoder_projection(decoder_states)[:, :, None, :].expand(
            -1, -1, candidates, -1
        )
        word = self.hotword_projection(hotword_means)[:, None, :, :].expand(
            -1, steps, -1, -1
        )
        current = self.current_token_projection(current_tokens)
        prefix = self.prefix_projection(prefixes)
        scalar_state = self.scalar_projection(scalars)
        query = self.query_builder(
            torch.cat((decoder, word, current, prefix, scalar_state), dim=-1)
        )
        flat_query = query.reshape(batch, steps * candidates, -1)

        # All rows in a GRPO group share the same frozen audio features.  One
        # projection is therefore sufficient and preserves gradients into the
        # projection weights by summing through the expanded view.
        audio = self.audio_projection(samples[0]["audio_features"])[None].expand(
            batch, -1, -1
        )
        audio, _, _ = self._append_null(audio, self.audio_null, None)
        audio_state = flat_query
        for block in self.audio_cross_attention:
            audio_state = block(audio_state, audio)

        history = self.history_projection(
            torch.stack([sample["history_features"] for sample in samples])
        )
        history_segments = torch.stack(
            [sample["history_segment_ids"] for sample in samples]
        )
        history = history + self.history_segment_embedding(history_segments)
        text_masks = torch.stack(
            [sample["history_attention_mask"] for sample in samples]
        ).repeat_interleave(candidates, dim=1)
        text_masks = text_masks.repeat_interleave(self.config.num_heads, dim=0)
        history, _, _ = self._append_null(history, self.text_null, None)
        text_masks = torch.cat(
            (
                torch.zeros(
                    text_masks.shape[0],
                    text_masks.shape[1],
                    1,
                    dtype=torch.bool,
                    device=text_masks.device,
                ),
                text_masks.bool(),
            ),
            dim=2,
        )
        text_state = flat_query
        for block in self.text_cross_attention:
            text_state = block(text_state, history, attention_mask=text_masks)

        text_weight = torch.sigmoid(
            self.text_mix(torch.cat((flat_query, audio_state, text_state), dim=-1))
        )
        independent = self.fusion_norm(
            flat_query + audio_state + text_weight * text_state
        )
        independent = independent + self.fusion_scale * self.fusion_ffn(
            self.fusion_ffn_norm(independent)
        )
        independent = independent.reshape(batch, steps, candidates, -1)

        related = independent.reshape(batch * steps, candidates, -1)
        for block in self.candidate_attention:
            related = block(related)
        related = related.reshape(batch, steps, candidates, -1)
        final_state = self.direct_relation_merge(
            torch.cat((independent, related), dim=-1)
        )
        raw_score = self.bias_head(final_state).squeeze(-1)
        gain = self.config.max_gain * torch.sigmoid(raw_score)
        base_logits = torch.stack([sample["base_logits"] for sample in samples])
        candidate_ids = torch.stack(
            [sample["candidate_token_ids"] for sample in samples]
        )
        candidate_logits = base_logits.gather(2, candidate_ids)
        top1_logits = base_logits.max(dim=-1, keepdim=True).values
        base_gap = (top1_logits - candidate_logits).clamp_min(0.0).detach()
        required_bias = base_gap + self.config.logit_margin
        candidate_bias = (gain.float() * required_bias.float()).clamp(
            min=0.0, max=self.config.max_bias
        )
        delta_logits = torch.zeros_like(base_logits)
        delta_logits.scatter_reduce_(
            2,
            candidate_ids,
            candidate_bias.to(base_logits.dtype),
            reduce="amax",
            include_self=True,
        )
        output = {
            "candidate_bias": candidate_bias,
            "delta_logits": delta_logits,
            "final_logits": base_logits + delta_logits,
            "gain": gain,
            "raw_score": raw_score,
            "base_gap": base_gap,
            "required_bias": required_bias,
            "independent_state": independent,
            "related_state": related,
        }
        return [
            {key: value[index] for key, value in output.items()}
            for index in range(batch)
        ]

    def _forward_padded_batch(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """Batch unrelated variable-length samples without changing semantics."""

        batch = len(samples)
        step_lengths = [sample["current_token_embeddings"].shape[0] for sample in samples]
        candidate_lengths = [
            sample["current_token_embeddings"].shape[1] for sample in samples
        ]
        audio_lengths = [sample["audio_features"].shape[0] for sample in samples]
        history_lengths = [sample["history_features"].shape[0] for sample in samples]
        max_steps = max(step_lengths)
        max_candidates = max(candidate_lengths)
        max_audio = max(audio_lengths)
        max_history = max(history_lengths)
        device = samples[0]["decoder_states"].device

        decoder_states = samples[0]["decoder_states"].new_zeros(
            batch, max_steps, self.config.text_dim
        )
        hotword_means = samples[0]["hotword_means"].new_zeros(
            batch, max_candidates, self.config.text_dim
        )
        current_tokens = samples[0]["current_token_embeddings"].new_zeros(
            batch, max_steps, max_candidates, self.config.text_dim
        )
        prefixes = samples[0]["prefix_embeddings"].new_zeros(
            batch, max_steps, max_candidates, self.config.text_dim
        )
        scalars = samples[0]["scalar_features"].new_zeros(
            batch, max_steps, max_candidates, self.config.scalar_dim
        )
        audio_features = samples[0]["audio_features"].new_zeros(
            batch, max_audio, self.config.audio_dim
        )
        history_features = samples[0]["history_features"].new_zeros(
            batch, max_history, self.config.text_dim
        )
        history_segments = torch.zeros(
            batch, max_history, dtype=torch.long, device=device
        )
        audio_padding = torch.ones(
            batch, max_audio, dtype=torch.bool, device=device
        )
        history_padding = torch.ones(
            batch, max_history, dtype=torch.bool, device=device
        )
        valid_candidates = torch.zeros(
            batch, max_steps, max_candidates, dtype=torch.bool, device=device
        )
        text_masks = torch.ones(
            batch,
            max_steps * max_candidates,
            max_history,
            dtype=torch.bool,
            device=device,
        )

        for index, sample in enumerate(samples):
            steps = step_lengths[index]
            candidates = candidate_lengths[index]
            audio_length = audio_lengths[index]
            history_length = history_lengths[index]
            decoder_states[index, :steps] = sample["decoder_states"]
            hotword_means[index, :candidates] = sample["hotword_means"]
            current_tokens[index, :steps, :candidates] = sample[
                "current_token_embeddings"
            ]
            prefixes[index, :steps, :candidates] = sample["prefix_embeddings"]
            scalars[index, :steps, :candidates] = sample["scalar_features"]
            audio_features[index, :audio_length] = sample["audio_features"]
            history_features[index, :history_length] = sample["history_features"]
            history_segments[index, :history_length] = sample["history_segment_ids"]
            audio_padding[index, :audio_length] = False
            history_padding[index, :history_length] = False

            sample_candidate_mask = sample.get("candidate_padding_mask")
            if sample_candidate_mask is None:
                sample_valid = torch.ones(
                    steps, candidates, dtype=torch.bool, device=device
                )
            else:
                sample_candidate_mask = sample_candidate_mask.bool()
                if sample_candidate_mask.ndim == 1:
                    sample_candidate_mask = sample_candidate_mask[None].expand(
                        steps, -1
                    )
                elif sample_candidate_mask.shape[0] == 1:
                    sample_candidate_mask = sample_candidate_mask.expand(steps, -1)
                sample_valid = ~sample_candidate_mask[:steps, :candidates]
            valid_candidates[index, :steps, :candidates] = sample_valid

            sample_text_mask = sample["history_attention_mask"].bool()
            for step in range(steps):
                query_start = step * max_candidates
                text_masks[
                    index,
                    query_start : query_start + candidates,
                    :history_length,
                ] = sample_text_mask[step, :history_length]

        decoder = self.decoder_projection(decoder_states)[:, :, None, :].expand(
            -1, -1, max_candidates, -1
        )
        word = self.hotword_projection(hotword_means)[:, None, :, :].expand(
            -1, max_steps, -1, -1
        )
        current = self.current_token_projection(current_tokens)
        prefix = self.prefix_projection(prefixes)
        scalar_state = self.scalar_projection(scalars)
        query = self.query_builder(
            torch.cat((decoder, word, current, prefix, scalar_state), dim=-1)
        )
        flat_query = query.reshape(batch, max_steps * max_candidates, -1)

        audio = self.audio_projection(audio_features)
        audio, audio_padding, _ = self._append_null(
            audio, self.audio_null, audio_padding
        )
        audio_state = flat_query
        for block in self.audio_cross_attention:
            audio_state = block(
                audio_state, audio, memory_padding_mask=audio_padding
            )

        history = self.history_projection(history_features)
        history = history + self.history_segment_embedding(history_segments)
        history, history_padding, _ = self._append_null(
            history, self.text_null, history_padding
        )
        text_masks = torch.cat(
            (
                torch.zeros(
                    batch,
                    max_steps * max_candidates,
                    1,
                    dtype=torch.bool,
                    device=device,
                ),
                text_masks,
            ),
            dim=2,
        ).repeat_interleave(self.config.num_heads, dim=0)
        text_state = flat_query
        for block in self.text_cross_attention:
            text_state = block(
                text_state,
                history,
                memory_padding_mask=history_padding,
                attention_mask=text_masks,
            )

        text_weight = torch.sigmoid(
            self.text_mix(torch.cat((flat_query, audio_state, text_state), dim=-1))
        )
        independent = self.fusion_norm(
            flat_query + audio_state + text_weight * text_state
        )
        independent = independent + self.fusion_scale * self.fusion_ffn(
            self.fusion_ffn_norm(independent)
        )
        independent = independent.reshape(
            batch, max_steps, max_candidates, -1
        )

        flat_independent = independent.reshape(
            batch * max_steps, max_candidates, -1
        )
        valid_steps = torch.tensor(
            [
                step < step_lengths[index]
                for index in range(batch)
                for step in range(max_steps)
            ],
            dtype=torch.bool,
            device=device,
        )
        active_indexes = valid_steps.nonzero(as_tuple=False).squeeze(1)
        related_active = flat_independent.index_select(0, active_indexes)
        candidate_padding = ~valid_candidates.reshape(
            batch * max_steps, max_candidates
        ).index_select(0, active_indexes)
        for block in self.candidate_attention:
            related_active = block(
                related_active, candidate_padding_mask=candidate_padding
            )
        related = torch.zeros_like(flat_independent)
        related.index_copy_(0, active_indexes, related_active)
        related = related.reshape(batch, max_steps, max_candidates, -1)
        final_state = self.direct_relation_merge(
            torch.cat((independent, related), dim=-1)
        )
        raw_score = self.bias_head(final_state).squeeze(-1)
        gain = self.config.max_gain * torch.sigmoid(raw_score)

        base_gap = raw_score.new_zeros(batch, max_steps, max_candidates)
        for index, sample in enumerate(samples):
            steps = step_lengths[index]
            candidates = candidate_lengths[index]
            if "base_gap" in sample:
                sample_gap = sample["base_gap"]
            else:
                base_logits = sample["base_logits"]
                candidate_ids = sample["candidate_token_ids"]
                candidate_logits = base_logits.gather(1, candidate_ids)
                top1_logits = base_logits.max(dim=-1, keepdim=True).values
                sample_gap = (top1_logits - candidate_logits).clamp_min(0.0)
            base_gap[index, :steps, :candidates] = sample_gap
        base_gap = base_gap.detach()
        required_bias = base_gap + self.config.logit_margin
        candidate_bias = (gain.float() * required_bias.float()).clamp(
            min=0.0, max=self.config.max_bias
        )
        candidate_bias = candidate_bias.masked_fill(~valid_candidates, 0.0)

        packed = {
            "candidate_bias": candidate_bias,
            "gain": gain,
            "raw_score": raw_score,
            "base_gap": base_gap,
            "required_bias": required_bias,
            "independent_state": independent,
            "related_state": related,
        }
        outputs: list[dict[str, torch.Tensor]] = []
        for index, sample in enumerate(samples):
            steps = step_lengths[index]
            candidates = candidate_lengths[index]
            output = {
                key: value[index, :steps, :candidates]
                for key, value in packed.items()
            }
            if sample.get("return_full_logits", True):
                if "base_logits" not in sample:
                    raise ValueError("full vocabulary output requires base_logits")
                base_logits = sample["base_logits"]
                delta_logits = torch.zeros_like(base_logits)
                delta_logits.scatter_reduce_(
                    1,
                    sample["candidate_token_ids"],
                    output["candidate_bias"].to(base_logits.dtype),
                    reduce="amax",
                    include_self=True,
                )
                output["delta_logits"] = delta_logits
                output["final_logits"] = base_logits + delta_logits
            outputs.append(output)
        return outputs

    def forward(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]] | torch.Tensor:
        if not samples:
            return sum(parameter.sum() * 0.0 for parameter in self.parameters())
        if len(samples) == 1:
            return [self._forward_sample(samples[0])]

        if not any(
            any(key.startswith("projected_") for key in sample)
            for sample in samples
        ):
            return self._forward_padded_batch(samples)

        outputs: list[dict[str, torch.Tensor] | None] = [None] * len(samples)
        groups: dict[tuple[int, int, int, int], list[int]] = {}
        for index, sample in enumerate(samples):
            steps, candidates = sample["current_token_embeddings"].shape[:2]
            key = (
                int(sample["audio_features"].data_ptr()),
                steps,
                candidates,
                int(sample["history_features"].shape[0]),
            )
            groups.setdefault(key, []).append(index)
        for indexes in groups.values():
            group_samples = [samples[index] for index in indexes]
            if len(group_samples) == 1 or any(
                sample.get("candidate_padding_mask") is not None
                or sample.get("audio_padding_mask") is not None
                or sample.get("history_padding_mask") is not None
                for sample in group_samples
            ):
                group_outputs = [
                    self._forward_sample(sample) for sample in group_samples
                ]
            else:
                group_outputs = self._forward_equal_length_batch(group_samples)
            for index, output in zip(indexes, group_outputs):
                outputs[index] = output
        if any(output is None for output in outputs):
            raise RuntimeError("Fusion batch output assignment is incomplete")
        return outputs  # type: ignore[return-value]


def trainable_parameter_summary(module: nn.Module) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        prefix = name.split(".", 1)[0]
        groups[prefix] = groups.get(prefix, 0) + parameter.numel()
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "groups": groups,
    }
