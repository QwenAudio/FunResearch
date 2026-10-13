from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .attention_fusion import (
    CandidateSelfAttentionBlock,
    CrossAttentionBlock,
    RMSNorm,
    SwiGLU,
)


@dataclass
class BinaryAttentionFusionConfig:
    """Configuration for the binary token-emission hotword branch."""

    audio_dim: int = 2048
    text_dim: int = 2048
    hidden_dim: int = 256
    num_heads: int = 4
    ffn_dim: int = 1024
    audio_cross_layers: int = 4
    text_cross_layers: int = 4
    candidate_layers: int = 8
    scalar_dim: int = 6
    dropout: float = 0.1
    layer_scale_init: float = 0.1
    emit_threshold: float = 0.5
    max_bias: float = 10.0
    logit_margin: float = 0.3
    decision_mode: str = "hard_multi"
    ctc_logit_residual: bool = False
    ctc_logit_residual_max: float = 4.0
    ctc_logit_residual_with_bias: bool = False
    ctc_logit_residual_relative: bool = False
    ctc_base_contrastive_residual: bool = False
    ctc_base_contrastive_residual_max: float = 8.0


class BinaryFrozenFeatureAttentionFusion(nn.Module):
    """Predict whether each current hotword token should be emitted.

    ``hard_multi`` preserves the original checkpoint behavior. The generic
    ``single_continuous`` mode permits at most one candidate per decoding step
    and scales the bounded gap bias by the learned emit probability. This keeps
    weak evidence from deterministically overriding the ASR base distribution.
    """

    base_scalar_names = (
        "base_probability",
        "base_logit_gap",
        "base_rank_normalized",
        "progress",
        "is_start",
        "is_end",
    )
    ctc_scalar_names = (
        "ctc_alignment_log_score",
        "ctc_min_label_log_probability",
    )
    relation_scalar_names = (
        "audio_hotword_top4_cosine",
        "history_hotword_max_cosine",
        "decoder_hotword_cosine",
    )
    candidate_input_scalar_names = (
        "candidate_input_rank_log_normalized",
        "candidate_pool_size_log_normalized",
    )
    ctc_orthographic_scalar_names = (
        "ctc_orthographic_share_log_normalized",
        "ctc_min_orthographic_share_log_normalized",
    )
    scalar_names = (
        base_scalar_names
        + ctc_scalar_names
        + relation_scalar_names
        + candidate_input_scalar_names
        + ctc_orthographic_scalar_names
    )

    def __init__(self, config: BinaryAttentionFusionConfig | None = None):
        super().__init__()
        self.config = config or BinaryAttentionFusionConfig()
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
        self.scalar_projection = nn.Sequential(
            nn.Linear(config.scalar_dim, hidden), RMSNorm(hidden), nn.SiLU()
        )

        # Prefix embedding was deliberately removed. The four inputs are the
        # decoder state, complete hotword, current token and scalar base scores.
        self.query_builder = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
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
        self.classifier_head = nn.Sequential(
            RMSNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        final = self.classifier_head[-1]
        nn.init.normal_(final.weight, std=1e-3)
        nn.init.constant_(final.bias, -2.0)
        if config.ctc_logit_residual:
            if config.scalar_dim < 15:
                raise ValueError(
                    "CTC logit residual requires the 15-column scalar input"
                )
            residual_input_dim = 8 if config.ctc_logit_residual_relative else 4
            self.ctc_logit_residual_head = nn.Linear(
                residual_input_dim, 1, bias=config.ctc_logit_residual_with_bias
            )
            nn.init.zeros_(self.ctc_logit_residual_head.weight)
            if self.ctc_logit_residual_head.bias is not None:
                nn.init.zeros_(self.ctc_logit_residual_head.bias)
        if config.ctc_base_contrastive_residual:
            # Candidate score, Base score, their margin, availability, plus
            # rank/progress/start/end.  This head is deliberately independent
            # from the main classifier so acoustic calibration cannot rewrite
            # the frozen semantic decision path.
            self.ctc_base_contrastive_residual_head = nn.Linear(8, 1, bias=False)
            nn.init.zeros_(self.ctc_base_contrastive_residual_head.weight)

    @staticmethod
    def _append_null(
        memory: torch.Tensor,
        null: torch.Tensor,
        padding_mask: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        memory = torch.cat((null.expand(memory.shape[0], -1, -1), memory), dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat(
                (
                    torch.zeros(
                        padding_mask.shape[0],
                        1,
                        dtype=torch.bool,
                        device=padding_mask.device,
                    ),
                    padding_mask.bool(),
                ),
                dim=1,
            )
        if attention_mask is not None:
            attention_mask = torch.cat(
                (
                    torch.zeros(
                        attention_mask.shape[0],
                        1,
                        dtype=torch.bool,
                        device=attention_mask.device,
                    ),
                    attention_mask.bool(),
                ),
                dim=1,
            )
        return memory, padding_mask, attention_mask

    def _decision(
        self,
        final_state: torch.Tensor,
        base_logits: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_padding_mask: torch.Tensor | None = None,
        scalar_features: torch.Tensor | None = None,
        ctc_residual_mask: torch.Tensor | None = None,
        candidate_competition_group_ids: torch.Tensor | None = None,
        ctc_base_contrastive_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base_emit_logits = self.classifier_head(final_state).squeeze(-1)
        ctc_logit_residual = torch.zeros_like(base_emit_logits)
        if self.config.ctc_logit_residual:
            if scalar_features is None or scalar_features.shape[-1] < 15:
                raise ValueError(
                    "CTC logit residual requires 15-column scalar features"
                )
            ctc_features = torch.cat(
                (scalar_features[..., 6:8], scalar_features[..., 13:15]), dim=-1
            )
            if self.config.ctc_logit_residual_relative:
                relative_padding_mask = candidate_padding_mask
                if ctc_residual_mask is not None:
                    residual_padding = ~ctc_residual_mask.bool()
                    relative_padding_mask = (
                        residual_padding
                        if relative_padding_mask is None
                        else relative_padding_mask.bool() | residual_padding
                    )
                relative_features = self._relative_ctc_features(
                    ctc_features,
                    base_emit_logits,
                    relative_padding_mask,
                    candidate_competition_group_ids,
                )
                ctc_features = torch.cat((ctc_features, relative_features), dim=-1)
            raw_residual = self.ctc_logit_residual_head(ctc_features).squeeze(-1)
            ctc_logit_residual = float(self.config.ctc_logit_residual_max) * torch.tanh(
                raw_residual
            )
            # The learned intercept calibrates real CTC evidence only. Missing
            # alignments (for example unsupported English words) stay unchanged.
            has_ctc_alignment = ctc_features[..., :2].gt(0.0).any(dim=-1)
            if ctc_residual_mask is not None:
                has_ctc_alignment = has_ctc_alignment & ctc_residual_mask.bool()
            ctc_logit_residual = ctc_logit_residual * has_ctc_alignment
        ctc_base_contrastive_residual = torch.zeros_like(base_emit_logits)
        if self.config.ctc_base_contrastive_residual:
            if (
                ctc_base_contrastive_features is None
                or ctc_base_contrastive_features.shape[-1] != 4
            ):
                raise ValueError(
                    "CTC-vs-Base residual requires four contrastive features"
                )
            if scalar_features is None or scalar_features.shape[-1] < 6:
                raise ValueError(
                    "CTC-vs-Base residual requires the six base scalar features"
                )
            direct_features = torch.cat(
                (
                    ctc_base_contrastive_features,
                    scalar_features[..., 2:6],
                ),
                dim=-1,
            )
            raw_contrastive = self.ctc_base_contrastive_residual_head(
                direct_features
            ).squeeze(-1)
            available = ctc_base_contrastive_features[..., 3].gt(0.5)
            ctc_base_contrastive_residual = (
                float(self.config.ctc_base_contrastive_residual_max)
                * torch.tanh(raw_contrastive)
                * available
            )
        emit_logits = (
            base_emit_logits
            + ctc_logit_residual
            + ctc_base_contrastive_residual
        )
        emit_probability = torch.sigmoid(emit_logits)
        candidate_logits = base_logits.gather(-1, candidate_token_ids)
        top1_logits = base_logits.max(dim=-1, keepdim=True).values
        base_gap = (top1_logits - candidate_logits).clamp_min(0.0).detach()
        required_bias = (base_gap + self.config.logit_margin).clamp(
            max=self.config.max_bias
        )
        valid_candidates = torch.ones_like(emit_probability, dtype=torch.bool)
        if candidate_padding_mask is not None:
            valid_candidates = ~candidate_padding_mask.bool()

        if self.config.decision_mode == "hard_multi":
            emit_decision = (
                emit_probability.ge(self.config.emit_threshold) & valid_candidates
            )
            candidate_bias = torch.where(
                emit_decision,
                required_bias.float(),
                torch.zeros_like(required_bias, dtype=torch.float32),
            )
        elif self.config.decision_mode in {"single_continuous", "single_hard"}:
            # Candidate competition is resolved before logit injection. Without
            # this constraint, several words can independently receive enough
            # bias to tie the base top-1 token in the same decoder step.
            masked_probability = emit_probability.masked_fill(
                ~valid_candidates, -1.0
            )
            winner_index = masked_probability.argmax(dim=-1, keepdim=True)
            winner = torch.zeros_like(valid_candidates)
            winner.scatter_(-1, winner_index, True)
            winner = winner & valid_candidates
            emit_decision = (
                winner & emit_probability.ge(self.config.emit_threshold)
            )
            if self.config.decision_mode == "single_continuous":
                applied_bias = emit_probability.float() * required_bias.float()
            else:
                # Probability controls the gate; a positive binary action must
                # still be strong enough to overtake the current Base top-1.
                applied_bias = required_bias.float()
            candidate_bias = torch.where(
                emit_decision,
                applied_bias,
                torch.zeros_like(required_bias, dtype=torch.float32),
            )
        else:
            raise ValueError(
                f"unsupported binary decision mode: {self.config.decision_mode}"
            )
        return {
            "base_emit_logits": base_emit_logits,
            "ctc_logit_residual": ctc_logit_residual,
            "ctc_base_contrastive_residual": ctc_base_contrastive_residual,
            "emit_logits": emit_logits,
            "emit_probability": emit_probability,
            "emit_decision": emit_decision,
            "candidate_bias": candidate_bias,
            "base_gap": base_gap,
            "required_bias": required_bias,
        }

    @staticmethod
    def _relative_ctc_features(
        ctc_features: torch.Tensor,
        base_emit_logits: torch.Tensor,
        candidate_padding_mask: torch.Tensor | None,
        candidate_competition_group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compare each candidate with its strongest valid emit competitor.

        When competition groups are supplied, only candidates on the same Trie
        branch can compete. This prevents an unrelated acoustically supported
        word from suppressing an otherwise valid word.
        """

        candidate_count = ctc_features.shape[-2]
        if candidate_count < 2:
            return torch.zeros_like(ctc_features)
        valid = torch.ones_like(base_emit_logits, dtype=torch.bool)
        if candidate_padding_mask is not None:
            valid = ~candidate_padding_mask.bool()
        if candidate_competition_group_ids is not None:
            if candidate_competition_group_ids.shape != base_emit_logits.shape:
                raise ValueError(
                    "candidate competition group ids must match candidate logits"
                )
            same_group = candidate_competition_group_ids.unsqueeze(-1).eq(
                candidate_competition_group_ids.unsqueeze(-2)
            )
            same_group = same_group & valid.unsqueeze(-2)
            same_group = same_group & ~torch.eye(
                candidate_count,
                dtype=torch.bool,
                device=ctc_features.device,
            ).view(*([1] * (ctc_features.ndim - 2)), candidate_count, candidate_count)
            competitor_logits = base_emit_logits.unsqueeze(-2).expand(
                *same_group.shape
            ).masked_fill(
                ~same_group, torch.finfo(base_emit_logits.dtype).min
            )
            competitor_indices = competitor_logits.argmax(dim=-1)
            competitor_features = ctc_features.gather(
                -2,
                competitor_indices.unsqueeze(-1).expand(*ctc_features.shape),
            )
            has_competitor = same_group.any(dim=-1)
            return (ctc_features - competitor_features) * (
                valid & has_competitor
            ).unsqueeze(-1)
        masked_logits = base_emit_logits.masked_fill(
            ~valid, torch.finfo(base_emit_logits.dtype).min
        )
        top_values, top_indices = masked_logits.topk(2, dim=-1)
        candidate_indices = torch.arange(
            candidate_count, device=ctc_features.device
        ).view(*([1] * (ctc_features.ndim - 2)), candidate_count)
        strongest = top_indices[..., 0].unsqueeze(-1)
        runner_up = top_indices[..., 1].unsqueeze(-1)
        competitor_indices = torch.where(
            candidate_indices.eq(strongest), runner_up, strongest
        )
        competitor_features = ctc_features.gather(
            -2,
            competitor_indices.unsqueeze(-1).expand(*ctc_features.shape),
        )
        has_competitor = valid.sum(dim=-1, keepdim=True).gt(1)
        return (ctc_features - competitor_features) * (
            valid & has_competitor
        ).unsqueeze(-1)

    def _forward_sample(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        decoder_states = sample["decoder_states"]
        hotword_means = sample["hotword_means"]
        current_token_embeddings = sample["current_token_embeddings"]
        scalar_features = sample["scalar_features"]
        steps, candidates = current_token_embeddings.shape[:2]

        decoder = self.decoder_projection(decoder_states)[:, None, :].expand(
            -1, candidates, -1
        )
        word = self.hotword_projection(hotword_means)[None, :, :].expand(
            steps, -1, -1
        )
        current = self.current_token_projection(current_token_embeddings)
        scalars = self.scalar_projection(scalar_features)
        query = self.query_builder(torch.cat((decoder, word, current, scalars), dim=-1))
        flat_query = query.reshape(1, steps * candidates, -1)

        audio = self.audio_projection(sample["audio_features"])[None, :, :]
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
        output = self._decision(
            final_state,
            sample["base_logits"],
            sample["candidate_token_ids"],
            sample.get("candidate_decision_padding_mask", candidate_padding),
            scalar_features,
            sample.get("ctc_residual_mask"),
            sample.get("candidate_competition_group_ids"),
            sample.get("ctc_base_contrastive_features"),
        )
        delta_logits = torch.zeros_like(sample["base_logits"])
        delta_logits.scatter_reduce_(
            1,
            sample["candidate_token_ids"],
            output["candidate_bias"].to(delta_logits.dtype),
            reduce="amax",
            include_self=True,
        )
        output.update(
            {
                "delta_logits": delta_logits,
                "final_logits": sample["base_logits"] + delta_logits,
                "independent_state": independent,
                "related_state": related,
            }
        )
        return output

    def forward_online_batch(
        self,
        *,
        decoder_states: torch.Tensor,
        hotword_means: torch.Tensor,
        current_token_embeddings: torch.Tensor,
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
        ctc_residual_mask: torch.Tensor | None = None,
        candidate_competition_group_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Vectorized one-step inference over live rollouts."""

        batch, candidates = current_token_embeddings.shape[:2]
        decoder = self.decoder_projection(decoder_states)[:, None, :].expand(
            -1, candidates, -1
        )
        word = self.hotword_projection(hotword_means)
        current = self.current_token_projection(current_token_embeddings)
        scalars = self.scalar_projection(scalar_features)
        query = self.query_builder(torch.cat((decoder, word, current, scalars), dim=-1))

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
            related = block(related, candidate_padding_mask=candidate_padding_mask)
        final_state = self.direct_relation_merge(
            torch.cat((independent, related), dim=-1)
        )
        output = self._decision(
            final_state,
            base_logits,
            candidate_token_ids,
            candidate_padding_mask,
            scalar_features,
            ctc_residual_mask,
            candidate_competition_group_ids,
        )
        output.update(
            {"independent_state": independent, "related_state": related}
        )
        return output

    def forward(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]] | torch.Tensor:
        if not samples:
            return sum(parameter.sum() * 0.0 for parameter in self.parameters())
        return [self._forward_sample(sample) for sample in samples]


def explicit_relation_scalar_features(
    sample: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return frozen-base audio, history and decoder relation evidence."""

    decoder = sample["decoder_states"].float()
    hotwords = sample["hotword_means"].float()
    audio = sample["audio_features"].float()
    history = sample["history_features"].float()
    history_mask = sample["history_attention_mask"].bool()
    steps = int(decoder.shape[0])
    candidates = int(hotwords.shape[0])

    def normalized(values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(values, dim=-1, eps=1e-6)

    hotword_unit = normalized(hotwords)
    if audio.numel():
        audio_scores = normalized(audio) @ hotword_unit.transpose(0, 1)
        width = min(4, int(audio_scores.shape[0]))
        audio_relation = audio_scores.topk(width, dim=0).values.mean(dim=0)
    else:
        audio_relation = hotwords.new_zeros((candidates,), dtype=torch.float32)
    audio_relation = audio_relation[None, :].expand(steps, -1)

    decoder_relation = normalized(decoder) @ hotword_unit.transpose(0, 1)

    if history.numel():
        history_scores = normalized(history) @ hotword_unit.transpose(0, 1)
        allowed = ~history_mask
        expanded = history_scores[None, :, :].expand(steps, -1, -1)
        masked = expanded.masked_fill(~allowed[:, :, None], float("-inf"))
        history_relation = masked.amax(dim=1)
        history_relation = torch.where(
            allowed.any(dim=1, keepdim=True),
            history_relation,
            torch.zeros_like(history_relation),
        )
    else:
        history_relation = hotwords.new_zeros(
            (steps, candidates), dtype=torch.float32
        )

    return torch.stack(
        (audio_relation, history_relation, decoder_relation), dim=-1
    ).clamp_(-1.0, 1.0)


def candidate_input_scalar_features(
    *,
    steps: int,
    candidate_count: int,
    device: torch.device,
    word_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode input-list rank and pool size using only request-side data."""

    count = max(int(candidate_count), 1)
    if word_indices is None:
        ranks = torch.arange(1, count + 1, dtype=torch.float32, device=device)
    else:
        ranks = word_indices.to(device=device, dtype=torch.float32).add(1.0)
    rank_feature = torch.log1p(ranks) / torch.log1p(
        ranks.new_tensor(float(count))
    ).clamp_min(1e-6)
    pool_feature = (
        torch.log1p(ranks.new_tensor(float(count)))
        / torch.log1p(ranks.new_tensor(2000.0))
    ).clamp_(0.0, 1.0)
    per_candidate = torch.stack(
        (rank_feature.clamp_(0.0, 1.0), pool_feature.expand_as(rank_feature)),
        dim=-1,
    )
    return per_candidate[None, :, :].expand(max(int(steps), 0), -1, -1)


def binary_candidate_labels(
    candidate_token_ids: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Build the teacher-forced emit/not-emit target for every candidate."""

    return candidate_token_ids.eq(target_ids[..., None]).float()


def binary_word_path_labels(
    hotword_token_ids: list[list[int]] | list[tuple[int, ...]],
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Label a candidate only when its complete remaining word path is correct.

    Token-only labels make every word sharing the next token positive. For
    example, ``xiao-yi`` and ``xiao-yu`` both become positive before ``xiao`` is
    emitted, so the decoder can start one word and finish another. This label
    follows the same Trie state as inference and requires the whole remaining
    candidate sequence to match the teacher-forced reference.
    """

    sequences = [tuple(int(token) for token in sequence) for sequence in hotword_token_ids]
    if any(not sequence for sequence in sequences):
        raise ValueError("hotword token sequences must be non-empty")
    targets = [int(token) for token in target_ids.tolist()]
    depths = [0] * len(sequences)
    labels = torch.zeros(
        len(targets),
        len(sequences),
        dtype=torch.float32,
        device=target_ids.device,
    )

    for step, target in enumerate(targets):
        for word_index, sequence in enumerate(sequences):
            depth = depths[word_index]
            token_index = depth if 0 < depth < len(sequence) else 0
            remaining = sequence[token_index:]
            if targets[step : step + len(remaining)] == list(remaining):
                labels[step, word_index] = 1.0

        old_depths = list(depths)
        for word_index, sequence in enumerate(sequences):
            depth = old_depths[word_index]
            if 0 < depth < len(sequence):
                if target == sequence[depth]:
                    next_depth = depth + 1
                    depths[word_index] = 0 if next_depth == len(sequence) else next_depth
                elif len(sequence) > 1 and target == sequence[0]:
                    depths[word_index] = 1
                else:
                    depths[word_index] = 0
            elif len(sequence) > 1 and target == sequence[0]:
                depths[word_index] = 1

    return labels


def initialize_binary_from_regression_checkpoint(
    checkpoint: Mapping[str, Any] | str | Path,
    *,
    audio_cross_layers: int = 4,
    text_cross_layers: int = 4,
    candidate_layers: int = 8,
) -> tuple[BinaryFrozenFeatureAttentionFusion, dict[str, Any]]:
    """Warm-start the binary model from the latest regression checkpoint."""

    payload = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, (str, Path))
        else dict(checkpoint)
    )
    source_config = dict(payload["config"])
    config = BinaryAttentionFusionConfig(
        audio_dim=int(source_config["audio_dim"]),
        text_dim=int(source_config["text_dim"]),
        hidden_dim=int(source_config["hidden_dim"]),
        num_heads=int(source_config["num_heads"]),
        ffn_dim=int(source_config["ffn_dim"]),
        audio_cross_layers=audio_cross_layers,
        text_cross_layers=text_cross_layers,
        candidate_layers=candidate_layers,
        scalar_dim=int(source_config["scalar_dim"]),
        dropout=float(source_config["dropout"]),
        layer_scale_init=float(source_config["layer_scale_init"]),
        max_bias=float(source_config["max_bias"]),
        logit_margin=float(source_config["logit_margin"]),
    )
    model = BinaryFrozenFeatureAttentionFusion(config)
    source = payload["model_state"]
    target = model.state_dict()
    initialized: dict[str, torch.Tensor] = {}

    for name, value in target.items():
        if name in source and source[name].shape == value.shape:
            initialized[name] = source[name]

    hidden = config.hidden_dim
    old_query = source.get("query_builder.0.weight")
    if old_query is not None and old_query.shape[1] == hidden * 5:
        initialized["query_builder.0.weight"] = torch.cat(
            (old_query[:, : hidden * 3], old_query[:, hidden * 4 : hidden * 5]),
            dim=1,
        )

    layer_specs = (
        ("audio_cross_attention", int(source_config["audio_cross_layers"]), audio_cross_layers),
        ("text_cross_attention", int(source_config["text_cross_layers"]), text_cross_layers),
        ("candidate_attention", int(source_config["candidate_layers"]), candidate_layers),
    )
    duplicated_layers: dict[str, dict[int, int]] = {}
    for prefix, source_layers, target_layers in layer_specs:
        duplicated_layers[prefix] = {}
        for target_index in range(target_layers):
            source_index = target_index % source_layers
            duplicated_layers[prefix][target_index] = source_index
            source_prefix = f"{prefix}.{source_index}."
            target_prefix = f"{prefix}.{target_index}."
            for source_name, value in source.items():
                if not source_name.startswith(source_prefix):
                    continue
                target_name = target_prefix + source_name[len(source_prefix) :]
                if target_name in target and target[target_name].shape == value.shape:
                    initialized[target_name] = value

    # Reuse the old head's representation layers, but keep a fresh binary
    # output layer because its target semantics differ from regression gain.
    for suffix in ("0.weight", "1.weight", "1.bias", "4.weight", "4.bias"):
        source_name = f"bias_head.{suffix}"
        target_name = f"classifier_head.{suffix}"
        if source_name in source and source[source_name].shape == target[target_name].shape:
            initialized[target_name] = source[source_name]

    result = model.load_state_dict(initialized, strict=False)
    report = {
        "source_checkpoint": str(checkpoint) if isinstance(checkpoint, (str, Path)) else "mapping",
        "source_architecture": {
            "audio_cross_layers": int(source_config["audio_cross_layers"]),
            "text_cross_layers": int(source_config["text_cross_layers"]),
            "candidate_layers": int(source_config["candidate_layers"]),
            "uses_prefix_embedding": True,
            "head": "regression_gain",
        },
        "target_architecture": {
            "audio_cross_layers": audio_cross_layers,
            "text_cross_layers": text_cross_layers,
            "candidate_layers": candidate_layers,
            "uses_prefix_embedding": False,
            "head": "binary_emit",
        },
        "duplicated_layers": duplicated_layers,
        "loaded_tensors": len(initialized),
        "total_tensors": len(target),
        "loaded_parameters": sum(target[name].numel() for name in initialized),
        "total_parameters": sum(value.numel() for value in target.values()),
        "fresh_tensors": sorted(result.missing_keys),
        "unexpected_tensors": sorted(result.unexpected_keys),
    }
    return model, report
