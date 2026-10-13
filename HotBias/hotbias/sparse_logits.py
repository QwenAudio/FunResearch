"""Exact losses for sparse logit interventions over a frozen vocabulary.

The ASR base logit vector is frozen. Only a small candidate set receives a
learned bias, so the final normalizer can be reconstructed from the base
``logsumexp`` and candidate logits without caching the full vocabulary tensor.
"""

from __future__ import annotations

import torch


def coalesce_candidate_bias(
    candidate_token_ids: torch.Tensor,
    candidate_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the runtime's duplicate-token ``amax`` reduction sparsely.

    Returns the effective bias at every candidate position and a mask selecting
    one representative for each unique token id. Candidate counts are small
    (normally at most 40), making the pairwise equality matrix inexpensive.
    """

    if candidate_token_ids.shape != candidate_bias.shape:
        raise ValueError(
            "candidate_token_ids and candidate_bias must have identical shapes"
        )
    if candidate_token_ids.ndim < 1:
        raise ValueError("candidate tensors must have a candidate dimension")
    same_token = candidate_token_ids.unsqueeze(-1).eq(
        candidate_token_ids.unsqueeze(-2)
    )
    expanded_bias = candidate_bias.unsqueeze(-2).expand_as(same_token)
    negative_infinity = torch.full_like(expanded_bias, float("-inf"))
    effective_bias = torch.where(
        same_token, expanded_bias, negative_infinity
    ).amax(dim=-1).clamp_min(0.0)
    has_earlier_duplicate = torch.tril(same_token, diagonal=-1).any(dim=-1)
    unique_mask = ~has_earlier_duplicate
    return effective_bias, unique_mask


def sparse_biased_nll(
    *,
    base_log_normalizer: torch.Tensor,
    base_target_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    base_candidate_logits: torch.Tensor,
    candidate_bias: torch.Tensor,
    target_token_ids: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute exact NLL after changing only candidate-token logits.

    This is mathematically equivalent to materializing ``base_logits +
    delta_logits`` and applying cross entropy. It needs only two base scalars
    per decoding step plus one logit per candidate.
    """

    if base_log_normalizer.shape != base_target_logits.shape:
        raise ValueError(
            "base_log_normalizer and base_target_logits must have identical shapes"
        )
    if target_token_ids.shape != base_target_logits.shape:
        raise ValueError(
            "target_token_ids must match the non-candidate dimensions"
        )
    if not (
        candidate_token_ids.shape
        == base_candidate_logits.shape
        == candidate_bias.shape
    ):
        raise ValueError("all candidate tensors must have identical shapes")
    if candidate_token_ids.shape[:-1] != base_target_logits.shape:
        raise ValueError(
            "candidate tensor prefix dimensions must match target dimensions"
        )
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    log_normalizer = base_log_normalizer.float()
    candidate_logits = base_candidate_logits.float()
    effective_bias, unique_mask = coalesce_candidate_bias(
        candidate_token_ids, candidate_bias.float()
    )

    base_candidate_probability = torch.exp(
        candidate_logits - log_normalizer.unsqueeze(-1)
    )
    partition_delta = torch.where(
        unique_mask,
        base_candidate_probability * torch.expm1(effective_bias),
        torch.zeros_like(effective_bias),
    ).sum(dim=-1)
    final_log_normalizer = log_normalizer + torch.log1p(
        partition_delta.clamp_min(-1.0 + torch.finfo(torch.float32).eps)
    )

    target_matches = candidate_token_ids.eq(target_token_ids.unsqueeze(-1))
    target_bias = torch.where(
        target_matches,
        effective_bias,
        torch.zeros_like(effective_bias),
    ).amax(dim=-1)
    losses = final_log_normalizer - (
        base_target_logits.float() + target_bias
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
