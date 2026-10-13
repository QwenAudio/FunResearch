"""Frozen FireRed2 features for the rank+Trie pilot (no native hotword prompt).

Candidate attention sees real candidates even when Top-K rejects every token.
Always use ``gated_fusion_output``: it enforces the rank gate at logit injection,
independently of attention masking, without modifying any shared core parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import torch

from firered_asr2_aed.model_bridge import FrozenFireRed2Bridge


def load_bridge(model_dir: str, firered_repo: str | None = None, device: str = "cpu") -> FrozenFireRed2Bridge:
    """Load official AED code and model supplied by the caller."""
    return FrozenFireRed2Bridge(model_dir=model_dir, firered_repo=firered_repo, device=device)


@dataclass
class Candidates:
    words: list[str]
    token_ids: list[list[int]]
    embeddings: list[torch.Tensor]
    means: torch.Tensor
    statistics: dict[str, int]


def _word_list(hotwords: Sequence[str] | str | None) -> list[str]:
    if hotwords is None:
        return []
    values = re.split(r"[,，、;；\n]+", hotwords) if isinstance(hotwords, str) else hotwords
    return [str(value).strip().strip("`").strip() for value in values if str(value).strip()]


@torch.no_grad()
def prepare_candidates(bridge, hotwords, max_candidates: int = 40) -> Candidates:
    """Preserve the first valid distinct token sequences in the supplied order.

    No reference, target ID, or oracle membership is accepted by this function.
    The raw list can contain English multiword phrases; spaces are not split.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    decoder = bridge.model.decoder
    embed = decoder.tgt_word_emb
    vocab = int(embed.num_embeddings)
    unknown = int(bridge.tokenizer.dict.unkid)
    forbidden = {0, unknown, int(decoder.pad_id), int(decoder.sos_id), int(decoder.eos_id)}
    words = _word_list(hotwords)
    kept_words, kept_ids, embeddings = [], [], []
    seen: set[tuple[int, ...]] = set()
    counts = {"input_hotwords": len(words), "invalid_hotwords": 0,
              "duplicate_hotwords": 0, "cap_dropped_hotwords": 0, "kept_hotwords": 0}
    for word in words:
        result = bridge.tokenize(word)
        ids = [int(value) for value in result["ids"]]
        if not ids or any(value in forbidden or not 0 <= value < vocab for value in ids):
            counts["invalid_hotwords"] += 1
            continue
        key = tuple(ids)
        if key in seen:
            counts["duplicate_hotwords"] += 1
            continue
        seen.add(key)
        if len(kept_ids) >= max_candidates:
            counts["cap_dropped_hotwords"] += 1
            continue
        token_tensor = torch.tensor(ids, dtype=torch.long, device=embed.weight.device)
        kept_words.append(word)
        kept_ids.append(ids)
        embeddings.append(embed(token_tensor).detach().float())
    counts["kept_hotwords"] = len(kept_ids)
    means = (torch.stack([value.mean(dim=0) for value in embeddings]) if embeddings
             else embed.weight.new_empty((0, int(embed.embedding_dim))).float())
    return Candidates(kept_words, kept_ids, embeddings, means, counts)


def matched_prefix_length(generated: Sequence[int], word: Sequence[int]) -> int:
    """Longest generated suffix equal to a proper prefix of this hotword."""
    for width in range(min(len(generated), len(word) - 1), 0, -1):
        if list(generated[-width:]) == list(word[:width]):
            return width
    return 0


@torch.no_grad()
def _assemble_sample(
    *, candidates: Candidates, base_logits: torch.Tensor, decoder_states: torch.Tensor,
    audio_features: torch.Tensor, history_features: torch.Tensor,
    history_attention_mask: torch.Tensor, generated_prefixes: Sequence[Sequence[int]],
    rank_gate: int,
) -> dict[str, Any]:
    if not candidates.token_ids:
        raise ValueError("no valid hotword candidates after target tokenization")
    if rank_gate < 1:
        raise ValueError("rank_gate must be positive (this pilot never disables it)")
    base_logits = base_logits.detach().float()
    device = base_logits.device
    steps, vocabulary = base_logits.shape
    if len(generated_prefixes) != steps:
        raise ValueError("one causal generated prefix is required per decoding step")
    index_values = [
        [matched_prefix_length(prefix, word) for word in candidates.token_ids]
        for prefix in generated_prefixes
    ]
    token_indexes = torch.tensor(index_values, dtype=torch.long, device=device)
    candidate_ids = torch.tensor([
        [word[index_values[step][index]] for index, word in enumerate(candidates.token_ids)]
        for step in range(steps)
    ], dtype=torch.long, device=device)
    padded_embeddings = torch.nn.utils.rnn.pad_sequence(candidates.embeddings, batch_first=True)
    prefixes = []
    for values in candidates.embeddings:
        prefix_means = torch.zeros_like(values)
        if len(values) > 1:
            divisors = torch.arange(1, len(values), device=device, dtype=values.dtype)[:, None]
            prefix_means[1:] = values.cumsum(dim=0)[:-1] / divisors
        prefixes.append(prefix_means)
    padded_prefixes = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
    gather = token_indexes.T[:, :, None].expand(-1, -1, padded_embeddings.shape[-1])
    current_embeddings = padded_embeddings.gather(1, gather).transpose(0, 1)
    prefix_embeddings = padded_prefixes.gather(1, gather).transpose(0, 1)
    lengths = torch.tensor([len(word) for word in candidates.token_ids], device=device)[None, :]
    candidate_logits = base_logits.gather(1, candidate_ids)
    normalizer = torch.logsumexp(base_logits, dim=-1)
    probabilities = torch.exp(candidate_logits - normalizer[:, None])
    rank_width = min(100, vocabulary)
    top_width = min(max(rank_width, rank_gate), vocabulary)
    top_values, top_ids = base_logits.topk(top_width, dim=-1)
    gap = (top_values[:, :1] - candidate_logits).clamp_min(0.0)
    matches = top_ids[:, :rank_width, None].eq(candidate_ids[:, None, :])
    rank_values = torch.arange(1, rank_width + 1, dtype=torch.float32, device=device)[None, :, None]
    found_rank = torch.where(matches, rank_values, float(rank_width + 1)).amin(dim=1)
    normalized_rank = torch.log1p(found_rank) / math.log1p(rank_width + 1)
    gate = top_ids[:, :min(rank_gate, vocabulary), None].eq(candidate_ids[:, None, :]).any(dim=1)
    scalar_features = torch.stack([
        probabilities, gap, normalized_rank,
        token_indexes.float() / (lengths - 1).clamp_min(1).float(),
        token_indexes.eq(0).float(), token_indexes.eq(lengths - 1).float(),
    ], dim=-1)
    return {
        "decoder_states": decoder_states.detach().float(),
        "audio_features": audio_features.detach().float(),
        "history_features": history_features.detach().float(),
        "history_segment_ids": torch.ones(history_features.shape[0], dtype=torch.long, device=device),
        "history_attention_mask": history_attention_mask.bool(),
        "hotword_means": candidates.means,
        "current_token_embeddings": current_embeddings,
        "prefix_embeddings": prefix_embeddings,
        "scalar_features": scalar_features,
        "candidate_token_ids": candidate_ids,
        "candidate_token_indexes": token_indexes,
        # Padding denotes absent candidates, not rank rejection. All retained
        # candidates are real, so no all-masked self-attention row is possible.
        "candidate_padding_mask": torch.zeros_like(gate),
        "candidate_gate_mask": gate,
        "base_candidate_ranks": found_rank,
        "base_logits": base_logits,
        "base_gap": gap,
        "base_log_normalizer": normalizer,
        "base_candidate_logits": candidate_logits,
        "candidate_statistics": dict(candidates.statistics),
        "rank_gate": int(rank_gate),
        "ctc_gate_enabled": False,
        "return_full_logits": False,
    }


@torch.no_grad()
def build_training_sample(
    bridge, encoded, target_ids, hotwords, rank_gate=50, max_candidates=40, step_indices=None,
) -> dict[str, Any]:
    """Build one sample; selected steps index ``target_ids + [EOS]``.

    Runs frozen teacher forcing once. Only the reference prefix at each step
    constructs candidate state; current/future labels never select hotwords.
    Empty/unmappable candidate sets raise ValueError and must be counted as skips.
    """
    targets = [int(value) for value in target_ids]
    if int(bridge.model.eos_id) in targets or int(bridge.model.sos_id) in targets:
        raise ValueError("target_ids must contain transcript tokens only, without SOS/EOS")
    candidates = prepare_candidates(bridge, hotwords, max_candidates)
    if not candidates.token_ids:
        raise ValueError("no valid hotword candidates after target tokenization")
    prepared = bridge.teacher_force(encoded, targets)
    labels = prepared["label_ids"].long()
    indices = list(range(len(labels))) if step_indices is None else [int(value) for value in step_indices]
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= value < len(labels) for value in indices):
        raise ValueError("step_indices must be nonempty, unique, and within target_ids+[EOS]")
    device = labels.device
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    # Match the source core's generated-token memory convention: no SOS; at t,
    # only transcript tokens strictly before t can be attended to.
    history = bridge.model.decoder.tgt_word_emb(labels).detach().float()
    history_mask = torch.arange(len(labels), device=device)[None, :] >= index_tensor[:, None]
    sample = _assemble_sample(
        candidates=candidates, base_logits=prepared["base_logits"].index_select(0, index_tensor),
        decoder_states=prepared["decoder_states"].index_select(0, index_tensor),
        audio_features=prepared["audio_features"], history_features=history,
        history_attention_mask=history_mask,
        generated_prefixes=[targets[:step] for step in indices], rank_gate=rank_gate,
    )
    sample["label_ids"] = labels.index_select(0, index_tensor)
    sample["target_token_ids"] = sample["label_ids"]
    sample["step_indices"] = index_tensor
    sample["target_candidate_mask"] = sample["candidate_token_ids"].eq(sample["label_ids"][:, None])
    return sample


@torch.no_grad()
def build_online_sample(bridge, encoded, generated_ids, decoder_state, raw_logits, candidates, rank_gate=50):
    embed = bridge.model.decoder.tgt_word_emb
    generated = [int(value) for value in generated_ids]
    ids = torch.tensor(generated, dtype=torch.long, device=raw_logits.device)
    history = embed(ids).detach().float()
    return _assemble_sample(
        candidates=candidates, base_logits=raw_logits.reshape(1, -1),
        decoder_states=decoder_state.reshape(1, -1),
        audio_features=encoded["encoder_outputs"][0, :int(encoded["encoder_lengths"][0])],
        history_features=history,
        history_attention_mask=torch.zeros((1, len(generated)), dtype=torch.bool, device=ids.device),
        generated_prefixes=[generated], rank_gate=rank_gate,
    )


def gated_fusion_output(fusion, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Apply Top-K explicitly, then coalesce duplicate token deltas with amax.

    Intentionally NOT no_grad: a frozen core must transmit gradients to the six
    projections. Do not use the core's un-gated candidate scores for training.
    """
    output = fusion([sample])[0]
    gate = sample["candidate_gate_mask"] & ~sample["candidate_padding_mask"]
    candidate_bias = output["candidate_bias"].float().masked_fill(~gate, 0.0)
    if not bool(torch.isfinite(candidate_bias).all()):
        raise FloatingPointError("non-finite gated candidate bias")
    base = sample["base_logits"]
    delta = torch.zeros_like(base).scatter_reduce(
        1, sample["candidate_token_ids"], candidate_bias.to(base.dtype),
        reduce="amax", include_self=True,
    )
    return {**output, "candidate_bias": candidate_bias, "delta_logits": delta,
            "final_logits": base + delta, "candidate_gate_mask": gate}
