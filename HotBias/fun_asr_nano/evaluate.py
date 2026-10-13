#!/usr/bin/env python3
"""Compare Fun-ASR-Nano hotword-prompt Base against the aligned bias adapter."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import unicodedata

import numpy as np
import torch
import torch.distributed as dist
from pypinyin import Style, lazy_pinyin
from transformers import LogitsProcessor, LogitsProcessorList


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hotbias import (  # noqa: E402
    AttentionFusionConfig,
    FrozenFeatureAttentionFusion,
    candidate_input_scalar_features,
    explicit_relation_scalar_features,
)
from fun_asr_nano.train import (  # noqa: E402
    normalize_for_match,
    reference_text,
    split_hotwords,
    tokenize_hotwords,
)
from fun_asr_nano.ctc_temporal_gate import (  # noqa: E402
    CTCAudioPosterior,
    CTCTemporalGate,
    OrderedAlignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--nano-model-code",
        type=Path,
        help="Path to model.py in the Fun-ASR repository when Nano is not registered.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON array of dataset specifications; see fun_asr_nano/README.md.",
    )
    parser.add_argument("--base-topk-gate", type=int, default=50)
    parser.add_argument(
        "--ablate-rank-gate",
        action="store_true",
        help="Allow every candidate rank while retaining rank as a model feature.",
    )
    parser.add_argument(
        "--ablate-trie-order",
        action="store_true",
        help="Use each hotword's highest-base-logit token instead of its Trie successor.",
    )
    parser.add_argument("--candidate-cap", type=int, default=40)
    parser.add_argument(
        "--bias-scale",
        type=float,
        default=1.0,
        help="Scale adapter bias after candidate and Top-K gating; 1.0 is unchanged.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-history-tokens", type=int, default=384)
    parser.add_argument(
        "--ctc-temporal-gate",
        action="store_true",
        help="Enable the optional external character CTC acoustic gate.",
    )
    parser.add_argument(
        "--ctc-model-dir",
        type=Path,
        help="Local Wav2Vec2ForCTC directory, required with --ctc-temporal-gate.",
    )
    parser.add_argument(
        "--fixed-bias",
        type=float,
        default=None,
        help="Replace each admitted candidate's learned increment with this fixed logit increment.",
    )
    parser.set_defaults(
        ablate_ctc_gate=False,
        ctc_match_mode="character",
        ctc_start_threshold=1e-3,
        ctc_start_min_label_prob=1e-5,
        ctc_lookback_seconds=0.25,
        ctc_start_min_lookahead_seconds=1.2,
        ctc_start_lookahead_per_label_seconds=0.55,
        ctc_start_max_lookahead_seconds=4.0,
        ctc_max_gap_seconds=0.8,
        ctc_continuation_lookahead_seconds=0.8,
        ctc_cursor_lookahead_seconds=2.5,
        ctc_cursor_min_score=1e-4,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--record-timing",
        action="store_true",
        help="Synchronize CUDA and record per-sample phase timings for profiling.",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=None,
        help=(
            "Reuse baseline_prediction from a previous predictions.jsonl. "
            "Dataset key, model, prompt hotwords, history, label, audio and "
            "generation limit must match exactly."
        ),
    )
    parser.add_argument(
        "--disable-adapter-runtime-cache",
        action="store_true",
        help="Disable cached static Adapter projections for numerical A/B checks.",
    )
    parser.add_argument(
        "--case-filter",
        default="",
        help=(
            "Optional comma-separated dataset:case_index list used for targeted "
            "preflight, for example context_0527_500:96,input_method_0612:71."
        ),
    )
    args = parser.parse_args()
    if args.fixed_bias is not None and (
        not math.isfinite(args.fixed_bias) or args.fixed_bias <= 0.0
    ):
        parser.error("--fixed-bias must be finite and positive")
    if args.ctc_temporal_gate and args.ctc_model_dir is None:
        parser.error("--ctc-model-dir is required with --ctc-temporal-gate")
    return args


def synchronized_time(enabled: bool, device: torch.device) -> float:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def evaluation_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["dataset"]),
        int(row.get("stage", 3)),
        int(row["case_index"]),
    )


def load_baseline_predictions(path: Path | None) -> dict[tuple[str, int, int], dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                raise ValueError(
                    f"baseline cache contains an error at {path}:{line_number}"
                )
            if "baseline_prediction" not in row:
                raise ValueError(
                    f"baseline cache has no baseline_prediction at {path}:{line_number}"
                )
            key = evaluation_key(row)
            if key in rows:
                raise ValueError(f"duplicate baseline cache key {key} in {path}")
            rows[key] = row
    if not rows:
        raise ValueError(f"baseline cache is empty: {path}")
    return rows


def cached_baseline_prediction(
    cache: dict[tuple[str, int, int], dict[str, Any]],
    *,
    key: tuple[str, int, int],
    model_dir: Path,
    max_new_tokens: int,
    audio_path: Path,
    hotwords: list[str],
    history_context: str,
    label: str,
) -> str:
    if key not in cache:
        raise ValueError(f"baseline cache is missing key {key}")
    row = cache[key]
    expected = {
        "model_dir": str(model_dir.resolve()),
        "max_new_tokens": int(max_new_tokens),
        "audio_path": str(audio_path.resolve()),
        "prompt_hotwords": hotwords,
        "history_context": history_context,
        "label": label,
    }
    observed = {
        "model_dir": str(Path(str(row.get("model_dir", ""))).resolve())
        if row.get("model_dir")
        else None,
        "max_new_tokens": row.get("max_new_tokens"),
        "audio_path": str(Path(str(row.get("audio_path", ""))).resolve())
        if row.get("audio_path")
        else None,
        "prompt_hotwords": row.get("prompt_hotwords"),
        "history_context": str(row.get("history_context") or ""),
        "label": str(row.get("label") or ""),
    }
    mismatches = [name for name in expected if observed[name] != expected[name]]
    if mismatches:
        details = ", ".join(
            f"{name}: cached={observed[name]!r} expected={expected[name]!r}"
            for name in mismatches
        )
        raise ValueError(f"baseline cache mismatch for {key}: {details}")
    return str(row["baseline_prediction"])


def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=2)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Nano adapter evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


class HiddenCapture:
    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def hook(self, _module, _inputs, output) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        self.value = hidden[:, -1, :].detach().float()


class AudioCapture:
    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def hook(self, _module, _inputs, output) -> None:
        frames, lengths = output
        length = int(lengths[0].item())
        self.value = frames[0, :length].detach().float()


_DIGIT_TO_SPOKEN_HAN = str.maketrans(
    {
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }
)
_SPOKEN_HAN_NUMERALS = frozenset("零〇一二两三四五六七八九十百千万亿")


def numeric_orthography_conflict(candidate: str, observed: str) -> bool:
    """Return true when bias would switch Arabic and spoken-Han numerals."""

    candidate_key = unicodedata.normalize("NFKC", candidate)
    observed_key = unicodedata.normalize("NFKC", observed)
    candidate_has_digit = any(value.isdigit() for value in candidate_key)
    observed_has_digit = any(value.isdigit() for value in observed_key)
    candidate_has_han = any(
        value in _SPOKEN_HAN_NUMERALS for value in candidate_key
    )
    observed_has_han = any(
        value in _SPOKEN_HAN_NUMERALS for value in observed_key
    )
    return bool(
        (candidate_has_digit and observed_has_han)
        or (observed_has_digit and candidate_has_han)
    )


def is_boundary_or_control_token(
    token_id: int,
    token_text: str,
    special_token_ids: set[int] | frozenset[int],
) -> bool:
    """Return whether Base is asking to emit formatting or end the utterance."""

    if int(token_id) in special_token_ids:
        return True
    return bool(token_text) and not normalize_for_match(token_text)


def common_latin_prefix_length(left: str, right: str) -> int:
    left_key = "".join(value for value in left.lower() if "a" <= value <= "z")
    right_key = "".join(value for value in right.lower() if "a" <= value <= "z")
    length = 0
    for left_value, right_value in zip(left_key, right_key):
        if left_value != right_value:
            break
        length += 1
    return length


def candidate_is_strict_subtoken(candidate: str, base_token: str) -> bool:
    """Return whether replacing Base would discard adjacent token text."""

    candidate_key = normalize_for_match(candidate)
    base_key = normalize_for_match(base_token)
    return bool(
        candidate_key
        and base_key
        and candidate_key != base_key
        and candidate_key in base_key
    )


def phonetic_units(text: str) -> tuple[str, ...]:
    """Return stable toneless pronunciation units for one decoder token."""

    normalized = unicodedata.normalize("NFKC", text).translate(
        _DIGIT_TO_SPOKEN_HAN
    )
    return tuple(
        str(value).lower()
        for value in lazy_pinyin(
            normalize_for_match(normalized),
            style=Style.NORMAL,
            neutral_tone_with_five=False,
            errors=lambda value: list(value.lower()),
        )
    )


def phonetic_group_ids(texts: list[str]) -> list[int]:
    """Assign equal ids only to current tokens with equal pronunciation."""

    group_by_units: dict[tuple[str, ...], int] = {}
    output: list[int] = []
    for text in texts:
        units = phonetic_units(text)
        # Empty decoder pieces must not become one shared comparison group.
        key = units or (f"__empty_{len(output)}",)
        output.append(group_by_units.setdefault(key, len(group_by_units)))
    return output


def candidate_competition_group_ids(
    token_lists: list[list[int]],
    token_indices: list[int],
    current_texts: list[str],
) -> list[int]:
    """Group candidates by the Trie branch already consumed by the decoder.

    At a fresh start there is no consumed branch, so equivalent starting
    pronunciations form a group. Once a branch has started, its exact decoder
    token prefix remains the group key; this keeps Latin pairs such as
    ``Raven/Raving`` and ``V6Lite/V6Light`` comparable at their divergence.
    """

    group_by_key: dict[tuple[object, ...], int] = {}
    output: list[int] = []
    for position, (token_ids, token_index, token_text) in enumerate(
        zip(token_lists, token_indices, current_texts)
    ):
        if token_index > 0:
            key: tuple[object, ...] = (
                "prefix",
                *tuple(int(value) for value in token_ids[:token_index]),
            )
        else:
            units = phonetic_units(token_text)
            key = ("start", *(units or (f"__empty_{position}",)))
        output.append(group_by_key.setdefault(key, len(group_by_key)))
    return output


def phonetic_prefix_match(candidate: str, observed: str) -> bool:
    """Check whether two decoder tokens start with the same pronunciation.

    Arabic digits are first converted to their digit-by-digit spoken Han form.
    This keeps an untrusted hotword from changing an acoustically equivalent
    Base spelling such as ``3号`` into ``三号`` merely because it received bias.
    """

    candidate_units = phonetic_units(candidate)
    observed_units = phonetic_units(observed)
    if not candidate_units or not observed_units:
        return False
    common = min(len(candidate_units), len(observed_units))
    return candidate_units[:common] == observed_units[:common]


def ctc_alignment_scalar_features(
    alignment: OrderedAlignment | None,
) -> tuple[float, float]:
    """Convert raw CTC confidence into bounded, trainable scalar features.

    CTC path scores are multiplicative and span several orders of magnitude.
    Log scaling preserves the distinction between weak and strong alignments
    without allowing a single tiny probability to dominate the fusion MLP.
    Missing alignments map to zero evidence.
    """

    if alignment is None:
        return 0.0, 0.0

    return (
        _normalized_log_probability(float(alignment.score)),
        _normalized_log_probability(float(alignment.minimum_label_probability)),
    )


def _normalized_log_probability(value: float) -> float:
    value = min(max(float(value), 1e-8), 1.0)
    return min(max((math.log10(value) + 8.0) / 8.0, 0.0), 1.0)


def ctc_orthographic_scalar_features(
    alignment: OrderedAlignment | None,
) -> tuple[float, float]:
    """Encode character-level support retained by a phonetic CTC alignment."""

    if alignment is None:
        return 0.0, 0.0
    return (
        _normalized_log_probability(float(alignment.orthographic_share)),
        _normalized_log_probability(
            float(alignment.minimum_orthographic_share)
        ),
    )


def ctc_current_token_supported(
    tokenizer,
    token_ids: list[int] | tuple[int, ...],
    token_groups,
    token_index: int,
) -> bool:
    """Return whether CTC evidence genuinely covers this decoder token.

    Rare Han characters can span several byte-level decoder tokens. Leading
    byte pieces decode as empty/replacement text and inherit the mapped group
    of the final piece. Ordinary Latin tokens never inherit a later numeric or
    Chinese group from the same hotword.
    """

    if token_index < 0 or token_index >= len(token_groups):
        return False
    if token_groups[token_index]:
        return True
    raw_text = tokenizer.decode(
        [int(token_ids[token_index])], skip_special_tokens=True
    )
    if raw_text and any(character != "\ufffd" for character in raw_text):
        return False
    for later_index in range(token_index + 1, len(token_groups)):
        if token_groups[later_index]:
            return True
        later_text = tokenizer.decode(
            [int(token_ids[later_index])], skip_special_tokens=True
        )
        if later_text and any(character != "\ufffd" for character in later_text):
            return False
    return False


def load_audio_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if int(sample_rate) != 16_000:
        divisor = math.gcd(int(sample_rate), 16_000)
        audio = resample_poly(
            audio, 16_000 // divisor, int(sample_rate) // divisor
        ).astype(np.float32)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def complete_token_texts(tokenizer, token_ids: list[int]) -> list[str]:
    """Decode whole byte sequences while preserving each token's text span."""

    ids = [int(token_id) for token_id in token_ids]
    fallback = [
        tokenizer.decode([token_id], skip_special_tokens=True) for token_id in ids
    ]
    if not ids or not callable(tokenizer):
        return fallback
    try:
        complete_text = tokenizer.decode(ids, skip_special_tokens=True)
        encoded = tokenizer(
            complete_text, add_special_tokens=False, return_offsets_mapping=True
        )
        encoded_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    except (KeyError, TypeError, ValueError, NotImplementedError):
        return fallback
    if encoded_ids != ids or len(offsets) != len(ids):
        return fallback
    output: list[str] = []
    for index, (start, end) in enumerate(offsets):
        if start < 0 or end < start or end > len(complete_text):
            return fallback
        shared_with_next = index + 1 < len(offsets) and offsets[index + 1] == (
            start,
            end,
        )
        output.append("" if shared_with_next else complete_text[start:end])
    return output


class NanoAttentionBiasProcessor(LogitsProcessor):
    """Stateful adapter runtime with Base Top-K and CTC temporal gates."""

    def __init__(
        self,
        *,
        fusion: FrozenFeatureAttentionFusion,
        hidden_capture: HiddenCapture,
        audio_capture: AudioCapture,
        embed_tokens,
        tokenizer,
        hotwords: list[str],
        hotword_token_ids: list[list[int]],
        hotword_token_embeddings: list[torch.Tensor],
        hotword_means: torch.Tensor,
        candidate_cap: int,
        base_topk_gate: int,
        start_topk_gate: int = 0,
        ctc_temporal_gate: CTCTemporalGate | None,
        ctc_audio: CTCAudioPosterior | None,
        hotword_ctc_groups: list[list[Any]] | None,
        ctc_start_threshold: float,
        ctc_start_min_label_prob: float,
        ctc_lookback_seconds: float,
        ctc_start_min_lookahead_seconds: float,
        ctc_start_lookahead_per_label_seconds: float,
        ctc_start_max_lookahead_seconds: float,
        ctc_max_gap_seconds: float,
        ctc_continuation_lookahead_seconds: float,
        ctc_continuation_threshold: float = 0.0,
        ctc_continuation_min_label_prob: float = 0.0,
        ctc_cursor_lookahead_seconds: float,
        ctc_cursor_min_score: float,
        initial_history_ids: list[int] | None = None,
        trace_hotwords: list[str] | None = None,
        bias_scale: float = 1.0,
        ablate_rank_gate: bool = False,
        ablate_trie_order: bool = False,
        fixed_bias: float | None = None,
        grounded_prior_scores: list[float] | None = None,
        grounded_prior_ranks: list[int] | None = None,
        grounded_input_ranks: list[int] | None = None,
        grounded_candidate_eligible: list[bool] | None = None,
        grounded_global_alignments: list[OrderedAlignment | None] | None = None,
        grounded_text_anchors: list[tuple[int, int, float] | None] | None = None,
        grounded_prior_threshold: float = float("-inf"),
        grounded_prior_topk: int = 0,
        grounded_input_topk: int = 0,
        grounded_fusion_mode: str = "prior_only",
        grounded_learned_gate_threshold: float = 0.0,
        grounded_alignment_tolerance_seconds: float = 0.6,
        grounded_anchor_tolerance_chars: int = 3,
        grounded_bias_margin: float = 0.15,
        grounded_max_bias: float = 8.0,
        grounded_base_consistency_margin: float = -1.0,
        grounded_base_phonetic_start: bool = False,
        grounded_require_base_phonetic_start: bool = False,
        require_base_phonetic_start: bool = False,
        lock_started_hotword: bool = False,
        restart_min_gap_tokens: int = 0,
        restart_requires_natural_start: bool = False,
        preserve_base_candidate_path: bool = False,
        preserve_base_candidate_token: bool = False,
        rank_rescue_topk: int = 0,
        ctc_min_orthographic_share: float = 0.0,
        trusted_candidate_mask: list[bool] | None = None,
        block_untrusted_homophone_override: bool = False,
        require_trusted_homophone_override: bool = False,
        trusted_candidate_threshold: float = -1.0,
        preserve_base_boundary_token: bool = False,
        ctc_mismatch_min_orthographic_share: float = 0.0,
        ctc_start_max_lead_seconds: float = -1.0,
        ctc_acoustic_rescue_threshold: float = -1.0,
        ctc_homophone_min_orthographic_share: float = 0.0,
        ctc_natural_start_min_orthographic_share: float = 0.0,
        latin_prefix_rescue_threshold: float = -1.0,
        ctc_contrastive_margin_threshold: float = -2.0,
        natural_phonetic_rescue_threshold: float = -1.0,
        active_continuation_rescue_threshold: float = -1.0,
        enable_runtime_cache: bool = True,
    ) -> None:
        self.fusion = fusion
        self.hidden_capture = hidden_capture
        self.audio_capture = audio_capture
        self.embed_tokens = embed_tokens
        self.tokenizer = tokenizer
        self.hotwords = hotwords
        self.hotword_token_ids = hotword_token_ids
        self.hotword_token_texts = [
            self._expanded_complete_token_texts(token_ids)
            for token_ids in hotword_token_ids
        ]
        self.hotword_token_embeddings = hotword_token_embeddings
        self.hotword_means = hotword_means
        self.enable_runtime_cache = bool(enable_runtime_cache)
        self._projected_audio_features: torch.Tensor | None = None
        self._projected_hotword_means: torch.Tensor | None = None
        self._projected_token_embeddings: list[torch.Tensor] | None = None
        self._projected_prefix_embeddings: list[torch.Tensor] | None = None
        self.candidate_cap = max(1, int(candidate_cap))
        self.base_topk_gate = max(1, int(base_topk_gate))
        self.start_topk_gate = (
            max(1, int(start_topk_gate))
            if int(start_topk_gate) > 0
            else self.base_topk_gate
        )
        if not math.isfinite(bias_scale) or bias_scale <= 0:
            raise ValueError(f"bias_scale must be finite and positive, got {bias_scale}")
        self.bias_scale = float(bias_scale)
        self.ablate_rank_gate = bool(ablate_rank_gate)
        self.ablate_trie_order = bool(ablate_trie_order)
        if fixed_bias is not None and (
            not math.isfinite(fixed_bias) or fixed_bias <= 0.0
        ):
            raise ValueError(f"fixed_bias must be finite and positive, got {fixed_bias}")
        self.fixed_bias = float(fixed_bias) if fixed_bias is not None else None
        if grounded_prior_scores is not None and len(grounded_prior_scores) != len(hotwords):
            raise ValueError("grounded prior scores must align with hotwords")
        if grounded_prior_ranks is not None and len(grounded_prior_ranks) != len(hotwords):
            raise ValueError("grounded prior ranks must align with hotwords")
        if grounded_input_ranks is not None and len(grounded_input_ranks) != len(hotwords):
            raise ValueError("grounded input ranks must align with hotwords")
        if grounded_candidate_eligible is not None and len(grounded_candidate_eligible) != len(hotwords):
            raise ValueError("grounded candidate mask must align with hotwords")
        if grounded_global_alignments is not None and len(grounded_global_alignments) != len(hotwords):
            raise ValueError("grounded global alignments must align with hotwords")
        if grounded_text_anchors is not None and len(grounded_text_anchors) != len(hotwords):
            raise ValueError("grounded text anchors must align with hotwords")
        if trusted_candidate_mask is not None and len(trusted_candidate_mask) != len(hotwords):
            raise ValueError("trusted candidate mask must align with hotwords")
        self.grounded_prior_scores = (
            [float(value) for value in grounded_prior_scores]
            if grounded_prior_scores is not None
            else None
        )
        self.grounded_prior_ranks = (
            [int(value) for value in grounded_prior_ranks]
            if grounded_prior_ranks is not None
            else None
        )
        self.grounded_input_ranks = (
            [int(value) for value in grounded_input_ranks]
            if grounded_input_ranks is not None
            else None
        )
        self.grounded_candidate_eligible = (
            [bool(value) for value in grounded_candidate_eligible]
            if grounded_candidate_eligible is not None
            else None
        )
        self.grounded_global_alignments = grounded_global_alignments
        self.grounded_text_anchors = grounded_text_anchors
        self.grounded_prior_threshold = float(grounded_prior_threshold)
        self.grounded_prior_topk = max(0, int(grounded_prior_topk))
        self.grounded_input_topk = max(0, int(grounded_input_topk))
        if grounded_fusion_mode not in {
            "prior_only",
            "binary_or_prior",
            "prior_and_learned",
        }:
            raise ValueError(
                "grounded_fusion_mode must be prior_only, binary_or_prior, "
                "or prior_and_learned"
            )
        self.grounded_fusion_mode = grounded_fusion_mode
        self.grounded_learned_gate_threshold = float(
            grounded_learned_gate_threshold
        )
        self.grounded_alignment_tolerance_seconds = max(
            0.0, float(grounded_alignment_tolerance_seconds)
        )
        self.grounded_anchor_tolerance_chars = max(
            0, int(grounded_anchor_tolerance_chars)
        )
        self.grounded_bias_margin = max(0.0, float(grounded_bias_margin))
        self.grounded_max_bias = max(0.0, float(grounded_max_bias))
        self.grounded_base_consistency_margin = float(
            grounded_base_consistency_margin
        )
        self.grounded_base_phonetic_start = bool(grounded_base_phonetic_start)
        self.grounded_require_base_phonetic_start = bool(
            grounded_require_base_phonetic_start
        )
        self.require_base_phonetic_start = bool(require_base_phonetic_start)
        self.lock_started_hotword = bool(lock_started_hotword)
        self.restart_min_gap_tokens = max(0, int(restart_min_gap_tokens))
        self.restart_requires_natural_start = bool(
            restart_requires_natural_start
        )
        self.preserve_base_candidate_path = bool(preserve_base_candidate_path)
        self.preserve_base_candidate_token = bool(
            preserve_base_candidate_token
        )
        self.rank_rescue_topk = max(0, int(rank_rescue_topk))
        self.trusted_candidate_mask = (
            [bool(value) for value in trusted_candidate_mask]
            if trusted_candidate_mask is not None
            else [False] * len(hotwords)
        )
        self.block_untrusted_homophone_override = bool(
            block_untrusted_homophone_override
        )
        self.require_trusted_homophone_override = bool(
            require_trusted_homophone_override
        )
        if trusted_candidate_threshold > 1.0:
            raise ValueError("trusted candidate threshold must not exceed one")
        self.trusted_candidate_threshold = float(trusted_candidate_threshold)
        self.preserve_base_boundary_token = bool(preserve_base_boundary_token)
        self.ctc_mismatch_min_orthographic_share = min(
            1.0, max(0.0, float(ctc_mismatch_min_orthographic_share))
        )
        self.ctc_start_max_lead_seconds = float(ctc_start_max_lead_seconds)
        self.ctc_acoustic_rescue_threshold = float(
            ctc_acoustic_rescue_threshold
        )
        self.ctc_homophone_min_orthographic_share = min(
            1.0, max(0.0, float(ctc_homophone_min_orthographic_share))
        )
        self.ctc_natural_start_min_orthographic_share = min(
            1.0,
            max(0.0, float(ctc_natural_start_min_orthographic_share)),
        )
        self.latin_prefix_rescue_threshold = float(
            latin_prefix_rescue_threshold
        )
        self.natural_phonetic_rescue_threshold = float(
            natural_phonetic_rescue_threshold
        )
        self.active_continuation_rescue_threshold = float(
            active_continuation_rescue_threshold
        )
        if not -2.0 <= float(ctc_contrastive_margin_threshold) <= 1.0:
            raise ValueError(
                "ctc contrastive margin threshold must be in [-2, 1]"
            )
        self.ctc_contrastive_margin_threshold = float(
            ctc_contrastive_margin_threshold
        )
        self.special_token_ids = frozenset(
            int(value) for value in getattr(tokenizer, "all_special_ids", ())
        )
        self.ctc_min_orthographic_share = min(
            1.0, max(0.0, float(ctc_min_orthographic_share))
        )
        self.ctc_temporal_gate = ctc_temporal_gate
        self.ctc_audio = ctc_audio
        self.hotword_ctc_groups = hotword_ctc_groups
        self.ctc_start_threshold = float(ctc_start_threshold)
        self.ctc_start_min_label_prob = float(ctc_start_min_label_prob)
        self.ctc_lookback_seconds = float(ctc_lookback_seconds)
        self.ctc_start_min_lookahead_seconds = float(
            ctc_start_min_lookahead_seconds
        )
        self.ctc_start_lookahead_per_label_seconds = float(
            ctc_start_lookahead_per_label_seconds
        )
        self.ctc_start_max_lookahead_seconds = float(
            ctc_start_max_lookahead_seconds
        )
        self.ctc_max_gap_seconds = float(ctc_max_gap_seconds)
        self.ctc_continuation_lookahead_seconds = float(
            ctc_continuation_lookahead_seconds
        )
        self.ctc_continuation_threshold = max(
            0.0, float(ctc_continuation_threshold)
        )
        self.ctc_continuation_min_label_prob = max(
            0.0, float(ctc_continuation_min_label_prob)
        )
        self.ctc_cursor_lookahead_seconds = float(ctc_cursor_lookahead_seconds)
        self.ctc_cursor_min_score = float(ctc_cursor_min_score)
        self.initial_history_ids = [int(value) for value in (initial_history_ids or [])]
        self.depths = [0] * len(hotword_token_ids)
        self.active_alignments: list[OrderedAlignment | None] = [
            None for _ in hotword_token_ids
        ]
        self.pending_start_alignments: dict[int, OrderedAlignment] = {}
        self.pending_base_anchored_starts: set[int] = set()
        self.active_base_anchored_starts: set[int] = set()
        self.committed_word_index: int | None = None
        self.pending_commit_word_index: int | None = None
        self.pending_unmappable_starts: set[int] = set()
        self.unmappable_started: set[int] = set()
        self.completed_end_frames = [-1 for _ in hotword_token_ids]
        self.completed_generated_positions = [-1 for _ in hotword_token_ids]
        self.generated: list[int] = []
        self.observed_sequence_length: int | None = None
        self.calls = 0
        self.applied_rows = 0
        self.bias_sum = 0.0
        self.bias_max = 0.0
        self.topk_rejected_rows = 0
        self.ctc_rejected_rows = 0
        self.ctc_unmappable_passthrough_rows = 0
        self.ctc_unmappable_restart_rejected_rows = 0
        self.ctc_reused_span_rejected_rows = 0
        self.restart_cooldown_rejected_rows = 0
        self.restart_natural_start_rows = 0
        self.natural_start_commit_deferred_rows = 0
        self._positive_bias_events: list[dict[str, Any]] = []
        trace_keys = {
            normalize_for_match(word)
            for word in (trace_hotwords or [])
            if normalize_for_match(word)
        }
        self.trace_word_indices = {
            word_index
            for word_index, word in enumerate(self.hotwords)
            if normalize_for_match(word) in trace_keys
        }
        self._target_gate_stats: dict[int, dict[str, Any]] = {
            word_index: {
                "hotword": self.hotwords[word_index],
                "token_ids": [int(value) for value in self.hotword_token_ids[word_index]],
                "observations": 0,
                "candidate_selected_steps": 0,
                "candidate_cap_rejected_steps": 0,
                "base_topk_allowed_steps": 0,
                "base_topk_rejected_steps": 0,
                "ctc_allowed_steps": 0,
                "ctc_rejected_steps": 0,
                "applied_steps": 0,
                "positive_bias_steps": 0,
                "binary_emit_steps": 0,
                "binary_reject_steps": 0,
                "adjusted_top1_steps": 0,
                "minimum_base_rank": None,
                "minimum_adjusted_rank": None,
                "maximum_bias": None,
                "maximum_actual_token_bias": None,
                "minimum_base_gap": None,
                "maximum_bias_minus_gap": None,
                "ctc_reasons": {},
                "tokens": {},
                "events": [],
            }
            for word_index in self.trace_word_indices
        }

    def _expanded_complete_token_texts(
        self, token_ids: list[int]
    ) -> list[str]:
        """Map every decoder token to its complete Unicode acoustic unit."""

        texts = complete_token_texts(self.tokenizer, token_ids)
        mapped = [index for index, text in enumerate(texts) if text]
        if not mapped:
            return [
                self.tokenizer.decode(
                    [int(token_id)], skip_special_tokens=True
                )
                for token_id in token_ids
            ]
        expanded: list[str] = []
        for token_index, text in enumerate(texts):
            if text:
                expanded.append(text)
                continue
            nearest = min(
                mapped,
                key=lambda index: (
                    abs(index - token_index),
                    index < token_index,
                ),
            )
            expanded.append(texts[nearest])
        return expanded

    def _candidate_token_text(self, word_index: int, token_index: int) -> str:
        token_texts = getattr(self, "hotword_token_texts", None)
        if token_texts is not None:
            return token_texts[word_index][token_index]
        return self.tokenizer.decode(
            [int(self.hotword_token_ids[word_index][token_index])],
            skip_special_tokens=True,
        )

    @staticmethod
    def _update_minimum(stats: dict[str, Any], key: str, value: float | int) -> None:
        previous = stats[key]
        stats[key] = value if previous is None else min(previous, value)

    @staticmethod
    def _update_maximum(stats: dict[str, Any], key: str, value: float | int) -> None:
        previous = stats[key]
        stats[key] = value if previous is None else max(previous, value)

    def _trace_token_stats(
        self,
        word_index: int,
        token_index: int,
        token_id: int,
    ) -> dict[str, Any]:
        stats = self._target_gate_stats[word_index]
        key = str(token_index)
        if key not in stats["tokens"]:
            stats["tokens"][key] = {
                "token_index": token_index,
                "token_id": token_id,
                "token_text": self._candidate_token_text(
                    word_index, token_index
                ),
                "observations": 0,
                "candidate_selected_steps": 0,
                "base_topk_allowed_steps": 0,
                "ctc_allowed_steps": 0,
                "applied_steps": 0,
                "positive_bias_steps": 0,
                "binary_emit_steps": 0,
                "binary_reject_steps": 0,
                "adjusted_top1_steps": 0,
                "minimum_base_rank": None,
                "minimum_adjusted_rank": None,
                "maximum_bias": None,
                "maximum_actual_token_bias": None,
                "minimum_base_gap": None,
                "maximum_bias_minus_gap": None,
                "ctc_reasons": {},
            }
        return stats["tokens"][key]

    def target_gate_stats(self) -> list[dict[str, Any]]:
        return [
            self._target_gate_stats[word_index]
            for word_index in sorted(self._target_gate_stats)
        ]

    def _ensure_static_projection_cache(self) -> None:
        if not self.enable_runtime_cache:
            return
        if self._projected_hotword_means is None:
            prefix_tables: list[torch.Tensor] = []
            for embeddings in self.hotword_token_embeddings:
                prefixes = torch.zeros_like(embeddings)
                if embeddings.shape[0] > 1:
                    divisors = torch.arange(
                        1,
                        embeddings.shape[0],
                        dtype=embeddings.dtype,
                        device=embeddings.device,
                    )[:, None]
                    prefixes[1:] = embeddings.cumsum(dim=0)[:-1] / divisors
                prefix_tables.append(prefixes)
            flat_tokens = torch.cat(self.hotword_token_embeddings, dim=0)
            flat_prefixes = torch.cat(prefix_tables, dim=0)
            lengths = [embeddings.shape[0] for embeddings in self.hotword_token_embeddings]
            device_type = self.hotword_means.device.type
            with torch.no_grad(), torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
            ):
                self._projected_hotword_means = self.fusion.hotword_projection(
                    self.hotword_means
                )
                projected_tokens = self.fusion.current_token_projection(flat_tokens)
                projected_prefixes = self.fusion.prefix_projection(flat_prefixes)
            self._projected_token_embeddings = list(projected_tokens.split(lengths))
            self._projected_prefix_embeddings = list(projected_prefixes.split(lengths))
        if self._projected_audio_features is None:
            if self.audio_capture.value is None:
                raise RuntimeError("audio feature hook was not ready for projection cache")
            device_type = self.audio_capture.value.device.type
            with torch.no_grad(), torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
            ):
                self._projected_audio_features = self.fusion.audio_projection(
                    self.audio_capture.value
                )

    def positive_bias_events(self) -> list[dict[str, Any]]:
        return self._positive_bias_events

    def _current_index(self, word_index: int) -> int:
        depth = self.depths[word_index]
        ids = self.hotword_token_ids[word_index]
        return depth if 0 < depth < len(ids) else 0

    def _shared_start_branch_count(self, token_id: int) -> int:
        """Count unresolved multi-token hotwords that start with ``token_id``."""
        return sum(
            1
            for word_index, ids in enumerate(self.hotword_token_ids)
            if self.depths[word_index] == 0
            and len(ids) > 1
            and int(ids[0]) == int(token_id)
        )

    def _restart_in_cooldown(
        self,
        word_index: int,
        *,
        generated_count: int | None = None,
    ) -> bool:
        minimum_gap = int(getattr(self, "restart_min_gap_tokens", 0))
        positions = getattr(self, "completed_generated_positions", [])
        if minimum_gap <= 0 or word_index >= len(positions):
            return False
        completed_position = int(positions[word_index])
        if completed_position < 0:
            return False
        count = len(self.generated) if generated_count is None else generated_count
        tokens_after_completion = count - completed_position - 1
        return tokens_after_completion < minimum_gap

    def _observe(self, target_id: int) -> None:
        pending_commit = self.pending_commit_word_index
        self.pending_commit_word_index = None
        previous_commit = self.committed_word_index
        old_depths = list(self.depths)
        generated_before = len(self.generated)
        restart_blocked = [
            bool(
                depth == 0
                and ids
                and target_id == ids[0]
                and self._restart_in_cooldown(
                    word_index,
                    generated_count=generated_before,
                )
            )
            for word_index, (depth, ids) in enumerate(
                zip(old_depths, self.hotword_token_ids)
            )
        ]
        aligned_ends: list[int] = []
        matched_alignments: dict[int, OrderedAlignment] = {}
        for word_index, ids in enumerate(self.hotword_token_ids):
            depth = old_depths[word_index]
            alignment: OrderedAlignment | None = None
            if depth == 0 and ids and target_id == ids[0]:
                alignment = self.pending_start_alignments.get(word_index)
                span_index = 0
            elif 0 < depth < len(ids) and target_id == ids[depth]:
                alignment = self.active_alignments[word_index]
                span_index = depth
            else:
                span_index = -1
            if (
                alignment is not None
                and 0 <= span_index < len(alignment.group_spans)
            ):
                current_span = alignment.group_spans[span_index]
                next_uses_same_character = bool(
                    span_index + 1 < len(alignment.group_spans)
                    and alignment.group_spans[span_index + 1] == current_span
                )
                # A rare character may occupy several byte-level LLM tokens.
                # Consume its CTC frame only after the final byte token.
                if not next_uses_same_character:
                    aligned_ends.append(current_span[1])
                matched_alignments[word_index] = alignment

        if self.ctc_audio is not None and aligned_ends:
            self.ctc_audio.cursor_frame = max(
                self.ctc_audio.cursor_frame,
                min(aligned_ends) + 1,
            )
        elif self.ctc_temporal_gate is not None and self.ctc_audio is not None:
            token_text = self.tokenizer.decode(
                [int(target_id)], skip_special_tokens=True
            )
            self.ctc_temporal_gate.advance_cursor(
                self.ctc_audio,
                token_text,
                lookback_seconds=self.ctc_lookback_seconds,
                lookahead_seconds=self.ctc_cursor_lookahead_seconds,
                max_gap_seconds=self.ctc_max_gap_seconds,
                minimum_score=self.ctc_cursor_min_score,
            )

        self.generated.append(int(target_id))
        for word_index, ids in enumerate(self.hotword_token_ids):
            depth = old_depths[word_index]
            if (
                depth == 0
                and ids
                and target_id == ids[0]
                and word_index in self.pending_unmappable_starts
            ):
                self.unmappable_started.add(word_index)
            completed = bool(
                ids
                and target_id == ids[-1]
                and (
                    (len(ids) == 1 and depth == 0)
                    or (0 < depth == len(ids) - 1)
                )
            )
            if completed:
                self.completed_generated_positions[word_index] = len(self.generated) - 1
                alignment = matched_alignments.get(word_index)
                if alignment is not None and alignment.group_spans:
                    self.completed_end_frames[word_index] = max(
                        self.completed_end_frames[word_index],
                        alignment.group_spans[-1][1],
                    )
            if 0 < depth < len(ids):
                if target_id == ids[depth]:
                    next_depth = depth + 1
                    self.depths[word_index] = 0 if next_depth == len(ids) else next_depth
                elif len(ids) > 1 and target_id == ids[0]:
                    self.depths[word_index] = 1
                else:
                    self.depths[word_index] = 0
            elif len(ids) > 1 and target_id == ids[0]:
                self.depths[word_index] = 0 if restart_blocked[word_index] else 1

        for word_index, (old_depth, new_depth) in enumerate(
            zip(old_depths, self.depths)
        ):
            if old_depth == 0 and new_depth == 1:
                self.active_alignments[word_index] = (
                    self.pending_start_alignments.get(word_index)
                )
                if word_index in self.pending_base_anchored_starts:
                    self.active_base_anchored_starts.add(word_index)
                else:
                    self.active_base_anchored_starts.discard(word_index)
            elif new_depth == 0:
                self.active_alignments[word_index] = None
                self.active_base_anchored_starts.discard(word_index)

        if previous_commit is not None:
            sequence = self.hotword_token_ids[previous_commit]
            depth = old_depths[previous_commit]
            matched = 0 < depth < len(sequence) and target_id == sequence[depth]
            completed = matched and depth == len(sequence) - 1
            if not matched or completed:
                self.committed_word_index = None
        elif (
            pending_commit is not None
            and self.hotword_token_ids[pending_commit]
        ):
            sequence = self.hotword_token_ids[pending_commit]
            previous_depth = old_depths[pending_commit]
            expected_index = 0 if previous_depth == 0 else previous_depth
            matched = (
                expected_index < len(sequence)
                and target_id == sequence[expected_index]
            )
            completed = matched and expected_index == len(sequence) - 1
            if matched and not completed:
                # A positive intervention at either the start or a later
                # continuation identifies one complete Trie branch. Competing
                # branches must not inherit that committed prefix.
                self.committed_word_index = pending_commit
                for word_index in range(len(self.depths)):
                    if word_index == pending_commit:
                        continue
                    self.depths[word_index] = 0
                    self.active_alignments[word_index] = None
                    self.active_base_anchored_starts.discard(word_index)
        self.pending_start_alignments = {}
        self.pending_base_anchored_starts = set()
        self.pending_unmappable_starts = set()

    def _orthographic_ctc_allowed(self, alignment: OrderedAlignment) -> bool:
        return (
            float(alignment.orthographic_share)
            >= float(getattr(self, "ctc_min_orthographic_share", 0.0))
        )

    def _ctc_start_is_near_cursor(self, alignment: OrderedAlignment) -> bool:
        maximum_lead_seconds = float(
            getattr(self, "ctc_start_max_lead_seconds", -1.0)
        )
        if maximum_lead_seconds < 0.0 or self.ctc_audio is None:
            return True
        frame_seconds = max(float(self.ctc_audio.frame_seconds), 1e-6)
        maximum_lead_frames = round(
            maximum_lead_seconds / frame_seconds
        )
        return int(alignment.label_frames[0]) <= (
            int(self.ctc_audio.cursor_frame) + maximum_lead_frames
        )

    def _base_path_ctc_is_credible(
        self,
        ctc_result: tuple[bool, str, OrderedAlignment | None] | None,
    ) -> bool:
        if ctc_result is None:
            return True
        allowed, _reason, alignment = ctc_result
        if not allowed:
            return False
        minimum_share = float(
            getattr(
                self,
                "ctc_natural_start_min_orthographic_share",
                0.0,
            )
        )
        return bool(
            alignment is None
            or minimum_share <= 0.0
            or float(alignment.orthographic_share) >= minimum_share
        )

    def _ctc_eligible(
        self,
        word_index: int,
        token_index: int,
        base_top1_text: str = "",
        base_top1_id: int | None = None,
    ) -> tuple[bool, str, OrderedAlignment | None]:
        if token_index == 0 and getattr(self, "require_base_phonetic_start", False):
            first_token_id = int(self.hotword_token_ids[word_index][0])
            exact_token_match = (
                base_top1_id is not None and first_token_id == int(base_top1_id)
            )
            if not exact_token_match and not phonetic_prefix_match(
                self.hotwords[word_index], base_top1_text
            ):
                return False, "base_phonetic_start_required", None
        if self.ctc_temporal_gate is None:
            return True, "disabled", None
        if self.ctc_audio is None or self.hotword_ctc_groups is None:
            return False, "missing_audio", None
        if token_index == 0 and self._restart_in_cooldown(word_index):
            self.restart_cooldown_rejected_rows += 1
            return False, "restart_cooldown", None
        token_groups = self.hotword_ctc_groups[word_index]
        if not token_groups or not any(token_groups):
            # The Chinese CTC vocabulary cannot judge pure English or symbols.
            # Allow one biased start and its continuation, but never let the
            # same acoustically-unverifiable path restart and self-excite.
            if token_index == 0 and word_index in self.unmappable_started:
                self.ctc_unmappable_restart_rejected_rows += 1
                return False, "unmappable_restart", None
            if token_index == 0:
                self.pending_unmappable_starts.add(word_index)
            self.ctc_unmappable_passthrough_rows += 1
            return True, "unmappable_passthrough", None

        if token_index == 0:
            if self.grounded_text_anchors is not None:
                anchor = self.grounded_text_anchors[word_index]
                if anchor is None:
                    return False, "missing_text_anchor", None
                generated_text = self.tokenizer.decode(
                    self.generated, skip_special_tokens=True
                )
                generated_position = len(normalize_for_match(generated_text))
                start, _end, _distance = anchor
                if abs(generated_position - int(start)) > self.grounded_anchor_tolerance_chars:
                    return False, "text_anchor_time", None
            if self.grounded_global_alignments is not None:
                completed_end = self.completed_end_frames[word_index]
                if completed_end >= 0:
                    frame_seconds = max(self.ctc_audio.frame_seconds, 1e-6)
                    lookback_frames = round(
                        self.ctc_lookback_seconds / frame_seconds
                    )
                    strict_cursor = max(
                        self.ctc_audio.cursor_frame,
                        completed_end + 1 + lookback_frames,
                    )
                    if hasattr(self.ctc_audio, "with_cursor"):
                        alignment_audio = self.ctc_audio.with_cursor(strict_cursor)
                    else:
                        alignment_audio = CTCAudioPosterior(
                            probabilities=self.ctc_audio.probabilities,
                            frame_seconds=self.ctc_audio.frame_seconds,
                            cursor_frame=strict_cursor,
                            initial_cursor_frame=self.ctc_audio.initial_cursor_frame,
                        )
                    alignment = self.ctc_temporal_gate.align_hotword_near_cursor(
                        alignment_audio,
                        token_groups,
                        lookback_seconds=self.ctc_lookback_seconds,
                        minimum_lookahead_seconds=(
                            self.ctc_start_min_lookahead_seconds
                        ),
                        lookahead_per_label_seconds=(
                            self.ctc_start_lookahead_per_label_seconds
                        ),
                        maximum_lookahead_seconds=(
                            self.ctc_start_max_lookahead_seconds
                        ),
                        max_gap_seconds=self.ctc_max_gap_seconds,
                    )
                    if alignment is None:
                        return False, "no_new_occurrence", None
                    if alignment.label_frames[0] <= completed_end:
                        self.ctc_reused_span_rejected_rows += 1
                        return False, "completed_span_reuse", alignment
                    if (
                        alignment.score < self.ctc_start_threshold
                        or alignment.minimum_label_probability
                        < self.ctc_start_min_label_prob
                    ):
                        return False, "new_occurrence_score", alignment
                    if not self._orthographic_ctc_allowed(alignment):
                        return False, "new_occurrence_orthographic", alignment
                    if not self._ctc_start_is_near_cursor(alignment):
                        return False, "new_occurrence_ahead_of_cursor", alignment
                    self.pending_start_alignments[word_index] = alignment
                    return True, "new_occurrence_match", alignment

                alignment = self.grounded_global_alignments[word_index]
                if alignment is None:
                    return False, "missing_global_alignment", None
                frame_seconds = max(self.ctc_audio.frame_seconds, 1e-6)
                tolerance = round(
                    self.grounded_alignment_tolerance_seconds / frame_seconds
                )
                start_frame = int(alignment.group_spans[0][0])
                cursor_matches = (
                    abs(int(self.ctc_audio.cursor_frame) - start_frame) <= tolerance
                )
                candidate_text = self._candidate_token_text(word_index, 0)
                base_phonetic_matches = (
                    (
                        self.grounded_base_phonetic_start
                        or self.grounded_require_base_phonetic_start
                    )
                    and phonetic_prefix_match(candidate_text, base_top1_text)
                )
                if (
                    self.grounded_require_base_phonetic_start
                    and not base_phonetic_matches
                ):
                    return False, "global_alignment_base_phonetic_required", alignment
                if not cursor_matches and not base_phonetic_matches:
                    return False, "global_alignment_time", alignment
                if (
                    alignment.score < self.ctc_start_threshold
                    or alignment.minimum_label_probability
                    < self.ctc_start_min_label_prob
                ):
                    return False, "global_alignment_score", alignment
                if not self._orthographic_ctc_allowed(alignment):
                    return False, "global_alignment_orthographic", alignment
                self.pending_start_alignments[word_index] = alignment
                if base_phonetic_matches:
                    self.pending_base_anchored_starts.add(word_index)
                    return True, "global_alignment_base_phonetic", alignment
                return True, "global_alignment_match", alignment
            alignment_audio = self.ctc_audio
            completed_end = self.completed_end_frames[word_index]
            if completed_end >= 0:
                frame_seconds = max(self.ctc_audio.frame_seconds, 1e-6)
                lookback_frames = round(
                    self.ctc_lookback_seconds / frame_seconds
                )
                # Move only the alignment query cursor. The search window then
                # starts strictly after the previously consumed occurrence.
                strict_cursor = max(
                    self.ctc_audio.cursor_frame,
                    completed_end + 1 + lookback_frames,
                )
                if hasattr(self.ctc_audio, "with_cursor"):
                    alignment_audio = self.ctc_audio.with_cursor(strict_cursor)
                else:
                    alignment_audio = CTCAudioPosterior(
                        probabilities=self.ctc_audio.probabilities,
                        frame_seconds=self.ctc_audio.frame_seconds,
                        cursor_frame=strict_cursor,
                        initial_cursor_frame=self.ctc_audio.initial_cursor_frame,
                    )
            alignment = self.ctc_temporal_gate.align_hotword_near_cursor(
                alignment_audio,
                token_groups,
                lookback_seconds=self.ctc_lookback_seconds,
                minimum_lookahead_seconds=self.ctc_start_min_lookahead_seconds,
                lookahead_per_label_seconds=(
                    self.ctc_start_lookahead_per_label_seconds
                ),
                maximum_lookahead_seconds=self.ctc_start_max_lookahead_seconds,
                max_gap_seconds=self.ctc_max_gap_seconds,
            )
            if alignment is not None and completed_end >= 0:
                if alignment.label_frames[0] <= completed_end:
                    self.ctc_reused_span_rejected_rows += 1
                    return False, "completed_span_reuse", alignment
            natural_base_start = bool(
                int(self.hotword_token_ids[word_index][0])
                == int(base_top1_id)
                and alignment is not None
                and self._ctc_start_is_near_cursor(alignment)
                and (
                    alignment.score < self.ctc_start_threshold
                    or alignment.minimum_label_probability
                    < self.ctc_start_min_label_prob
                )
            )
            if natural_base_start:
                # Keeping a Trie branch alive cannot change this step because
                # Base already selected the same token. Later spelling changes
                # still require their own phonetic/semantic decision.
                self.pending_start_alignments[word_index] = alignment
                self.pending_base_anchored_starts.add(word_index)
                return True, "natural_base_start", alignment
            if (
                alignment is None
                or alignment.score < self.ctc_start_threshold
                or alignment.minimum_label_probability
                < self.ctc_start_min_label_prob
            ):
                return False, "start_score", alignment
            if not self._orthographic_ctc_allowed(alignment):
                return False, "start_orthographic", alignment
            if not self._ctc_start_is_near_cursor(alignment):
                return False, "start_ahead_of_cursor", alignment
            self.pending_start_alignments[word_index] = alignment
            if int(self.hotword_token_ids[word_index][0]) == int(base_top1_id):
                self.pending_base_anchored_starts.add(word_index)
            return True, "start_match", alignment

        alignment = self.active_alignments[word_index]
        if alignment is None or token_index >= len(alignment.group_spans):
            return False, "no_active_alignment", alignment
        if word_index in self.active_base_anchored_starts:
            minimum_share = float(
                getattr(
                    self,
                    "ctc_natural_start_min_orthographic_share",
                    0.0,
                )
            )
            if (
                minimum_share > 0.0
                and float(alignment.orthographic_share) < minimum_share
            ):
                return False, "base_anchored_orthographic", alignment
            return True, "base_anchored_continuation", alignment
        frame_seconds = max(self.ctc_audio.frame_seconds, 1e-6)
        left = self.ctc_audio.cursor_frame - round(
            self.ctc_lookback_seconds / frame_seconds
        )
        right = self.ctc_audio.cursor_frame + round(
            self.ctc_continuation_lookahead_seconds / frame_seconds
        )
        span_start, span_end = alignment.group_spans[token_index]
        if span_end < left or span_start > right:
            return False, "continuation_time", alignment
        if (
            alignment.score < self.ctc_continuation_threshold
            or alignment.minimum_label_probability
            < self.ctc_continuation_min_label_prob
        ):
            return False, "continuation_score", alignment
        return True, "continuation_match", alignment

    def _candidate_ctc_eligible(
        self,
        word_index: int,
        token_index: int,
        base_top1_text: str = "",
        base_top1_id: int | None = None,
    ) -> tuple[bool, str, OrderedAlignment | None]:
        if not self.ablate_trie_order or token_index == self._current_index(word_index):
            return self._ctc_eligible(
                word_index, token_index, base_top1_text, base_top1_id
            )
        if self.ctc_temporal_gate is None:
            return True, "disabled", None
        if self.ctc_audio is None or self.hotword_ctc_groups is None:
            return False, "missing_audio", None
        token_groups = self.hotword_ctc_groups[word_index]
        if not token_groups or token_index >= len(token_groups) or not token_groups[token_index]:
            return True, "unmappable_passthrough", None

        # An out-of-order token has no active whole-word alignment.  Check
        # that token's own CTC labels at the current cursor.
        alignment = self.ctc_temporal_gate.align_hotword_near_cursor(
            self.ctc_audio,
            [token_groups[token_index]],
            lookback_seconds=self.ctc_lookback_seconds,
            minimum_lookahead_seconds=self.ctc_start_min_lookahead_seconds,
            lookahead_per_label_seconds=self.ctc_start_lookahead_per_label_seconds,
            maximum_lookahead_seconds=self.ctc_start_max_lookahead_seconds,
            max_gap_seconds=self.ctc_max_gap_seconds,
        )
        if alignment is None:
            return False, "unordered_no_alignment", None
        if (
            alignment.score < self.ctc_start_threshold
            or alignment.minimum_label_probability < self.ctc_start_min_label_prob
        ):
            return False, "unordered_score", alignment
        frame_seconds = max(self.ctc_audio.frame_seconds, 1e-6)
        left = self.ctc_audio.cursor_frame - round(
            self.ctc_lookback_seconds / frame_seconds
        )
        right = self.ctc_audio.cursor_frame + round(
            self.ctc_continuation_lookahead_seconds / frame_seconds
        )
        span_start, span_end = alignment.group_spans[0]
        if span_end < left or span_start > right:
            return False, "unordered_time", alignment
        if not self._orthographic_ctc_allowed(alignment):
            return False, "unordered_orthographic", alignment
        # Downstream CTC features address spans by the original hotword-token
        # index.  Preserve that indexing for this one-token alignment.
        alignment = OrderedAlignment(
            score=alignment.score,
            minimum_label_probability=alignment.minimum_label_probability,
            label_frames=alignment.label_frames,
            group_spans=tuple(alignment.group_spans[0] for _ in token_groups),
            orthographic_share=alignment.orthographic_share,
            minimum_orthographic_share=alignment.minimum_orthographic_share,
        )
        return True, "unordered_token_match", alignment

    def _ctc_contrastive_support(
        self,
        *,
        word_index: int,
        token_index: int,
        token_id: int,
        base_top1_id: int,
        base_top1_text: str,
        ctc_result: tuple[bool, str, OrderedAlignment | None],
    ) -> tuple[float, float, float] | None:
        method = getattr(
            self.ctc_temporal_gate,
            "contrastive_token_support",
            None,
        )
        candidate_text = self._candidate_token_text(word_index, token_index)
        if (
            not callable(method)
            or not ctc_result[0]
            or ctc_result[2] is None
            or int(token_id) == int(base_top1_id)
            or normalize_for_match(candidate_text)
            == normalize_for_match(base_top1_text)
            or is_boundary_or_control_token(
                base_top1_id,
                base_top1_text,
                self.special_token_ids,
            )
        ):
            return None
        return method(
            self.ctc_audio,
            candidate_text,
            base_top1_text,
            ctc_result[2],
            token_index,
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if scores.shape[0] != 1:
            raise RuntimeError("Nano adapter evaluator requires batch=1")
        sequence = input_ids[0].tolist()
        if self.observed_sequence_length is None:
            self.observed_sequence_length = len(sequence)
        else:
            for token_id in sequence[self.observed_sequence_length :]:
                self._observe(int(token_id))
            self.observed_sequence_length = len(sequence)
        if self.hidden_capture.value is None or self.audio_capture.value is None:
            raise RuntimeError("Nano feature hooks were not ready before logits processing")

        row_scores = scores[0]
        float_scores = row_scores.float()
        candidate_word_indices = (
            [self.committed_word_index]
            if self.lock_started_hotword and self.committed_word_index is not None
            else range(len(self.hotword_token_ids))
        )
        if self.ablate_trie_order:
            # Keep one candidate row per hotword, exactly as in the complete
            # system.  Only the choice of token within that word changes.
            flat_ids = [
                int(token_id)
                for word_index in candidate_word_indices
                for token_id in self.hotword_token_ids[word_index]
            ]
            flat_scores = float_scores[
                torch.tensor(flat_ids, dtype=torch.long, device=row_scores.device)
            ].tolist()
            offset = 0
            all_rows = []
            for word_index in candidate_word_indices:
                ids = self.hotword_token_ids[word_index]
                token_index = max(
                    range(len(ids)), key=lambda index: flat_scores[offset + index]
                )
                all_rows.append((word_index, token_index, int(ids[token_index])))
                offset += len(ids)
        else:
            all_rows = [
                (
                    word_index,
                    self._current_index(word_index),
                    int(ids[self._current_index(word_index)]),
                )
                for word_index in candidate_word_indices
                for ids in [self.hotword_token_ids[word_index]]
            ]
        candidate_ids_all = torch.tensor(
            [row[2] for row in all_rows], dtype=torch.long, device=row_scores.device
        )
        selected_count = (
            len(all_rows)
            if self.ablate_rank_gate
            else min(self.candidate_cap, len(all_rows))
        )
        selected = row_scores[candidate_ids_all].topk(selected_count).indices.tolist()
        rows = [all_rows[index] for index in selected]
        selected_word_indices = {word_index for word_index, _, _ in rows}
        candidate_index_by_word = {
            word_index: token_index for word_index, token_index, _ in all_rows
        }
        step_trace_events: dict[int, dict[str, Any]] = {}
        for word_index in self.trace_word_indices:
            token_index = candidate_index_by_word[word_index]
            token_id = int(self.hotword_token_ids[word_index][token_index])
            base_logit = float(float_scores[token_id].item())
            base_rank = int(float_scores.gt(base_logit).sum().item()) + 1
            base_gap = max(0.0, float(float_scores.max().item()) - base_logit)
            stats = self._target_gate_stats[word_index]
            token_stats = self._trace_token_stats(word_index, token_index, token_id)
            for current in (stats, token_stats):
                current["observations"] += 1
                self._update_minimum(current, "minimum_base_rank", base_rank)
                self._update_minimum(current, "minimum_base_gap", base_gap)
            if word_index in selected_word_indices:
                stats["candidate_selected_steps"] += 1
                token_stats["candidate_selected_steps"] += 1
            else:
                stats["candidate_cap_rejected_steps"] += 1
            top1_id = int(float_scores.argmax().item())
            event = {
                "decoder_step": self.calls,
                "generated_prefix": self.tokenizer.decode(
                    self.generated, skip_special_tokens=True
                ),
                "token_index": token_index,
                "token_id": token_id,
                "token_text": self._candidate_token_text(
                    word_index, token_index
                ),
                "base_token_logit": base_logit,
                "base_probability": None,
                "base_rank": base_rank,
                "base_gap": base_gap,
                "base_top1_token_id": top1_id,
                "base_top1_token_text": self.tokenizer.decode(
                    [top1_id], skip_special_tokens=True
                ),
                "base_top1_logit": float(float_scores[top1_id].item()),
                "trusted_candidate": bool(
                    self.trusted_candidate_mask[word_index]
                ),
                "candidate_selected": word_index in selected_word_indices,
                "base_topk_allowed": None,
                "ctc_allowed": None,
                "ctc_reason": None,
                "ctc_score": None,
                "ctc_minimum_label_probability": None,
                "ctc_orthographic_share": None,
                "ctc_minimum_orthographic_share": None,
                "ctc_candidate_token_score": None,
                "ctc_base_token_score": None,
                "ctc_contrastive_margin": None,
                "raw_score": None,
                "gain": None,
                "emit_logit": None,
                "emit_probability": None,
                "emit_decision": None,
                "required_bias": None,
                "model_bias": None,
                "scaled_bias": None,
                "actual_token_bias": 0.0,
                "adjusted_token_logit": None,
                "adjusted_rank": None,
                "winner_token_id": None,
                "winner_token_text": None,
                "winner_logit": None,
                "emitted": False,
            }
            stats["events"].append(event)
            step_trace_events[word_index] = event
        candidate_ids = torch.tensor(
            [row[2] for row in rows], dtype=torch.long, device=row_scores.device
        )
        rank_width = min(100, int(float_scores.numel()))
        topk_width = min(
            max(rank_width, self.base_topk_gate, self.start_topk_gate),
            int(float_scores.numel()),
        )
        ranked_ids = float_scores.topk(topk_width).indices
        allowed_ids = set(
            int(value)
            for value in ranked_ids[: min(self.base_topk_gate, topk_width)].tolist()
        )
        start_allowed_ids = set(
            int(value)
            for value in ranked_ids[: min(self.start_topk_gate, topk_width)].tolist()
        )

        def base_topk_allowed(token_index: int, token_id: int) -> bool:
            if self.ablate_rank_gate:
                return True
            pool = start_allowed_ids if token_index == 0 else allowed_ids
            return int(token_id) in pool

        precomputed_ctc: dict[tuple[int, int], tuple[bool, str, Any]] = {}
        precomputed_contrastive: dict[tuple[int, int], tuple[float, float, float]] = {}
        decision_padding_mask = None
        fallback_thresholds = None
        if (
            self.grounded_prior_scores is None
            and getattr(self.fusion.config, "decision_mode", "hard_multi")
            in {"single_continuous", "single_hard"}
        ):
            decision_allowed = []
            fallback_threshold_values = []
            base_top1_id = int(float_scores.argmax().item())
            base_top1_text = self.tokenizer.decode(
                [base_top1_id], skip_special_tokens=True
            )
            for word_index, token_index, token_id in rows:
                ctc_result = self._candidate_ctc_eligible(
                    word_index,
                    token_index,
                    base_top1_text,
                    base_top1_id,
                )
                candidate_text = self._candidate_token_text(
                    word_index, token_index
                )
                natural_phonetic_evidence = bool(
                    token_index > 0
                    and word_index in self.active_base_anchored_starts
                    and (
                        phonetic_prefix_match(candidate_text, base_top1_text)
                        or common_latin_prefix_length(
                            candidate_text, base_top1_text
                        )
                        >= 3
                    )
                )
                contrastive_support = None
                if (
                    self.ctc_contrastive_margin_threshold >= -1.0
                    or getattr(
                        self.fusion.config,
                        "ctc_base_contrastive_residual",
                        False,
                    )
                ):
                    contrastive_support = self._ctc_contrastive_support(
                        word_index=word_index,
                        token_index=token_index,
                        token_id=token_id,
                        base_top1_id=base_top1_id,
                        base_top1_text=base_top1_text,
                        ctc_result=ctc_result,
                    )
                if contrastive_support is not None:
                    precomputed_contrastive[word_index, token_index] = contrastive_support
                    if (
                        self.ctc_contrastive_margin_threshold >= -1.0
                        and float(contrastive_support[2])
                        < self.ctc_contrastive_margin_threshold
                        and not natural_phonetic_evidence
                    ):
                        ctc_result = (
                            False,
                            "contrastive_ctc_margin",
                            ctc_result[2],
                        )
                trace_event = step_trace_events.get(word_index)
                if trace_event is not None and contrastive_support is not None:
                    trace_event.update({
                        "ctc_candidate_token_score": float(
                            contrastive_support[0]
                        ),
                        "ctc_base_token_score": float(contrastive_support[1]),
                        "ctc_contrastive_margin": float(contrastive_support[2]),
                    })
                base_logit = float(float_scores[int(token_id)].item())
                base_rank = int(float_scores.gt(base_logit).sum().item()) + 1
                if (
                    not ctc_result[0]
                    and self.rank_rescue_topk > 0
                    and base_rank <= self.rank_rescue_topk
                ):
                    ctc_result = (True, "base_rank_rescue", ctc_result[2])
                precomputed_ctc[word_index, token_index] = ctc_result
                alignment = ctc_result[2]
                strong_acoustic_evidence = bool(
                    getattr(
                        self.ctc_temporal_gate,
                        "exact_orthographic_rescue",
                        False,
                    )
                    and alignment is not None
                    and self._ctc_start_is_near_cursor(alignment)
                    and float(alignment.score) >= 0.8
                    and float(alignment.minimum_label_probability) >= 0.5
                    and float(alignment.orthographic_share) >= 0.02
                )
                homophone_orthographic_evidence = bool(
                    self.ctc_homophone_min_orthographic_share > 0.0
                    and alignment is not None
                    and self._ctc_start_is_near_cursor(alignment)
                    and float(alignment.score) >= 0.5
                    and float(alignment.minimum_label_probability) >= 0.5
                    and float(alignment.orthographic_share)
                    >= self.ctc_homophone_min_orthographic_share
                )
                latin_prefix_evidence = bool(
                    common_latin_prefix_length(
                        self.hotwords[word_index], base_top1_text
                    )
                    >= 3
                )
                strict_subtoken_override = bool(
                    int(token_id) != base_top1_id
                    and candidate_is_strict_subtoken(
                        candidate_text, base_top1_text
                    )
                    and not latin_prefix_evidence
                )
                numeric_conflict = numeric_orthography_conflict(
                    candidate_text, base_top1_text
                )
                untrusted_homophone_override = bool(
                    self.block_untrusted_homophone_override
                    and not self.trusted_candidate_mask[word_index]
                    and int(token_id) != base_top1_id
                    and (
                        numeric_conflict
                        or (
                            ctc_result[1] == "unmappable_passthrough"
                            and not latin_prefix_evidence
                        )
                        or (
                            normalize_for_match(candidate_text)
                            != normalize_for_match(base_top1_text)
                            and phonetic_prefix_match(
                                candidate_text, base_top1_text
                            )
                            and not latin_prefix_evidence
                            # Source trust is a retrieval prior, not acoustic
                            # evidence.  An exact bilingual CTC path with a
                            # strong weakest-token score may safely override
                            # that prior; weak/pinyin-only CTC remains blocked.
                            and not strong_acoustic_evidence
                            and (
                                self.require_trusted_homophone_override
                                or not homophone_orthographic_evidence
                            )
                        )
                    )
                )
                boundary_override = bool(
                    self.preserve_base_boundary_token
                    and int(token_id) != base_top1_id
                    and is_boundary_or_control_token(
                        base_top1_id,
                        base_top1_text,
                        self.special_token_ids,
                    )
                )
                weak_mismatched_start = bool(
                    self.ctc_mismatch_min_orthographic_share > 0.0
                    and token_index == 0
                    and int(token_id) != base_top1_id
                    and not phonetic_prefix_match(
                        candidate_text, base_top1_text
                    )
                    and not latin_prefix_evidence
                    and (
                        alignment is None
                        or float(alignment.orthographic_share)
                        < self.ctc_mismatch_min_orthographic_share
                    )
                )
                decision_allowed.append(
                    base_topk_allowed(token_index, token_id)
                    and bool(ctc_result[0])
                    and not untrusted_homophone_override
                    and not boundary_override
                    and not strict_subtoken_override
                    and not weak_mismatched_start
                )
                fallback_threshold = -1.0
                if (
                    self.trusted_candidate_threshold >= 0.0
                    and self.trusted_candidate_mask[word_index]
                ):
                    fallback_threshold = self.trusted_candidate_threshold
                if (
                    self.ctc_acoustic_rescue_threshold >= 0.0
                    and strong_acoustic_evidence
                ):
                    fallback_threshold = (
                        self.ctc_acoustic_rescue_threshold
                        if fallback_threshold < 0.0
                        else min(
                            fallback_threshold,
                            self.ctc_acoustic_rescue_threshold,
                        )
                    )
                if (
                    self.latin_prefix_rescue_threshold >= 0.0
                    and latin_prefix_evidence
                ):
                    fallback_threshold = (
                        self.latin_prefix_rescue_threshold
                        if fallback_threshold < 0.0
                        else min(
                            fallback_threshold,
                            self.latin_prefix_rescue_threshold,
                        )
                    )
                if (
                    self.natural_phonetic_rescue_threshold >= 0.0
                    and natural_phonetic_evidence
                    and self.trusted_candidate_mask[word_index]
                ):
                    fallback_threshold = (
                        self.natural_phonetic_rescue_threshold
                        if fallback_threshold < 0.0
                        else min(
                            fallback_threshold,
                            self.natural_phonetic_rescue_threshold,
                        )
                    )
                active_continuation_evidence = bool(
                    token_index > 0
                    and self.active_alignments[word_index] is not None
                    and ctc_result[1]
                    in {"base_anchored_continuation", "continuation_match"}
                )
                if (
                    self.active_continuation_rescue_threshold >= 0.0
                    and active_continuation_evidence
                ):
                    continuation_threshold = (
                        0.0
                        if self.committed_word_index == word_index
                        else self.active_continuation_rescue_threshold
                    )
                    fallback_threshold = (
                        continuation_threshold
                        if fallback_threshold < 0.0
                        else min(fallback_threshold, continuation_threshold)
                    )
                fallback_allowed = bool(
                    fallback_threshold >= 0.0
                    and base_topk_allowed(token_index, token_id)
                    and bool(ctc_result[0])
                    and ctc_result[1] != "base_rank_rescue"
                    and not numeric_conflict
                    and not untrusted_homophone_override
                    and normalize_for_match(candidate_text)
                    != normalize_for_match(base_top1_text)
                    and not boundary_override
                    and not strict_subtoken_override
                    and not weak_mismatched_start
                    and (
                        token_index > 0
                        or phonetic_prefix_match(candidate_text, base_top1_text)
                        or strong_acoustic_evidence
                        or latin_prefix_evidence
                    )
                )
                fallback_threshold_values.append(
                    fallback_threshold if fallback_allowed else -1.0
                )
            decision_padding_mask = torch.tensor(
                [[not value for value in decision_allowed]],
                dtype=torch.bool,
                device=row_scores.device,
            )
            fallback_thresholds = torch.tensor(
                fallback_threshold_values,
                dtype=torch.float32,
                device=row_scores.device,
            )

        self._ensure_static_projection_cache()
        current_embeddings = torch.stack(
            [
                self.hotword_token_embeddings[word_index][token_index]
                for word_index, token_index, _ in rows
            ]
        )
        prefix_embeddings = torch.stack(
            [
                (
                    self.hotword_token_embeddings[word_index][:token_index].mean(0)
                    if token_index > 0
                    else self.hotword_means.new_zeros(self.hotword_means.shape[-1])
                )
                for word_index, token_index, _ in rows
            ]
        )
        selected_means = torch.stack(
            [self.hotword_means[word_index] for word_index, _, _ in rows]
        )
        candidate_logits = float_scores[candidate_ids]
        log_normalizer = torch.logsumexp(float_scores, dim=-1)
        candidate_probabilities = torch.exp(candidate_logits - log_normalizer)
        for word_index, event in step_trace_events.items():
            token_index = candidate_index_by_word[word_index]
            token_id = int(self.hotword_token_ids[word_index][token_index])
            event["base_probability"] = float(
                torch.exp(float_scores[token_id] - log_normalizer).item()
            )
        gap = (float_scores.max() - candidate_logits).clamp_min(0.0)
        top_ids = ranked_ids[:rank_width]
        matches = candidate_ids[:, None].eq(top_ids[None, :])
        rank_values = torch.arange(
            1, rank_width + 1, dtype=torch.float32, device=row_scores.device
        )[None, :]
        ranks = torch.where(matches, rank_values, rank_width + 1.0).amin(dim=1)
        rank_normalized = torch.log1p(ranks) / math.log1p(rank_width + 1.0)
        progress = torch.tensor(
            [
                token_index / max(len(self.hotword_token_ids[word_index]) - 1, 1)
                for word_index, token_index, _ in rows
            ],
            dtype=torch.float32,
            device=row_scores.device,
        )
        is_start = torch.tensor(
            [float(token_index == 0) for _, token_index, _ in rows],
            dtype=torch.float32,
            device=row_scores.device,
        )
        is_end = torch.tensor(
            [
                float(token_index == len(self.hotword_token_ids[word_index]) - 1)
                for word_index, token_index, _ in rows
            ],
            dtype=torch.float32,
            device=row_scores.device,
        )
        scalar_columns = [
            candidate_probabilities,
            gap,
            rank_normalized,
            progress,
            is_start,
            is_end,
        ]
        if int(getattr(self.fusion.config, "scalar_dim", 6)) >= 8:
            ctc_features = torch.tensor(
                [
                    ctc_alignment_scalar_features(
                        precomputed_ctc.get(
                            (word_index, token_index), (False, "", None)
                        )[2]
                    )
                    for word_index, token_index, _token_id in rows
                ],
                dtype=torch.float32,
                device=row_scores.device,
            )
            scalar_columns.extend((ctc_features[:, 0], ctc_features[:, 1]))
        scalar_features = torch.stack(
            (
                *scalar_columns,
            ),
            dim=-1,
        )[None]

        generated = self.generated[-128:]
        history_ids = self.initial_history_ids + generated
        if history_ids:
            history_tensor = torch.tensor(
                history_ids, dtype=torch.long, device=row_scores.device
            )
            history = self.embed_tokens(history_tensor).detach().float()
        else:
            history = self.hotword_means.new_zeros((0, self.hotword_means.shape[-1]))
        history_segments = torch.cat(
            (
                torch.zeros(
                    len(self.initial_history_ids),
                    dtype=torch.long,
                    device=row_scores.device,
                ),
                torch.ones(
                    len(generated), dtype=torch.long, device=row_scores.device
                ),
            )
        )
        history_mask = torch.zeros(
            1, history.shape[0], dtype=torch.bool, device=row_scores.device
        )
        sample = {
            "decoder_states": self.hidden_capture.value[0:1],
            "audio_features": self.audio_capture.value,
            "history_features": history,
            "history_segment_ids": history_segments,
            "history_attention_mask": history_mask,
            "hotword_means": selected_means,
            "current_token_embeddings": current_embeddings[None],
            "prefix_embeddings": prefix_embeddings[None],
            "scalar_features": scalar_features,
            "candidate_token_ids": candidate_ids[None],
            "base_logits": float_scores[None],
            # The decoder only consumes candidate_bias. Materializing two
            # [1, vocab] tensors here costs memory bandwidth at every token.
            "return_full_logits": False,
            "candidate_competition_group_ids": torch.tensor(
                [
                    candidate_competition_group_ids(
                        [self.hotword_token_ids[word_index] for word_index, _, _ in rows],
                        [token_index for _word_index, token_index, _token_id in rows],
                        [
                            self._candidate_token_text(word_index, token_index)
                            for word_index, token_index, _token_id in rows
                        ],
                    )
                ],
                dtype=torch.long,
                device=row_scores.device,
            ),
            "ctc_base_contrastive_features": torch.tensor(
                [[
                    (
                        *precomputed_contrastive[word_index, token_index],
                        1.0,
                    )
                    if (word_index, token_index) in precomputed_contrastive
                    else (0.0, 0.0, 0.0, 0.0)
                    for word_index, token_index, _token_id in rows
                ]],
                dtype=torch.float32,
                device=row_scores.device,
            ),
        }
        if self.enable_runtime_cache:
            if (
                self._projected_hotword_means is None
                or self._projected_token_embeddings is None
                or self._projected_prefix_embeddings is None
                or self._projected_audio_features is None
            ):
                raise RuntimeError("Adapter static projection cache is incomplete")
            sample.update(
                {
                    "projected_hotword_means": torch.stack(
                        [
                            self._projected_hotword_means[word_index]
                            for word_index, _token_index, _token_id in rows
                        ]
                    ),
                    "projected_current_token_embeddings": torch.stack(
                        [
                            self._projected_token_embeddings[word_index][token_index]
                            for word_index, token_index, _token_id in rows
                        ]
                    )[None],
                    "projected_prefix_embeddings": torch.stack(
                        [
                            self._projected_prefix_embeddings[word_index][token_index]
                            for word_index, token_index, _token_id in rows
                        ]
                    )[None],
                    "projected_audio_features": self._projected_audio_features,
                }
            )
        if getattr(self.fusion.config, "ctc_logit_residual", False):
            sample["ctc_residual_mask"] = torch.tensor(
                [[
                    bool(
                        self.hotword_ctc_groups is not None
                        and ctc_current_token_supported(
                            self.tokenizer,
                            self.hotword_token_ids[word_index],
                            self.hotword_ctc_groups[word_index],
                            token_index,
                        )
                    )
                    for word_index, token_index, _token_id in rows
                ]],
                dtype=torch.bool,
                device=row_scores.device,
            )
        if int(getattr(self.fusion.config, "scalar_dim", 6)) >= 11:
            sample["scalar_features"] = torch.cat(
                (
                    sample["scalar_features"],
                    explicit_relation_scalar_features(sample),
                ),
                dim=-1,
            )
        if int(getattr(self.fusion.config, "scalar_dim", 6)) >= 13:
            sample["scalar_features"] = torch.cat(
                (
                    sample["scalar_features"],
                    candidate_input_scalar_features(
                        steps=1,
                        candidate_count=len(self.hotwords),
                        word_indices=torch.tensor(
                            [
                                word_index
                                for word_index, _token_index, _token_id in rows
                            ],
                            dtype=torch.long,
                            device=row_scores.device,
                        ),
                        device=row_scores.device,
                    ),
                ),
                dim=-1,
            )
        if int(getattr(self.fusion.config, "scalar_dim", 6)) >= 15:
            sample["scalar_features"] = torch.cat(
                (
                    sample["scalar_features"],
                    torch.tensor(
                        [
                            ctc_orthographic_scalar_features(
                                precomputed_ctc.get(
                                    (word_index, token_index), (False, "", None)
                                )[2]
                            )
                            for word_index, token_index, _token_id in rows
                        ],
                        dtype=torch.float32,
                        device=row_scores.device,
                    )[None],
                ),
                dim=-1,
            )
        if decision_padding_mask is not None:
            sample["candidate_decision_padding_mask"] = decision_padding_mask
        if self.grounded_prior_scores is None:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = self.fusion([sample])[0]
            model_candidate_bias = output["candidate_bias"][0].float()
            if fallback_thresholds is not None and fallback_thresholds.ge(0.0).any():
                emit_probability = output["emit_probability"][0].float()
                eligible = fallback_thresholds.ge(0.0) & emit_probability.ge(
                    fallback_thresholds.clamp_min(0.0)
                )
                if eligible.any():
                    winner_index = emit_probability.masked_fill(~eligible, -1.0).argmax()
                    winner = torch.zeros_like(eligible)
                    winner[winner_index] = True
                    fallback_bias = torch.where(
                        winner,
                        output["required_bias"][0].float(),
                        torch.zeros_like(model_candidate_bias),
                    )
                    # The decoder mode is single-winner. Keeping an independent
                    # model-selected bias can create an exact adjusted-logit tie
                    # and leave the trusted spelling unused.
                    model_candidate_bias = fallback_bias
                    output["trusted_fallback_decision"] = winner[None]
            candidate_bias = (
                model_candidate_bias * self.bias_scale
            ).clamp_max(float(self.fusion.config.max_bias))
            if self.fixed_bias is not None:
                candidate_bias = torch.full_like(candidate_bias, self.fixed_bias)
        else:
            learned_output = None
            if self.grounded_fusion_mode != "prior_only":
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    learned_output = self.fusion([sample])[0]
            prior_scores = torch.tensor(
                [self.grounded_prior_scores[word_index] for word_index, _, _ in rows],
                dtype=torch.float32,
                device=row_scores.device,
            )
            if self.grounded_prior_ranks is None:
                prior_rank_allowed = torch.ones_like(prior_scores, dtype=torch.bool)
            else:
                prior_rank_allowed = torch.tensor(
                    [
                        self.grounded_prior_topk <= 0
                        or self.grounded_prior_ranks[word_index] <= self.grounded_prior_topk
                        for word_index, _, _ in rows
                    ],
                    dtype=torch.bool,
                    device=row_scores.device,
                )
            if self.grounded_input_ranks is None:
                input_rank_allowed = torch.ones_like(prior_scores, dtype=torch.bool)
            else:
                input_rank_allowed = torch.tensor(
                    [
                        self.grounded_input_topk <= 0
                        or self.grounded_input_ranks[word_index]
                        <= self.grounded_input_topk
                        for word_index, _, _ in rows
                    ],
                    dtype=torch.bool,
                    device=row_scores.device,
                )
            if self.grounded_candidate_eligible is None:
                candidate_allowed = torch.ones_like(prior_scores, dtype=torch.bool)
            else:
                candidate_allowed = torch.tensor(
                    [
                        self.grounded_candidate_eligible[word_index]
                        for word_index, _, _ in rows
                    ],
                    dtype=torch.bool,
                    device=row_scores.device,
                )
            prior_allowed = (
                prior_scores >= self.grounded_prior_threshold
            ) & prior_rank_allowed & input_rank_allowed & candidate_allowed
            if self.grounded_base_consistency_margin >= 0.0:
                base_top1_id = int(float_scores.argmax().item())
                base_consistent = candidate_ids.eq(base_top1_id)
                # Bias is unnecessary for a token the Base already selects.
                # A competing token may override that path only when its
                # complete-word prior is clearly stronger.
                consistency_allowed = ~base_consistent
                if base_consistent.any():
                    best_base_prior = prior_scores.masked_fill(
                        ~base_consistent, float("-inf")
                    ).max()
                    consistency_allowed &= prior_scores.ge(
                        best_base_prior + self.grounded_base_consistency_margin
                    )
                prior_allowed &= consistency_allowed
            dynamic_bias = (gap + self.grounded_bias_margin).clamp_max(
                self.grounded_max_bias
            )
            learned_confidence = None
            if learned_output is not None:
                learned_confidence_tensor = learned_output.get("emit_probability")
                if learned_confidence_tensor is None:
                    learned_confidence_tensor = learned_output.get("gain")
                if learned_confidence_tensor is None:
                    raise RuntimeError(
                        "prior_and_learned requires Fusion gain or emit_probability"
                    )
                learned_confidence = learned_confidence_tensor[0].float()
            if self.grounded_fusion_mode == "prior_and_learned":
                assert learned_confidence is not None
                prior_allowed &= learned_confidence.ge(
                    self.grounded_learned_gate_threshold
                )
            if learned_output is None:
                model_candidate_bias = torch.where(
                    prior_allowed, dynamic_bias, torch.zeros_like(dynamic_bias)
                )
            elif self.grounded_fusion_mode == "prior_and_learned":
                model_candidate_bias = torch.where(
                    prior_allowed, dynamic_bias, torch.zeros_like(dynamic_bias)
                )
            else:
                learned_bias = learned_output["candidate_bias"][0].float()
                model_candidate_bias = torch.where(
                    prior_allowed, torch.maximum(learned_bias, dynamic_bias), learned_bias
                )
            candidate_bias = model_candidate_bias * self.bias_scale
            output = dict(learned_output or {})
            output.update({
                "required_bias": gap[None],
                "raw_score": prior_scores[None],
                "gain": prior_allowed.float()[None],
            })
            if learned_confidence is not None:
                output["learned_confidence"] = learned_confidence[None]
        if self.preserve_base_candidate_path or self.preserve_base_candidate_token:
            base_top1_id = int(float_scores.argmax().item())
            base_top1_text = normalize_for_match(
                self.tokenizer.decode(
                    [base_top1_id], skip_special_tokens=True
                )
            )
            base_path_is_valid = any(
                (
                    int(token_id) == base_top1_id
                    or (
                        bool(base_top1_text)
                        and normalize_for_match(
                            self.tokenizer.decode(
                                [int(token_id)], skip_special_tokens=True
                            )
                        )
                        == base_top1_text
                    )
                )
                and base_topk_allowed(_token_index, token_id)
                and (
                    self.preserve_base_candidate_token
                    or self._base_path_ctc_is_credible(
                        precomputed_ctc.get((word_index, _token_index))
                    )
                )
                for word_index, _token_index, token_id in rows
            )
            if base_path_is_valid:
                candidate_bias = torch.zeros_like(candidate_bias)
        token_bias: dict[int, torch.Tensor] = {}
        applied_candidates: list[tuple[int, int, int, float]] = []
        trace_applied: list[tuple[int, int, int, float, float]] = []
        for row_position, ((word_index, token_index, token_id), bias) in enumerate(
            zip(rows, candidate_bias)
        ):
            stats = self._target_gate_stats.get(word_index)
            event = step_trace_events.get(word_index)
            token_stats = (
                self._trace_token_stats(word_index, token_index, token_id)
                if stats is not None
                else None
            )
            if event is not None:
                event.update({
                    "required_bias": float(
                        output["required_bias"][0, row_position].item()
                    ),
                    "model_bias": float(model_candidate_bias[row_position].item()),
                    "scaled_bias": float(bias.item()),
                    "base_topk_allowed": base_topk_allowed(
                        token_index, token_id
                    ),
                })
                if "emit_probability" in output:
                    event.update({
                        "emit_logit": float(
                            output["emit_logits"][0, row_position].item()
                        ),
                        "emit_probability": float(
                            output["emit_probability"][0, row_position].item()
                        ),
                        "emit_decision": bool(
                            output["emit_decision"][0, row_position].item()
                        ),
                    })
                    if "base_emit_logits" in output:
                        event["base_emit_logit"] = float(
                            output["base_emit_logits"][0, row_position].item()
                        )
                    if "ctc_logit_residual" in output:
                        event["ctc_logit_residual"] = float(
                            output["ctc_logit_residual"][0, row_position].item()
                        )
                    if "ctc_base_contrastive_residual" in output:
                        event["ctc_base_contrastive_residual"] = float(
                            output["ctc_base_contrastive_residual"][
                                0, row_position
                            ].item()
                        )
                else:
                    event.update({
                        "raw_score": float(
                            output["raw_score"][0, row_position].item()
                        ),
                        "gain": float(output["gain"][0, row_position].item()),
                    })
                    if "learned_confidence" in output:
                        event["learned_confidence"] = float(
                            output["learned_confidence"][0, row_position].item()
                        )
            if not base_topk_allowed(token_index, token_id):
                self.topk_rejected_rows += 1
                if stats is not None and token_stats is not None:
                    stats["base_topk_rejected_steps"] += 1
                    token_stats.setdefault("base_topk_rejected_steps", 0)
                    token_stats["base_topk_rejected_steps"] += 1
                continue
            if stats is not None and token_stats is not None:
                stats["base_topk_allowed_steps"] += 1
                token_stats["base_topk_allowed_steps"] += 1
            if (word_index, token_index) in precomputed_ctc:
                ctc_allowed, ctc_reason, alignment = precomputed_ctc[
                    word_index, token_index
                ]
            else:
                ctc_allowed, ctc_reason, alignment = self._candidate_ctc_eligible(
                    word_index,
                    token_index,
                    self.tokenizer.decode(
                        [int(float_scores.argmax().item())],
                        skip_special_tokens=True,
                    ),
                    int(float_scores.argmax().item()),
                )
            if event is not None:
                event["ctc_allowed"] = ctc_allowed
                event["ctc_reason"] = ctc_reason
                event["ctc_cursor_frame"] = (
                    int(self.ctc_audio.cursor_frame)
                    if self.ctc_audio is not None
                    else None
                )
                if alignment is not None:
                    event["ctc_score"] = float(alignment.score)
                    event["ctc_minimum_label_probability"] = float(
                        alignment.minimum_label_probability
                    )
                    event["ctc_orthographic_share"] = float(
                        alignment.orthographic_share
                    )
                    event["ctc_minimum_orthographic_share"] = float(
                        alignment.minimum_orthographic_share
                    )
                    event["ctc_group_spans"] = [
                        [int(start), int(end)]
                        for start, end in alignment.group_spans
                    ]
            if stats is not None and token_stats is not None:
                for current in (stats, token_stats):
                    reasons = current["ctc_reasons"]
                    reasons[ctc_reason] = int(reasons.get(ctc_reason, 0)) + 1
            if not ctc_allowed:
                self.ctc_rejected_rows += 1
                if stats is not None and token_stats is not None:
                    stats["ctc_rejected_steps"] += 1
                    token_stats.setdefault("ctc_rejected_steps", 0)
                    token_stats["ctc_rejected_steps"] += 1
                continue
            natural_restart_only = bool(
                self.restart_requires_natural_start
                and token_index == 0
                and self.completed_generated_positions[word_index] >= 0
            )
            effective_bias = torch.zeros_like(bias) if natural_restart_only else bias
            if natural_restart_only:
                self.restart_natural_start_rows += 1
                if event is not None:
                    event["restart_natural_start_only"] = True
            own_bias = float(effective_bias.item())
            if own_bias > 0.0:
                applied_candidates.append(
                    (word_index, token_index, int(token_id), own_bias)
                )
            base_gap = max(
                0.0,
                float(float_scores.max().item())
                - float(float_scores[int(token_id)].item()),
            )
            if stats is not None and token_stats is not None:
                for current in (stats, token_stats):
                    current["ctc_allowed_steps"] += 1
                    current["applied_steps"] += 1
                    if own_bias > 0.0:
                        current["positive_bias_steps"] += 1
                    if "emit_probability" in output:
                        decision = bool(
                            output["emit_decision"][0, row_position].item()
                        )
                        current[
                            "binary_emit_steps" if decision else "binary_reject_steps"
                        ] += 1
                    self._update_maximum(current, "maximum_bias", own_bias)
                    self._update_maximum(
                        current, "maximum_bias_minus_gap", own_bias - base_gap
                    )
                trace_applied.append(
                    (word_index, token_index, int(token_id), own_bias, base_gap)
                )
            if self.trace_word_indices and own_bias > 0.0:
                emit_probability = output.get("emit_probability")
                self._positive_bias_events.append({
                    "decoder_step": self.calls,
                    "hotword": self.hotwords[word_index],
                    "token_index": token_index,
                    "token_id": int(token_id),
                    "token_text": self._candidate_token_text(
                        word_index, token_index
                    ),
                    "base_rank": int(ranks[row_position].item()),
                    "base_gap": float(base_gap),
                    "emit_probability": (
                        float(emit_probability[0, row_position].item())
                        if emit_probability is not None
                        else None
                    ),
                    "bias": own_bias,
                    "ctc_reason": ctc_reason,
                    "ctc_score": (
                        float(alignment.score) if alignment is not None else None
                    ),
                    "ctc_minimum_label_probability": (
                        float(alignment.minimum_label_probability)
                        if alignment is not None
                        else None
                    ),
                    "ctc_orthographic_share": (
                        float(alignment.orthographic_share)
                        if alignment is not None
                        else None
                    ),
                    "ctc_group_spans": (
                        [
                            [int(start), int(end)]
                            for start, end in alignment.group_spans
                        ]
                        if alignment is not None
                        else None
                    ),
                })
            previous = token_bias.get(int(token_id))
            token_bias[int(token_id)] = (
                effective_bias
                if previous is None
                else torch.maximum(previous, effective_bias)
            )
        if token_bias:
            ids = torch.tensor(list(token_bias), dtype=torch.long, device=row_scores.device)
            biases = torch.stack([token_bias[token_id] for token_id in token_bias])
            row_scores.index_add_(0, ids, biases.to(row_scores.dtype))
            self.applied_rows += len(token_bias)
            self.bias_sum += float(biases.sum().item())
            self.bias_max = max(self.bias_max, float(biases.max().item()))
        if (
            self.lock_started_hotword
            and self.committed_word_index is None
            and applied_candidates
        ):
            winner_id = int(row_scores.argmax().item())
            base_winner_id = int(float_scores.argmax().item())
            continuations = [
                item for item in applied_candidates
                if item[1] > 0
                and item[2] == winner_id
                and winner_id != base_winner_id
            ]
            if continuations:
                # Once a non-start token actually changes the Base winner, the
                # decoder has selected a concrete word branch. Latch it so the
                # remaining token(s) inherit the same complete-word evidence.
                self.pending_commit_word_index = max(
                    continuations, key=lambda item: item[3]
                )[0]
            starts = [
                item for item in applied_candidates
                if item[1] == 0 and item[2] == winner_id
            ]
            if starts and self.pending_commit_word_index is None:
                shared_start = self._shared_start_branch_count(winner_id) > 1
                if winner_id == base_winner_id or shared_start:
                    # Emitting a shared prefix (for example "阿" in both
                    # "阿晞" and "阿希流程") does not identify one complete
                    # hotword, regardless of whether bias changed the winner.
                    # Keep every matching Trie branch alive until their token
                    # paths actually diverge.
                    self.natural_start_commit_deferred_rows += 1
                else:
                    self.pending_commit_word_index = max(
                        starts, key=lambda item: item[3]
                    )[0]
        for word_index, token_index, token_id, _own_bias, _base_gap in trace_applied:
            stats = self._target_gate_stats[word_index]
            token_stats = self._trace_token_stats(word_index, token_index, token_id)
            actual_bias = float(token_bias[token_id].item())
            adjusted_logit = float(row_scores[token_id].float().item())
            adjusted_rank = int(row_scores.float().gt(adjusted_logit).sum().item()) + 1
            for current in (stats, token_stats):
                self._update_maximum(
                    current, "maximum_actual_token_bias", actual_bias
                )
                self._update_minimum(
                    current, "minimum_adjusted_rank", adjusted_rank
                )
                if adjusted_rank == 1:
                    current["adjusted_top1_steps"] += 1
        adjusted_scores = row_scores.float()
        winner_id = int(adjusted_scores.argmax().item())
        winner_logit = float(adjusted_scores[winner_id].item())
        winner_text = self.tokenizer.decode(
            [winner_id], skip_special_tokens=True
        )
        for event in step_trace_events.values():
            token_id = int(event["token_id"])
            adjusted_logit = float(adjusted_scores[token_id].item())
            event.update({
                "actual_token_bias": float(
                    token_bias[token_id].item() if token_id in token_bias else 0.0
                ),
                "adjusted_token_logit": adjusted_logit,
                "adjusted_rank": int(adjusted_scores.gt(adjusted_logit).sum().item()) + 1,
                "winner_token_id": winner_id,
                "winner_token_text": winner_text,
                "winner_logit": winner_logit,
                "emitted": winner_id == token_id,
            })
        self.calls += 1
        return scores

    def summary(self) -> dict[str, Any]:
        """Return decoder-side counters in the shared evaluator format."""

        return {
            "decoder_steps": self.calls,
            "candidate_rows": self.applied_rows,
            "unique_token_rows": self.applied_rows,
            "mean_candidate_bias": (
                self.bias_sum / self.applied_rows if self.applied_rows else 0.0
            ),
            "max_candidate_bias": self.bias_max,
            "generated_prefix_tokens_observed": len(self.generated),
            "candidate_cap": self.candidate_cap,
            "base_topk_gate": self.base_topk_gate,
            "start_topk_gate": self.start_topk_gate,
            "ablate_rank_gate": self.ablate_rank_gate,
            "ablate_trie_order": self.ablate_trie_order,
            "fixed_bias": self.fixed_bias,
            "static_projection_cache": self.enable_runtime_cache,
            "bias_mode": (
                "evidence_tiered_dynamic_gap"
                if self.grounded_prior_scores is not None
                else "learned_attention_fusion"
            ),
            "grounded_fusion_mode": self.grounded_fusion_mode,
            "grounded_learned_gate_threshold": (
                self.grounded_learned_gate_threshold
            ),
            "grounded_bias_margin": self.grounded_bias_margin,
            "grounded_max_bias": self.grounded_max_bias,
            "grounded_require_base_phonetic_start": (
                self.grounded_require_base_phonetic_start
            ),
            "require_base_phonetic_start": self.require_base_phonetic_start,
            "ctc_temporal_gate": self.ctc_temporal_gate is not None,
            "ctc_min_orthographic_share": self.ctc_min_orthographic_share,
            "ctc_continuation_threshold": self.ctc_continuation_threshold,
            "ctc_continuation_min_label_prob": (
                self.ctc_continuation_min_label_prob
            ),
            "active_continuation_rescue_threshold": (
                self.active_continuation_rescue_threshold
            ),
            "preserve_base_candidate_path": self.preserve_base_candidate_path,
            "preserve_base_candidate_token": self.preserve_base_candidate_token,
            "topk_rejected_candidate_rows": self.topk_rejected_rows,
            "ctc_rejected_candidate_rows": self.ctc_rejected_rows,
            "ctc_unmappable_passthrough_rows": (
                self.ctc_unmappable_passthrough_rows
            ),
            "ctc_unmappable_restart_rejected_rows": (
                self.ctc_unmappable_restart_rejected_rows
            ),
            "ctc_reused_span_rejected_rows": self.ctc_reused_span_rejected_rows,
            "restart_min_gap_tokens": self.restart_min_gap_tokens,
            "restart_cooldown_rejected_rows": self.restart_cooldown_rejected_rows,
            "restart_requires_natural_start": self.restart_requires_natural_start,
            "restart_natural_start_rows": self.restart_natural_start_rows,
            "natural_start_commit_deferred_rows": (
                self.natural_start_commit_deferred_rows
            ),
            "final_ctc_cursor_seconds": (
                self.ctc_audio.cursor_frame * self.ctc_audio.frame_seconds
                if self.ctc_audio is not None
                else None
            ),
        }


def resolve_audio(row: dict[str, Any], local_root: Path) -> Path:
    candidates = [
        str(row.get("raw_audio_path") or ""),
        str(row.get("audio_path") or ""),
        str(row.get("转录音频OSS") or ""),
        str(row.get("source_raw_audio_path") or ""),
    ]
    for raw in candidates:
        path = Path(raw)
        if raw and path.is_absolute() and path.is_file():
            return path
    for raw in candidates:
        if raw:
            clean = raw.split("?", 1)[0]
            if clean.startswith("oss://"):
                clean = clean[6:].partition("/")[2]
            for suffix in (clean, Path(clean).name):
                local = local_root / suffix
                if local.is_file():
                    return local
    raise FileNotFoundError(candidates[0] if candidates else "missing audio path")


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row_index, left in enumerate(reference, start=1):
        current = [row_index]
        for column, right in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left != right),
                )
            )
        previous = current
    return previous[-1]


def case_counts(label: str, prediction: str, hotwords: list[str]) -> tuple[int, int, int]:
    label_key = normalize_for_match(label)
    prediction_key = normalize_for_match(prediction)
    tp = fp = fn = 0
    for word in hotwords:
        key = normalize_for_match(word)
        if not key:
            continue
        in_label = key in label_key
        in_prediction = key in prediction_key
        if in_label and in_prediction:
            tp += 1
        elif in_label:
            fn += 1
        elif in_prediction:
            fp += 1
    return tp, fp, fn


def generate_one(
    wrapper,
    *,
    audio_path: Path,
    hotwords: list[str],
    max_new_tokens: int,
    history_context: str = "",
    logits_processor: LogitsProcessor | None = None,
) -> str:
    kwargs: dict[str, Any] = {}
    if logits_processor is not None:
        kwargs["llm_kwargs"] = {
            "logits_processor": LogitsProcessorList([logits_processor])
        }
    model = wrapper.model
    original_get_prompt = model.get_prompt
    prompt = original_get_prompt(hotwords, language="中文", itn=True)
    if history_context.strip():
        words = ", ".join(hotwords)
        prompt = (
            "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
            "**上下文信息：**\n\n\n"
            f"历史转写结果：[{history_context.strip()}]\n"
            f"热词列表：[{words}]\n"
            "语音转写成中文："
        )
    model.get_prompt = lambda *_args, **_kwargs: prompt
    try:
        result = wrapper.generate(
            input=[str(audio_path)],
            cache={},
            batch_size=1,
            hotwords=hotwords,
            language="中文",
            itn=True,
            max_length=max_new_tokens,
            disable_pbar=True,
            **kwargs,
        )
    finally:
        model.get_prompt = original_get_prompt
    return str(result[0]["text"])


def resolve_manifest_words(row: dict[str, Any], field_spec: str) -> list[str]:
    """Match the canonical evaluator: use the first non-empty listed field."""
    for field in str(field_spec or "").split(","):
        value = row.get(field.strip())
        words = split_hotwords(value)
        if words:
            return words
    return []


def select_inference_hotwords(value: Any, maximum: int) -> list[str]:
    """Keep the supplied order; labels are never read for candidate selection."""

    if maximum < 1:
        raise ValueError("candidate cap must be positive")
    return split_hotwords(value)[:maximum]


def manifest_work_items(manifest_path: Path) -> list[dict[str, Any]]:
    specs = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for spec in specs:
        jsonl_path = Path(spec["jsonl"])
        if not jsonl_path.is_absolute():
            jsonl_path = manifest_path.parent / jsonl_path
        audio_root = Path(spec.get("audio_root") or jsonl_path.parent)
        if not audio_root.is_absolute():
            audio_root = manifest_path.parent / audio_root
        source_rows = [
            line
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for stage in spec.get("stages", [3]):
            for case_index, line in enumerate(source_rows):
                source = json.loads(line)
                duration = float(source.get("speech_length") or 0.0)
                if duration <= 0:
                    messages = source.get("messages") or []
                    for message in messages:
                        if message.get("role") == "user":
                            duration = max(
                                duration,
                                float(message.get("end_time") or 0.0)
                                - float(message.get("start_time") or 0.0),
                            )
                items.append(
                    {
                        "dataset": str(spec["name"]),
                        "stage": int(stage),
                        "case_index": case_index,
                        "line": line,
                    "jsonl_parent": str(audio_root),
                        "hotword_field": str(spec.get("hotword_field") or "asr_hotwords"),
                        "eval_hotword_field": str(
                            spec.get("eval_hotword_field")
                            or spec.get("hotword_field")
                            or "asr_hotwords"
                        ),
                        "estimated_cost": max(duration, 1.0),
                    }
                )
    return items


def balanced_rank_items(
    items: list[dict[str, Any]], world_size: int, rank: int
) -> list[dict[str, Any]]:
    """Length-balanced LPT assignment avoids one slow fixed dataset per GPU."""
    bins: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    costs = [0.0] * world_size
    for item in sorted(items, key=lambda value: value["estimated_cost"], reverse=True):
        target = min(range(world_size), key=lambda index: (costs[index], index))
        bins[target].append(item)
        costs[target] += float(item["estimated_cost"])
    return sorted(
        bins[rank],
        key=lambda value: (
            value["dataset"],
            int(value["stage"]),
            int(value["case_index"]),
        ),
    )


def merge_and_summarize(
    output_dir: Path, world_size: int, checkpoint: Path, args: argparse.Namespace
) -> None:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = output_dir / f"predictions_rank{rank:02d}.jsonl"
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    rows.sort(
        key=lambda row: (
            row["dataset"],
            int(row.get("stage", 3)),
            int(row["case_index"]),
        )
    )
    merged = output_dir / "predictions.jsonl"
    with merged.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    metric_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, int | str]] = sorted(
        {(row["dataset"], int(row.get("stage", 3))) for row in rows}
    )
    groups.append(("overall", "all"))
    for dataset, stage in groups:
        selected = rows if dataset == "overall" else [
            row
            for row in rows
            if row["dataset"] == dataset and int(row.get("stage", 3)) == stage
        ]
        decoded = [
            row
            for row in selected
            if not row.get("error")
            and all(
                key in row
                for key in (
                    "reference_chars",
                    "baseline_edits",
                    "adapter_edits",
                    "baseline_tp",
                    "baseline_fp",
                    "baseline_fn",
                    "adapter_tp",
                    "adapter_fp",
                    "adapter_fn",
                )
            )
        ]
        for config in ("baseline", "adapter"):
            edits = sum(int(row[f"{config}_edits"]) for row in decoded)
            ref_chars = sum(int(row["reference_chars"]) for row in decoded)
            tp = sum(int(row[f"{config}_tp"]) for row in decoded)
            fp = sum(int(row[f"{config}_fp"]) for row in decoded)
            fn = sum(int(row[f"{config}_fn"]) for row in decoded)
            metric_rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "config": config,
                    "decoded": len(decoded),
                    "total": len(selected),
                    "cer": 100.0 * edits / max(ref_chars, 1),
                    "hotword_precision": tp / max(tp + fp, 1),
                    "hotword_recall": tp / max(tp + fn, 1),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    lookup = {
        (row["dataset"], str(row["stage"]), row["config"]): row
        for row in metric_rows
    }
    lines = [
        "# Fun-ASR-Nano Adapter Alignment Evaluation",
        "",
        f"- checkpoint: `{checkpoint}`",
        (
            "- gate: Base Top-K disabled"
            if args.ablate_rank_gate
            else f"- gate: Base Top-{args.base_topk_gate}"
        ),
        (
            f"- CTC temporal gate: enabled ({args.ctc_match_mode}, "
            f"`{args.ctc_model_dir}`)"
            if args.ctc_temporal_gate and not args.ablate_ctc_gate
            else "- CTC temporal gate: disabled"
        ),
        f"- Trie order gate: {'disabled' if args.ablate_trie_order else 'enabled'}",
        (
            f"- bias: fixed +{args.fixed_bias:g} logit per gated candidate"
            if args.fixed_bias is not None
            else "- bias: learned dynamic increment"
        ),
        f"- candidate cap: {args.candidate_cap}",
        f"- Adapter static projection cache: {not args.disable_adapter_runtime_cache}",
        f"- reused Base predictions: `{args.baseline_predictions or 'disabled'}`",
        f"- manifest: `{args.manifest}`",
        "- comparison: identical Nano base, audio, history and hotword prompt; Adapter biasing varies by ablation flag.",
        "",
        "| dataset | stage | decoded | Base CER | Adapter CER | delta CER | Base P | Adapter P | Base R | Adapter R | delta R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, stage in groups:
        base = lookup[(dataset, str(stage), "baseline")]
        adapter = lookup[(dataset, str(stage), "adapter")]
        lines.append(
            f"| {dataset} | {stage} | {adapter['decoded']}/{adapter['total']} | "
            f"{base['cer']:.4f} | {adapter['cer']:.4f} | "
            f"{adapter['cer'] - base['cer']:+.4f} | "
            f"{base['hotword_precision']:.4f} | {adapter['hotword_precision']:.4f} | "
            f"{base['hotword_recall']:.4f} | {adapter['hotword_recall']:.4f} | "
            f"{adapter['hotword_recall'] - base['hotword_recall']:+.4f} |"
        )
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    errors = [row for row in rows if row.get("error")]
    status = "COMPLETE" if not errors else "FAILED"
    (output_dir / "status.txt").write_text(
        f"{status} rows={len(rows)} errors={len(errors)} checkpoint={checkpoint}\n",
        encoding="utf-8",
    )
    if errors:
        samples = "; ".join(str(row.get("error")) for row in errors[:3])
        raise RuntimeError(f"evaluation produced {len(errors)} row errors: {samples}")


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size, device = distributed_context()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.nano_model_code is not None:
        model_code_dir = str(args.nano_model_code.resolve(strict=True).parent)
        if model_code_dir not in sys.path:
            sys.path.insert(0, model_code_dir)
    from funasr import AutoModel

    model_code = (
        {"remote_code": str(args.nano_model_code.resolve(strict=True))}
        if args.nano_model_code is not None
        else {}
    )
    wrapper = AutoModel(
        model=str(args.model_dir),
        trust_remote_code=True,
        device=str(device),
        disable_update=True,
        disable_log=True,
        disable_pbar=True,
        **model_code,
    )
    model = wrapper.model
    model.ctc_decoder = None
    if hasattr(model, "ctc"):
        model.ctc = None
    model.eval().requires_grad_(False)
    tokenizer = wrapper.kwargs["tokenizer"]
    embed_tokens = model.llm.get_input_embeddings()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("initialization_only") or checkpoint.get("ready_for_target_inference") is False:
        raise ValueError("checkpoint is an initialization, not an aligned Nano model")
    config = AttentionFusionConfig(**checkpoint["config"])
    fusion = FrozenFeatureAttentionFusion(config)
    fusion.load_state_dict(checkpoint["model_state"], strict=True)
    fusion.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    baseline_cache = load_baseline_predictions(args.baseline_predictions)
    if rank == 0 and baseline_cache:
        print(
            f"[eval] loaded {len(baseline_cache)} strict baseline cache rows from "
            f"{args.baseline_predictions}",
            flush=True,
        )

    ctc_gate = None
    if args.ctc_temporal_gate:
        ctc_gate = CTCTemporalGate(
            model_dir=args.ctc_model_dir,
            device=device,
            dtype=torch.bfloat16,
            match_mode="character",
        )

    case_filter: set[tuple[str, int]] = set()
    if args.case_filter.strip():
        for item in args.case_filter.split(","):
            dataset_name, separator, case_text = item.strip().partition(":")
            if not separator:
                raise ValueError(f"invalid --case-filter item: {item!r}")
            case_filter.add((dataset_name, int(case_text)))

    work_items = manifest_work_items(args.manifest)
    if args.limit:
        work_items = [
            item for item in work_items if int(item["case_index"]) < args.limit
        ]
    if case_filter:
        work_items = [
            item
            for item in work_items
            if (item["dataset"], int(item["case_index"])) in case_filter
        ]
    assigned_items = balanced_rank_items(work_items, world_size, rank)
    print(
        f"[eval rank={rank}] assigned={len(assigned_items)} "
        f"total={len(work_items)} manifest={args.manifest}",
        flush=True,
    )

    output_path = args.output_dir / f"predictions_rank{rank:02d}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_stream:
        for completed, item in enumerate(assigned_items, 1):
            dataset = item["dataset"]
            stage = int(item["stage"])
            case_index = int(item["case_index"])
            row: dict[str, Any] = {
                "dataset": dataset,
                "stage": stage,
                "case_index": case_index,
            }
            try:
                sample_started = synchronized_time(args.record_timing, device)
                source = json.loads(item["line"])
                label = reference_text(source)
                all_hotwords = resolve_manifest_words(source, item["hotword_field"])
                eval_hotwords = resolve_manifest_words(
                    source, item["eval_hotword_field"]
                )
                if not all_hotwords:
                    raise ValueError(
                        f"empty hotword field: {item['hotword_field']}"
                    )
                hotwords = select_inference_hotwords(
                    all_hotwords, maximum=args.candidate_cap
                )
                label_key_for_counts = normalize_for_match(label)
                selected_positive_count = sum(
                    bool(normalize_for_match(word) in label_key_for_counts)
                    for word in hotwords
                )
                selected_negative_count = len(hotwords) - selected_positive_count
                audio_path = resolve_audio(source, Path(item["jsonl_parent"]))
                history_context = (
                    str(source.get("hist_context") or "") if stage == 4 else ""
                )
                baseline_started = synchronized_time(args.record_timing, device)
                baseline_cache_hit = bool(baseline_cache)
                if baseline_cache_hit:
                    baseline = cached_baseline_prediction(
                        baseline_cache,
                        key=(dataset, int(stage), int(case_index)),
                        model_dir=args.model_dir,
                        max_new_tokens=args.max_new_tokens,
                        audio_path=audio_path,
                        hotwords=hotwords,
                        history_context=history_context,
                        label=label,
                    )
                else:
                    baseline = generate_one(
                        wrapper,
                        audio_path=audio_path,
                        hotwords=hotwords,
                        history_context=history_context,
                        max_new_tokens=args.max_new_tokens,
                    )
                baseline_finished = synchronized_time(args.record_timing, device)
                hotword_prepare_started = baseline_finished
                kept, token_ids, token_embeddings, hotword_means = tokenize_hotwords(
                    hotwords,
                    tokenizer=tokenizer,
                    embed_tokens=embed_tokens,
                    device=device,
                )
                hotword_prepare_finished = synchronized_time(
                    args.record_timing, device
                )
                ctc_audio = None
                hotword_ctc_groups = None
                ctc_started = hotword_prepare_finished
                if ctc_gate is not None:
                    hotword_ctc_groups = ctc_gate.prepare_hotword_groups(
                        tokenizer, token_ids
                    )
                    ctc_audio = ctc_gate.encode_audio(load_audio_16k(audio_path))
                ctc_finished = synchronized_time(args.record_timing, device)
                initial_history_ids = [
                    int(value)
                    for value in tokenizer(
                        history_context, add_special_tokens=False
                    )["input_ids"][-args.max_history_tokens :]
                ]
                hidden_capture = HiddenCapture()
                audio_capture = AudioCapture()
                processor = NanoAttentionBiasProcessor(
                    fusion=fusion,
                    hidden_capture=hidden_capture,
                    audio_capture=audio_capture,
                    embed_tokens=embed_tokens,
                    tokenizer=tokenizer,
                    hotwords=kept,
                    hotword_token_ids=token_ids,
                    hotword_token_embeddings=token_embeddings,
                    hotword_means=hotword_means,
                    candidate_cap=args.candidate_cap,
                    base_topk_gate=args.base_topk_gate,
                    bias_scale=args.bias_scale,
                    ablate_rank_gate=args.ablate_rank_gate,
                    ablate_trie_order=args.ablate_trie_order,
                    fixed_bias=args.fixed_bias,
                    ctc_temporal_gate=None if args.ablate_ctc_gate else ctc_gate,
                    ctc_audio=None if args.ablate_ctc_gate else ctc_audio,
                    hotword_ctc_groups=None if args.ablate_ctc_gate else hotword_ctc_groups,
                    ctc_start_threshold=args.ctc_start_threshold,
                    ctc_start_min_label_prob=args.ctc_start_min_label_prob,
                    ctc_lookback_seconds=args.ctc_lookback_seconds,
                    ctc_start_min_lookahead_seconds=args.ctc_start_min_lookahead_seconds,
                    ctc_start_lookahead_per_label_seconds=args.ctc_start_lookahead_per_label_seconds,
                    ctc_start_max_lookahead_seconds=args.ctc_start_max_lookahead_seconds,
                    ctc_max_gap_seconds=args.ctc_max_gap_seconds,
                    ctc_continuation_lookahead_seconds=args.ctc_continuation_lookahead_seconds,
                    ctc_cursor_lookahead_seconds=args.ctc_cursor_lookahead_seconds,
                    ctc_cursor_min_score=args.ctc_cursor_min_score,
                    initial_history_ids=initial_history_ids,
                    enable_runtime_cache=not args.disable_adapter_runtime_cache,
                )
                hidden_handle = model.llm.model.register_forward_hook(
                    hidden_capture.hook
                )
                audio_handle = model.audio_adaptor.register_forward_hook(
                    audio_capture.hook
                )
                adapter_started = synchronized_time(args.record_timing, device)
                try:
                    adapted = generate_one(
                        wrapper,
                        audio_path=audio_path,
                        hotwords=kept,
                        history_context=history_context,
                        max_new_tokens=args.max_new_tokens,
                        logits_processor=processor,
                    )
                finally:
                    hidden_handle.remove()
                    audio_handle.remove()
                adapter_finished = synchronized_time(args.record_timing, device)
                reference_key = normalize_for_match(label)
                baseline_key = normalize_for_match(baseline)
                adapted_key = normalize_for_match(adapted)
                baseline_tp, baseline_fp, baseline_fn = case_counts(
                    label, baseline, eval_hotwords
                )
                adapter_tp, adapter_fp, adapter_fn = case_counts(
                    label, adapted, eval_hotwords
                )
                prompt_text = model.get_prompt(kept, language="中文", itn=True)
                if history_context:
                    prompt_text = (
                        "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
                        "**上下文信息：**\n\n\n"
                        f"历史转写结果：[{history_context.strip()}]\n"
                        f"热词列表：[{', '.join(kept)}]\n"
                        "语音转写成中文："
                    )
                row.update(
                    {
                        "audio_path": str(audio_path),
                        "model_dir": str(args.model_dir.resolve()),
                        "max_new_tokens": int(args.max_new_tokens),
                        "baseline_cache_hit": baseline_cache_hit,
                        "hotword_field": item["hotword_field"],
                        "eval_hotword_field": item["eval_hotword_field"],
                        "prompt_hotwords": kept,
                        "original_prompt_hotword_count": len(all_hotwords),
                        "selected_positive_count": selected_positive_count,
                        "selected_negative_count": selected_negative_count,
                        "eval_hotwords": eval_hotwords,
                        "history_context": history_context,
                        "prompt_text": prompt_text,
                        "prompt_sha256": hashlib.sha256(
                            prompt_text.encode("utf-8")
                        ).hexdigest(),
                        "label": label,
                        "baseline_prediction": baseline,
                        "adapter_prediction": adapted,
                        "reference_chars": len(reference_key),
                        "baseline_edits": edit_distance(reference_key, baseline_key),
                        "adapter_edits": edit_distance(reference_key, adapted_key),
                        "baseline_tp": baseline_tp,
                        "baseline_fp": baseline_fp,
                        "baseline_fn": baseline_fn,
                        "adapter_tp": adapter_tp,
                        "adapter_fp": adapter_fp,
                        "adapter_fn": adapter_fn,
                        "adapter_processor_calls": processor.calls,
                        "adapter_applied_rows": processor.applied_rows,
                        "adapter_mean_applied_bias": processor.bias_sum
                        / max(processor.applied_rows, 1),
                        "adapter_max_applied_bias": processor.bias_max,
                        "adapter_topk_rejected_rows": processor.topk_rejected_rows,
                        "adapter_ctc_rejected_rows": processor.ctc_rejected_rows,
                        "adapter_ctc_unmappable_passthrough_rows": processor.ctc_unmappable_passthrough_rows,
                        "adapter_ctc_unmappable_restart_rejected_rows": processor.ctc_unmappable_restart_rejected_rows,
                        "adapter_ctc_reused_span_rejected_rows": processor.ctc_reused_span_rejected_rows,
                        "adapter_ctc_final_cursor_frame": (
                            ctc_audio.cursor_frame if ctc_audio is not None else None
                        ),
                        "timing_base_seconds": (
                            baseline_finished - baseline_started
                            if args.record_timing
                            else None
                        ),
                        "timing_hotword_prepare_seconds": (
                            hotword_prepare_finished - hotword_prepare_started
                            if args.record_timing
                            else None
                        ),
                        "timing_ctc_seconds": (
                            ctc_finished - ctc_started
                            if args.record_timing
                            else None
                        ),
                        "timing_adapter_seconds": (
                            adapter_finished - adapter_started
                            if args.record_timing
                            else None
                        ),
                        "timing_total_seconds": (
                            adapter_finished - sample_started
                            if args.record_timing
                            else None
                        ),
                        "error": "",
                    }
                )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_stream.flush()
            if completed % 10 == 0 or completed == len(assigned_items):
                print(
                    f"[eval rank={rank}] completed={completed}/{len(assigned_items)} "
                    f"{dataset} stage={stage} case={case_index} "
                    f"error={bool(row.get('error'))}",
                    flush=True,
                )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merge_and_summarize(args.output_dir, world_size, args.checkpoint, args)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
