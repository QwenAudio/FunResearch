"""Small, model-free checks for the optional Nano character CTC gate."""

from __future__ import annotations

import math

import pytest
import torch

from fun_asr_nano.ctc_temporal_gate import (
    CTCAudioPosterior,
    CTCTemporalGate,
    best_ordered_alignment,
)


class ByteSplitTokenizer:
    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "池" if list(ids) == [10, 11] else "�"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert text == "池"
        assert not add_special_tokens and return_offsets_mapping
        return {"input_ids": [10, 11], "offset_mapping": [(0, 1), (0, 1)]}


def _unloaded_gate() -> CTCTemporalGate:
    gate = CTCTemporalGate.__new__(CTCTemporalGate)
    gate.vocab = {"池": 4, "州": 5, "三": 6, "点": 7, "五": 8}
    return gate


def _posterior() -> CTCAudioPosterior:
    probabilities = torch.full((7, 9), 0.001, dtype=torch.float32)
    probabilities[2, 4] = 0.8
    probabilities[4, 5] = 0.6
    return CTCAudioPosterior(
        probabilities=probabilities,
        frame_seconds=0.1,
        cursor_frame=2,
        initial_cursor_frame=2,
    )


def test_ordered_alignment_preserves_token_spans() -> None:
    audio = _posterior()
    alignment = best_ordered_alignment(
        audio.probabilities,
        [4, 5],
        [1, 1],
        start_frame=0,
        end_frame=7,
        max_gap_frames=3,
    )
    assert alignment is not None
    assert alignment.label_frames == (2, 4)
    assert alignment.group_spans == ((2, 2), (4, 4))
    assert alignment.score == pytest.approx(math.sqrt(0.8 * 0.6))
    assert alignment.minimum_label_probability == pytest.approx(0.6)
    assert best_ordered_alignment(
        audio.probabilities,
        [4, 5], [1, 1], start_frame=2, end_frame=4,
        max_gap_frames=3,
    ) is not None


def test_byte_split_and_unmapped_groups_keep_decoder_token_alignment() -> None:
    gate = _unloaded_gate()
    assert gate.prepare_hotword_groups(ByteSplitTokenizer(), [[10, 11]]) == [
        [[], [4]]
    ]
    audio = _posterior()
    alignment = gate.align_hotword_near_cursor(
        audio,
        [[], [4], [], [5]],
        lookback_seconds=0.2,
        minimum_lookahead_seconds=0.5,
        lookahead_per_label_seconds=0.1,
        maximum_lookahead_seconds=1.0,
        max_gap_seconds=0.3,
    )
    assert alignment is not None
    assert alignment.label_frames == (2, 4)
    assert alignment.group_spans == ((2, 2), (2, 2), (4, 4), (4, 4))
    assert alignment.orthographic_share == 1.0


def test_character_normalization_and_monotonic_cursor() -> None:
    gate = _unloaded_gate()
    assert gate.normalize_text_for_ctc("A３.５") == "A三点五"
    assert gate.text_to_ids("A３.５") == [6, 7, 8]
    audio = _posterior()
    score = gate.advance_cursor(
        audio,
        "池",
        lookback_seconds=0.1,
        lookahead_seconds=0.5,
        max_gap_seconds=0.3,
        minimum_score=0.5,
    )
    assert score == pytest.approx(0.8)
    assert audio.cursor_frame == 3
    gate.advance_cursor(
        audio,
        "州",
        lookback_seconds=0.1,
        lookahead_seconds=0.5,
        max_gap_seconds=0.3,
        minimum_score=0.5,
    )
    assert audio.cursor_frame == 5


def test_legacy_bin_requires_verified_torch_or_safetensors(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "pytorch_model.bin").touch()
    monkeypatch.setattr(torch, "__version__", "2.3.0+cu121")
    with pytest.raises(RuntimeError, match="PyTorch >= 2.6"):
        CTCTemporalGate(model_dir=tmp_path, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="character CTC"):
        CTCTemporalGate(
            model_dir=tmp_path,
            device=torch.device("cpu"),
            match_mode="pinyin_toneless",
        )
