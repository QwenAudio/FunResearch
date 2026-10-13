"""Optional character CTC acoustic gate for the Nano decoder.

This module mirrors the character mode used by the original Nano evaluator.
The CTC acoustic model is a separate, frozen Wav2Vec2ForCTC checkpoint; no
CTC weights are stored in a HotBias adapter checkpoint. The gate only checks
whether a hotword path is plausible near a monotonic audio cursor. The HotBias
attention module determines the bias magnitude independently.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class OrderedAlignment:
    score: float
    minimum_label_probability: float
    label_frames: tuple[int, ...]
    group_spans: tuple[tuple[int, int], ...]
    # These fields keep the decoder's alignment feature schema compatible.
    # Exact character matching has no pinyin alternatives, so both are 1.
    orthographic_share: float = 1.0
    minimum_orthographic_share: float = 1.0


@dataclass
class CTCAudioPosterior:
    probabilities: torch.Tensor
    frame_seconds: float
    cursor_frame: int
    initial_cursor_frame: int

    @property
    def frames(self) -> int:
        return int(self.probabilities.shape[0])


def best_ordered_alignment(
    probabilities: torch.Tensor,
    label_ids: Sequence[int],
    group_lengths: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
    max_gap_frames: int,
) -> OrderedAlignment | None:
    """Find the highest-posterior ordered label path in a local window.

    Intermediate frames are free. Consecutive labels are at most
    ``max_gap_frames`` apart. This is a local Viterbi objective, rather than a
    full-utterance CTC probability.
    """

    if probabilities.dim() != 2 or not label_ids:
        return None
    if sum(int(value) for value in group_lengths) != len(label_ids):
        raise ValueError("group_lengths must sum to the label count")
    start = max(0, int(start_frame))
    end = min(int(probabilities.shape[0]), int(end_frame))
    if end - start < len(label_ids):
        return None
    if any(label < 0 or label >= probabilities.shape[1] for label in label_ids):
        return None

    local = probabilities[start:end].float().clamp_min(1e-12)
    log_local = local.log()
    frame_count = int(local.shape[0])
    gap = max(1, int(max_gap_frames))

    previous = log_local[:, int(label_ids[0])].clone()
    backpointers: list[torch.Tensor] = []
    for label_id in label_ids[1:]:
        current = torch.full((frame_count,), float("-inf"), dtype=torch.float32)
        back = torch.full((frame_count,), -1, dtype=torch.long)
        for frame in range(1, frame_count):
            left = max(0, frame - gap)
            candidates = previous[left:frame]
            if candidates.numel() == 0:
                continue
            relative = int(torch.argmax(candidates).item())
            best_previous = left + relative
            current[frame] = candidates[relative] + log_local[frame, int(label_id)]
            back[frame] = best_previous
        previous = current
        backpointers.append(back)

    final_frame = int(torch.argmax(previous).item())
    if not bool(torch.isfinite(previous[final_frame])):
        return None
    path = [final_frame]
    cursor = final_frame
    for back in reversed(backpointers):
        cursor = int(back[cursor].item())
        if cursor < 0:
            return None
        path.append(cursor)
    path.reverse()
    absolute_path = tuple(start + value for value in path)
    score = float(torch.exp(previous[final_frame] / len(label_ids)).item())
    label_probabilities = [
        float(probabilities[frame, int(label)].item())
        for frame, label in zip(absolute_path, label_ids)
    ]

    spans: list[tuple[int, int]] = []
    offset = 0
    for length in group_lengths:
        group_length = int(length)
        group = absolute_path[offset : offset + group_length]
        spans.append((group[0], group[-1]))
        offset += group_length
    return OrderedAlignment(
        score=score,
        minimum_label_probability=min(label_probabilities),
        label_frames=absolute_path,
        group_spans=tuple(spans),
    )


class CTCTemporalGate:
    """Frozen Wav2Vec2 CTC posterior plus a monotonic decoder cursor."""

    _DIGIT_TO_CHINESE = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九", "〇": "零",
    }

    def __init__(
        self,
        *,
        model_dir: Path,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        match_mode: str = "character",
    ) -> None:
        if match_mode != "character":
            raise ValueError("this Nano release supports character CTC matching only")
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        has_safetensors = bool(list(model_dir.glob("*.safetensors")))
        torch_version = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        if not has_safetensors and (
            torch_version is None
            or tuple(map(int, torch_version.groups())) < (2, 6)
        ):
            raise RuntimeError(
                "The CTC model has no safetensors file. Loading its legacy "
                "pytorch_model.bin through Transformers requires PyTorch >= 2.6. "
                "Use a verified environment or a safetensors copy of the same model."
            )

        from transformers import AutoModelForCTC, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True
        )
        model = AutoModelForCTC.from_pretrained(
            str(model_dir),
            local_files_only=True,
            use_safetensors=True if has_safetensors else None,
        )
        self.model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank_id = int(getattr(self.processor.tokenizer, "pad_token_id", 0))
        self.match_mode = match_mode

    @classmethod
    def normalize_text_for_ctc(cls, text: str) -> str:
        """Map digits to Chinese pronunciations, including decimal points."""

        normalized = unicodedata.normalize("NFKC", str(text or "")).upper()
        converted: list[str] = []
        for index, character in enumerate(normalized):
            mapped = cls._DIGIT_TO_CHINESE.get(character)
            if mapped is not None:
                converted.append(mapped)
                continue
            if (
                character == "."
                and index > 0
                and index + 1 < len(normalized)
                and normalized[index - 1].isdigit()
                and normalized[index + 1].isdigit()
            ):
                converted.append("点")
                continue
            converted.append(character)
        return "".join(converted)

    def text_to_ids(self, text: str) -> list[int]:
        normalized = self.normalize_text_for_ctc(text)
        ids: list[int] = []
        for character in normalized:
            # English letters are not judged by this Chinese character model.
            if character.isascii() and character.isalpha():
                continue
            item = "|" if character.isspace() else character
            token_id = self.vocab.get(item)
            if token_id is not None and int(token_id) > 3:
                ids.append(int(token_id))
        return ids

    @staticmethod
    def _token_texts_with_complete_bytes(
        tokenizer: Any, token_ids: Sequence[int]
    ) -> list[str]:
        """Assign a split UTF-8 character to the final byte-level token."""

        ids = [int(token_id) for token_id in token_ids]
        fallback = [
            tokenizer.decode([token_id], skip_special_tokens=True)
            for token_id in ids
        ]
        if not ids or not callable(tokenizer):
            return fallback
        try:
            complete_text = tokenizer.decode(ids, skip_special_tokens=True)
            encoded = tokenizer(
                complete_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
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
                start, end,
            )
            output.append("" if shared_with_next else complete_text[start:end])
        return output

    def prepare_hotword_groups(
        self,
        tokenizer: Any,
        hotword_token_ids: Sequence[Sequence[int]],
    ) -> list[list[list[int]]]:
        groups: list[list[list[int]]] = []
        for token_ids in hotword_token_ids:
            token_texts = self._token_texts_with_complete_bytes(
                tokenizer, token_ids
            )
            adjusted_token_texts: list[str] = []
            for index, token_text in enumerate(token_texts):
                # Decimal points are often isolated by the decoder tokenizer.
                if (
                    token_text.strip() in {".", "．"}
                    and index > 0
                    and index + 1 < len(token_texts)
                    and token_texts[index - 1].rstrip()
                    and token_texts[index + 1].lstrip()
                    and token_texts[index - 1].rstrip()[-1].isdigit()
                    and token_texts[index + 1].lstrip()[0].isdigit()
                ):
                    token_text = "点"
                adjusted_token_texts.append(token_text)
            groups.append([self.text_to_ids(text) for text in adjusted_token_texts])
        return groups

    @torch.no_grad()
    def encode_audio(self, audio_16k: np.ndarray) -> CTCAudioPosterior:
        audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        inputs = self.processor(
            audio, sampling_rate=16_000, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device, dtype=self.dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits = self.model(
            input_values, attention_mask=attention_mask
        ).logits[0]
        probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        frame_seconds = (audio.shape[0] / 16_000.0) / max(probabilities.shape[0], 1)
        nonblank = probabilities.clone()
        nonblank[:, self.blank_id] = 0.0
        confidence = nonblank.max(dim=-1).values
        speech = torch.nonzero(confidence >= 0.05, as_tuple=False).flatten()
        first_speech = int(speech[0].item()) if speech.numel() else 0
        return CTCAudioPosterior(
            probabilities=probabilities,
            frame_seconds=frame_seconds,
            cursor_frame=first_speech,
            initial_cursor_frame=first_speech,
        )

    def align_hotword_near_cursor(
        self,
        audio: CTCAudioPosterior,
        token_groups: Sequence[Sequence[int]],
        *,
        lookback_seconds: float,
        minimum_lookahead_seconds: float,
        lookahead_per_label_seconds: float,
        maximum_lookahead_seconds: float,
        max_gap_seconds: float,
    ) -> OrderedAlignment | None:
        if not token_groups:
            return None
        mapped_group_indexes = [
            index for index, group in enumerate(token_groups) if group
        ]
        if not mapped_group_indexes:
            return None
        mapped_groups = [token_groups[index] for index in mapped_group_indexes]
        labels = [label for group in mapped_groups for label in group]
        frame_seconds = max(audio.frame_seconds, 1e-6)
        lookahead = min(
            maximum_lookahead_seconds,
            max(minimum_lookahead_seconds, lookahead_per_label_seconds * len(labels)),
        )
        start = audio.cursor_frame - round(lookback_seconds / frame_seconds)
        end = audio.cursor_frame + round(lookahead / frame_seconds) + 1
        alignment = best_ordered_alignment(
            audio.probabilities,
            labels,
            [len(group) for group in mapped_groups],
            start_frame=start,
            end_frame=end,
            max_gap_frames=round(max_gap_seconds / frame_seconds),
        )
        if alignment is None or len(mapped_groups) == len(token_groups):
            return alignment

        # Empty decoder tokens borrow their nearest mapped acoustic span.
        mapped_spans = dict(zip(mapped_group_indexes, alignment.group_spans))
        expanded_spans: list[tuple[int, int]] = []
        for token_index in range(len(token_groups)):
            if token_index in mapped_spans:
                expanded_spans.append(mapped_spans[token_index])
                continue
            nearest_index = min(
                mapped_group_indexes,
                key=lambda mapped_index: (
                    abs(mapped_index - token_index),
                    mapped_index < token_index,
                ),
            )
            expanded_spans.append(mapped_spans[nearest_index])
        return OrderedAlignment(
            score=alignment.score,
            minimum_label_probability=alignment.minimum_label_probability,
            label_frames=alignment.label_frames,
            group_spans=tuple(expanded_spans),
            orthographic_share=alignment.orthographic_share,
            minimum_orthographic_share=alignment.minimum_orthographic_share,
        )

    def advance_cursor(
        self,
        audio: CTCAudioPosterior,
        text: str,
        *,
        lookback_seconds: float,
        lookahead_seconds: float,
        max_gap_seconds: float,
        minimum_score: float,
    ) -> float | None:
        labels = self.text_to_ids(text)
        if not labels:
            return None
        frame_seconds = max(audio.frame_seconds, 1e-6)
        alignment = best_ordered_alignment(
            audio.probabilities,
            labels,
            [len(labels)],
            start_frame=audio.cursor_frame - round(lookback_seconds / frame_seconds),
            end_frame=audio.cursor_frame + round(lookahead_seconds / frame_seconds) + 1,
            max_gap_frames=round(max_gap_seconds / frame_seconds),
        )
        if alignment is None or alignment.score < minimum_score:
            return None
        audio.cursor_frame = max(audio.cursor_frame, alignment.label_frames[-1] + 1)
        return alignment.score

    @staticmethod
    def frame_to_seconds(audio: CTCAudioPosterior, frame: int) -> float:
        return float(frame) * audio.frame_seconds
