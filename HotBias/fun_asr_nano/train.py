#!/usr/bin/env python3
"""Align HotBias to a frozen Fun-ASR-Nano base.

The default mode preserves the original six-projection alignment experiment.
``--train-all-adapter --hard-first-objective`` enables full Adapter SFT while
the Nano audio encoder, decoder, token embeddings, and LM head stay frozen.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import contextlib
import datetime
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterator
import unicodedata

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HOTWORD_SPLIT = re.compile(r"[,，、;；\n]+")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hotbias import (  # noqa: E402
    AttentionFusionConfig,
    FrozenFeatureAttentionFusion,
    coalesce_candidate_bias,
    sparse_biased_nll,
)
from hotbias.transfer import (  # noqa: E402
    BACKBONE_ADAPTER_PREFIXES,
    is_backbone_adapter_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--nano-model-code",
        type=Path,
        help="Path to model.py in the Fun-ASR repository when Nano is not registered.",
    )
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--local-audio-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Number of utterances processed together by the frozen Nano base.",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--audio-workers",
        type=int,
        default=0,
        help="CPU threads used to load audio and extract FBank within each micro-batch.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=0,
        help="Future CPU-prepared batches overlapped with the current GPU batch.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Use TF32 in the frozen FP32 audio encoder for additional throughput.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--bias-l2-weight", type=float, default=1e-3)
    parser.add_argument(
        "--train-all-adapter",
        action="store_true",
        help="Train the complete Attention Fusion Adapter instead of six projections.",
    )
    parser.add_argument(
        "--hard-first-objective",
        action="store_true",
        help="Enable Base-miss weighting, margin, keep, and active-negative losses.",
    )
    parser.add_argument("--positive-token-weight", type=float, default=1.5)
    parser.add_argument("--base-miss-word-weight", type=float, default=8.0)
    parser.add_argument("--base-miss-token-weight", type=float, default=12.0)
    parser.add_argument("--hard-margin", type=float, default=0.3)
    parser.add_argument("--hard-margin-loss-weight", type=float, default=0.5)
    parser.add_argument("--base-preservation-loss-weight", type=float, default=2.0)
    parser.add_argument("--active-negative-loss-weight", type=float, default=0.1)
    parser.add_argument("--base-correct-override-loss-weight", type=float, default=0.2)
    parser.add_argument("--base-correct-override-margin", type=float, default=0.0)
    parser.add_argument(
        "--base-topk-training-gate",
        type=int,
        default=0,
        help="Only apply candidate bias inside Base Top-K; 0 disables the gate.",
    )
    parser.add_argument(
        "--mine-nano-base-prediction",
        action="store_true",
        help="Batch-decode frozen Nano Base and use it for whole-word miss weighting.",
    )
    parser.add_argument(
        "--base-decode-max-new-tokens",
        type=int,
        default=192,
        help="Generation limit used only for Nano Base hard-case mining.",
    )
    parser.add_argument("--max-hotwords", type=int, default=40)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=384)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=2)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Nano adapter alignment requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def set_seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def load_selection(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    shards = [row for row in payload["shards"] if int(row["selected_rows"]) > 0]
    if not shards:
        raise ValueError(f"selection contains no rows: {path}")
    for shard in shards:
        shard_path = Path(shard["path"])
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        shard["path"] = str(shard_path.resolve())
    return shards, payload


def assigned_rows(
    shards: list[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
    epochs: int,
) -> Iterator[dict[str, Any] | None]:
    """Stream deterministic DDP shards and pad every rank to equal length."""

    rows_per_epoch = sum(int(shard["selected_rows"]) for shard in shards)
    slots_per_rank = math.ceil(rows_per_epoch / world_size)
    for epoch in range(epochs):
        emitted = 0
        global_index = 0
        for shard in shards:
            source_start = int(shard.get("start_row", 0))
            source_count = int(shard["selected_rows"])
            path = Path(shard["path"])
            with path.open(encoding="utf-8") as stream:
                for source_index, line in enumerate(stream):
                    if source_index < source_start:
                        continue
                    if source_index >= source_start + source_count:
                        break
                    selected_index = global_index
                    global_index += 1
                    if selected_index % world_size != rank:
                        continue
                    emitted += 1
                    try:
                        row = json.loads(line)
                        row["_nano_alignment_epoch"] = epoch
                        row["_nano_alignment_global_index"] = selected_index
                        row["_nano_alignment_source_path"] = str(path)
                        row["_nano_alignment_source_row"] = source_index
                        yield row
                    except Exception:
                        yield None
        while emitted < slots_per_rank:
            emitted += 1
            yield None


def reference_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return str(row.get("text") or row.get("label") or "")


def local_audio_path(row: dict[str, Any], local_root: Path) -> Path:
    raw = str(row.get("raw_audio_path") or row.get("audio_path") or "")
    if not raw:
        for message in row.get("messages") or []:
            content = str(message.get("content") or "")
            marker = "<|startofspeech|>!"
            if marker in content:
                raw = content.split(marker, 1)[1].split(
                    "<|endofspeech|>", 1
                )[0]
                break
    if raw.startswith("oss://"):
        # Map oss://bucket/key to a caller supplied local mirror root/key.
        key = raw[6:].partition("/")[2]
        if not key:
            raise ValueError(f"invalid OSS audio URI: {raw}")
        return local_root / key
    path = Path(raw)
    if path.is_absolute():
        return path
    return local_root / raw.lstrip("/")


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def split_hotwords(value: Any) -> list[str]:
    raw = [str(item) for item in value] if isinstance(value, list) else _HOTWORD_SPLIT.split(str(value or ""))
    output: list[str] = []
    seen: set[str] = set()
    for item in raw:
        word = item.strip().strip("`").strip()
        key = unicodedata.normalize("NFKC", word).casefold()
        if not word or key in seen:
            continue
        seen.add(key)
        output.append(word)
    return output


def select_hotwords(
    value: Any,
    reference: str,
    *,
    maximum: int,
    seed: int,
) -> tuple[list[str], int, int]:
    words = split_hotwords(value)
    reference_key = normalize_for_match(reference)
    positive = [
        index
        for index, word in enumerate(words)
        if normalize_for_match(word) in reference_key
    ]
    if len(words) <= maximum:
        return words, len(positive), len(words) - len(positive)
    positive = positive[:maximum]
    selected = set(positive)
    negatives = [index for index in range(len(words)) if index not in selected]
    generator = random.Random(seed)
    generator.shuffle(negatives)
    selected.update(negatives[: max(0, maximum - len(selected))])
    output = [word for index, word in enumerate(words) if index in selected]
    return output, len(positive), len(output) - len(positive)


def tokenize_hotwords(
    words: list[str],
    *,
    tokenizer,
    embed_tokens,
    device: torch.device,
) -> tuple[list[str], list[list[int]], list[torch.Tensor], torch.Tensor]:
    kept: list[str] = []
    token_ids: list[list[int]] = []
    lengths: list[int] = []
    for word in words:
        ids = [
            int(value)
            for value in tokenizer(word, add_special_tokens=False)["input_ids"]
        ]
        if not ids:
            continue
        kept.append(word)
        token_ids.append(ids)
        lengths.append(len(ids))
    if not kept:
        raise ValueError("all selected hotwords tokenize to empty")
    flattened = torch.tensor(
        [token_id for ids in token_ids for token_id in ids],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        flat_embeddings = embed_tokens(flattened).detach().float()
    token_embeddings = list(flat_embeddings.split(lengths))
    means = torch.stack([embeddings.mean(dim=0) for embeddings in token_embeddings])
    return kept, token_ids, token_embeddings, means


def build_dialog(model, audio_path: Path, hotwords: list[str], reference: str):
    prompt = model.get_prompt(hotwords, language="中文", itn=True)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                f"{prompt}<|startofspeech|>!{audio_path}<|endofspeech|>"
            ),
        },
        {"role": "assistant", "content": reference},
    ]


def prepare_nano_teacher_forcing(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    frontend,
    device: torch.device,
) -> dict[str, Any]:
    reference = reference_text(row).strip()
    if not reference:
        raise ValueError("empty reference")
    global_index = int(row.get("_nano_alignment_global_index", 0))
    hotwords, positive_words, negative_words = select_hotwords(
        row.get("asr_hotwords"),
        reference,
        maximum=args.max_hotwords,
        seed=args.seed + global_index,
    )
    if not hotwords:
        raise ValueError("sample has no hotwords")
    audio_path = local_audio_path(row, args.local_audio_root)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    embed_tokens = model.llm.get_input_embeddings()
    hotwords, token_ids, token_embeddings, hotword_means = tokenize_hotwords(
        hotwords,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        device=device,
    )
    dialog = build_dialog(model, audio_path, hotwords, reference)
    with torch.no_grad():
        inputs_embeds, _, batch, source_ids, meta = model.inference_prepare(
            [dialog],
            tokenizer=tokenizer,
            frontend=frontend,
            device=str(device),
            teacherforcing=True,
            batch_size=1,
            # The released Nano checkpoint keeps the frozen audio encoder in
            # FP32.  Autocast is applied only to the Qwen forward below; casting
            # FBank input to BF16 here breaks the remote-code load path.
            bf16=False,
            llm_dtype="bf16",
            disable_pbar=True,
        )
        attention_mask = batch["attention_mask"]
        output = model.llm.model(
            inputs_embeds=inputs_embeds.to(torch.bfloat16),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        target_ids = batch["target_ids"][0].long()[: args.max_target_tokens]
        source_length = int(source_ids.shape[1])
        target_length = int(target_ids.numel())
        decoder_states = output.last_hidden_state[
            0, source_length - 1 : source_length - 1 + target_length
        ].detach().float()
        if decoder_states.shape[0] != target_length:
            raise RuntimeError(
                f"teacher-forcing alignment mismatch: states={decoder_states.shape[0]} "
                f"targets={target_length} source={source_length}"
            )
        base_logits = model.llm.lm_head(decoder_states.to(torch.bfloat16)).detach().float()
        audio_features = meta["audio_adaptor_out"][0]
        audio_length = int(meta["audio_adaptor_out_lens"][0].item())
        audio_features = audio_features[:audio_length].detach().float()
        context_ids = tokenizer(
            str(row.get("one_pass_result") or ""),
            add_special_tokens=False,
        )["input_ids"][-args.max_context_tokens :]

    try:
        duration = float(row.get("duration") or row.get("raw_audio_duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "target_ids": [int(value) for value in target_ids.tolist()],
        "context_ids": [int(value) for value in context_ids],
        "hotword_token_ids": token_ids,
        "hotword_token_embeddings": token_embeddings,
        "hotword_means": hotword_means,
        "base_logits": base_logits,
        "decoder_states": decoder_states,
        "audio_features": audio_features,
        "positive_words": positive_words,
        "negative_words": negative_words,
        "duration": duration,
    }


def row_micro_batches(
    rows: Iterator[dict[str, Any] | None], size: int
) -> Iterator[list[dict[str, Any] | None]]:
    while True:
        batch = list(islice(rows, size))
        if not batch:
            return
        yield batch


def prepare_nano_cpu(
    row: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    frontend,
) -> dict[str, Any]:
    """Load one sample on CPU; all expensive neural work stays batched."""

    if row is None:
        raise ValueError("padded or malformed row")
    reference = reference_text(row).strip()
    if not reference:
        raise ValueError("empty reference")
    global_index = int(row.get("_nano_alignment_global_index", 0))
    hotwords, positive_words, negative_words = select_hotwords(
        row.get("asr_hotwords"),
        reference,
        maximum=args.max_hotwords,
        seed=args.seed + global_index,
    )
    if not hotwords:
        raise ValueError("sample has no hotwords")
    audio_path = local_audio_path(row, args.local_audio_root)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    dialog = build_dialog(model, audio_path, hotwords, reference)
    metadata: dict[str, Any] = {}
    batch = model.data_load_speech(
        model.data_template(dialog),
        tokenizer,
        frontend,
        meta_data=metadata,
        teacherforcing=True,
        batch_size=1,
        bf16=False,
        llm_dtype="bf16",
        disable_pbar=True,
    )
    context_ids = tokenizer(
        str(row.get("one_pass_result") or ""), add_special_tokens=False
    )["input_ids"][-args.max_context_tokens :]
    try:
        duration = float(row.get("duration") or row.get("raw_audio_duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "batch": batch,
        "audio_path": audio_path,
        "prompt": model.get_prompt(hotwords, language="中文", itn=True),
        "reference": reference,
        "hotwords": hotwords,
        "context_ids": [int(value) for value in context_ids],
        "positive_words": positive_words,
        "negative_words": negative_words,
        "duration": duration,
        "global_index": global_index,
        "curriculum_kind": str(
            row.get("event_mbr_category")
            or row.get("_hard_first_500h_kind")
            or row.get("_binary_curriculum_kind")
            or ""
        ),
        "difficulty_tags": list(row.get("difficulty_tags") or []),
    }


def prepare_nano_teacher_forcing_batch(
    rows: list[dict[str, Any] | None],
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    frontend,
    device: torch.device,
    executor: ThreadPoolExecutor | None,
    cpu_outcomes: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run audio encoder and teacher-forced Nano decoder as a true batch."""

    def load(row: dict[str, Any] | None) -> dict[str, Any]:
        return prepare_nano_cpu(
            row, args=args, model=model, tokenizer=tokenizer, frontend=frontend
        )

    loaded: list[dict[str, Any]] = []
    errors: list[str] = []
    if cpu_outcomes is not None:
        outcomes = cpu_outcomes
    elif executor is None:
        outcomes: list[Any] = []
        for row in rows:
            try:
                outcomes.append(load(row))
            except Exception as error:
                outcomes.append(error)
    else:
        futures = [executor.submit(load, row) for row in rows]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception as error:
                outcomes.append(error)
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            errors.append(f"{type(outcome).__name__}: {outcome}")
        else:
            loaded.append(outcome)
    if not loaded:
        return [], errors

    base_predictions = [str(item.get("base_prediction") or "") for item in loaded]
    if args.mine_nano_base_prediction:
        from hotbias.nano_batch_runtime import (
            generate_nano_batch,
            prepare_nano_generation_batch,
        )

        generation_batch = prepare_nano_generation_batch(
            model=model,
            tokenizer=tokenizer,
            frontend=frontend,
            prompts=[str(item["prompt"]) for item in loaded],
            audio_paths=[Path(item["audio_path"]) for item in loaded],
            device=device,
            audio_workers=0,
        )
        _, base_predictions = generate_nano_batch(
            llm=model.llm,
            tokenizer=tokenizer,
            prepared=generation_batch,
            max_new_tokens=args.base_decode_max_new_tokens,
        )

    speech_rows = [item["batch"]["speech"][0] for item in loaded]
    speech_lengths = torch.tensor(
        [int(item["batch"]["speech_lengths"][0, 0].item()) for item in loaded],
        dtype=torch.int32,
    )
    speech = torch.nn.utils.rnn.pad_sequence(speech_rows, batch_first=True)
    if device.type == "cuda":
        speech = speech.pin_memory()
        speech_lengths = speech_lengths.pin_memory()
    speech = speech.to(device=device, non_blocking=True)
    speech_lengths = speech_lengths.to(device=device, non_blocking=True)
    with torch.no_grad():
        encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
        adaptor_out, adaptor_out_lens = model.audio_adaptor(
            encoder_out, encoder_out_lens
        )

    input_rows: list[torch.Tensor] = []
    source_lengths: list[int] = []
    target_rows: list[torch.Tensor] = []
    for item in loaded:
        sample_batch = item["batch"]
        source_length = int(sample_batch["source_ids"].shape[1])
        target = sample_batch["target_ids"][0].long()[: args.max_target_tokens]
        source_lengths.append(source_length)
        target_rows.append(target)
        input_rows.append(
            sample_batch["input_ids"][0, : source_length + target.numel()].long()
        )
    padded_ids = torch.nn.utils.rnn.pad_sequence(
        input_rows, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones_like(row, dtype=torch.long) for row in input_rows],
        batch_first=True,
        padding_value=0,
    )
    if device.type == "cuda":
        padded_ids = padded_ids.pin_memory()
        attention_mask = attention_mask.pin_memory()
    padded_ids = padded_ids.to(device=device, non_blocking=True)
    attention_mask = attention_mask.to(device=device, non_blocking=True)
    embed_tokens = model.llm.get_input_embeddings()
    with torch.no_grad():
        inputs_embeds = embed_tokens(padded_ids)
        for index, item in enumerate(loaded):
            sample_batch = item["batch"]
            for turn_id in range(sample_batch["fbank_beg"].shape[1]):
                begin = int(sample_batch["fbank_beg"][0, turn_id].item())
                if begin <= 0:
                    continue
                fake_length = int(
                    sample_batch["fake_token_len"][0, turn_id].item()
                )
                available = int(adaptor_out_lens[index].item())
                length = min(fake_length, available)
                inputs_embeds[index, begin : begin + length] = adaptor_out[
                    index, :length
                ].to(inputs_embeds.dtype)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.llm.model(
                inputs_embeds=inputs_embeds.to(torch.bfloat16),
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

    decoder_rows = [
        output.last_hidden_state[
            index,
            source_length - 1 : source_length - 1 + target.numel(),
        ]
        for index, (source_length, target) in enumerate(
            zip(source_lengths, target_rows)
        )
    ]
    padded_decoder = torch.nn.utils.rnn.pad_sequence(
        decoder_rows, batch_first=True
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        padded_logits = model.llm.lm_head(padded_decoder.to(torch.bfloat16))

    prepared: list[dict[str, Any]] = []
    for index, (item, target, decoder, base_prediction) in enumerate(
        zip(loaded, target_rows, decoder_rows, base_predictions)
    ):
        hotwords, token_ids, token_embeddings, hotword_means = tokenize_hotwords(
            item["hotwords"],
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            device=device,
        )
        target_length = int(target.numel())
        audio_length = int(adaptor_out_lens[index].item())
        prepared.append(
            {
                "reference": item["reference"],
                "hotwords": hotwords,
                "base_prediction": base_prediction,
                "target_ids": [int(value) for value in target.tolist()],
                "context_ids": item["context_ids"],
                "hotword_token_ids": token_ids,
                "hotword_token_embeddings": token_embeddings,
                "hotword_means": hotword_means,
                "base_logits": padded_logits[index, :target_length].detach().float(),
                "decoder_states": decoder.detach().float(),
                "audio_features": adaptor_out[index, :audio_length].detach().float(),
                "source_embeddings": inputs_embeds[
                    index, : source_lengths[index]
                ].detach().float(),
                "audio_path": str(item["audio_path"]),
                "positive_words": item["positive_words"],
                "negative_words": item["negative_words"],
                "duration": item["duration"],
                "global_index": item["global_index"],
                "curriculum_kind": item["curriculum_kind"],
                "difficulty_tags": item["difficulty_tags"],
            }
        )
    return prepared, errors


def prefetch_nano_cpu_batches(
    batches: Iterator[list[dict[str, Any] | None]],
    *,
    depth: int,
    executor: ThreadPoolExecutor,
    args: argparse.Namespace,
    model,
    tokenizer,
    frontend,
) -> Iterator[tuple[list[dict[str, Any] | None], list[Any]]]:
    """Overlap CPU audio/FBank work for future batches with GPU execution."""

    iterator = iter(batches)
    pending: deque[tuple[list[dict[str, Any] | None], list[Any]]] = deque()

    def submit(rows: list[dict[str, Any] | None]):
        return (
            rows,
            [
                executor.submit(
                    prepare_nano_cpu,
                    row,
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    frontend=frontend,
                )
                for row in rows
            ],
        )

    for _ in range(max(depth, 1)):
        try:
            pending.append(submit(next(iterator)))
        except StopIteration:
            break
    while pending:
        rows, futures = pending.popleft()
        try:
            pending.append(submit(next(iterator)))
        except StopIteration:
            pass
        outcomes: list[Any] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception as error:
                outcomes.append(error)
        yield rows, outcomes


def build_attention_input(
    prepared: dict[str, Any],
    *,
    embed_tokens,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    target_ids = prepared["target_ids"]
    token_lists = prepared["hotword_token_ids"]
    token_embeddings = prepared["hotword_token_embeddings"]
    base_logits = prepared["base_logits"]
    steps = len(target_ids)
    depths = [0] * len(token_lists)
    token_index_rows: list[list[int]] = []
    candidate_id_rows: list[list[int]] = []
    positive_candidates = 0

    for target_id in target_ids:
        token_indexes: list[int] = []
        candidate_ids: list[int] = []
        for word_index, ids in enumerate(token_lists):
            depth = depths[word_index]
            token_index = depth if 0 < depth < len(ids) else 0
            candidate_id = ids[token_index]
            token_indexes.append(token_index)
            candidate_ids.append(candidate_id)
            positive_candidates += int(candidate_id == target_id)
        token_index_rows.append(token_indexes)
        candidate_id_rows.append(candidate_ids)
        for word_index, ids in enumerate(token_lists):
            depth = depths[word_index]
            if 0 < depth < len(ids):
                if target_id == ids[depth]:
                    next_depth = depth + 1
                    depths[word_index] = 0 if next_depth == len(ids) else next_depth
                elif len(ids) > 1 and target_id == ids[0]:
                    depths[word_index] = 1
                else:
                    depths[word_index] = 0
            elif len(ids) > 1 and target_id == ids[0]:
                depths[word_index] = 1

    token_indexes = torch.tensor(token_index_rows, dtype=torch.long, device=device)
    candidate_ids = torch.tensor(candidate_id_rows, dtype=torch.long, device=device)
    padded_embeddings = torch.nn.utils.rnn.pad_sequence(
        token_embeddings, batch_first=True
    )
    prefix_tables: list[torch.Tensor] = []
    for embeddings in token_embeddings:
        prefixes = torch.zeros_like(embeddings)
        if embeddings.shape[0] > 1:
            divisors = torch.arange(
                1, embeddings.shape[0], dtype=embeddings.dtype, device=device
            )[:, None]
            prefixes[1:] = embeddings.cumsum(dim=0)[:-1] / divisors
        prefix_tables.append(prefixes)
    padded_prefixes = torch.nn.utils.rnn.pad_sequence(prefix_tables, batch_first=True)
    gather_indexes = token_indexes.transpose(0, 1)[:, :, None].expand(
        -1, -1, padded_embeddings.shape[-1]
    )
    current_embeddings = padded_embeddings.gather(1, gather_indexes).transpose(0, 1)
    prefix_embeddings = padded_prefixes.gather(1, gather_indexes).transpose(0, 1)
    token_lengths = torch.tensor(
        [len(ids) for ids in token_lists], dtype=torch.long, device=device
    )[None, :]
    progress = token_indexes.float() / (token_lengths - 1).clamp_min(1).float()
    is_start = token_indexes.eq(0).float()
    is_end = token_indexes.eq(token_lengths - 1).float()

    # Cache all frozen-vocabulary statistics in one pass. Downstream hard-case
    # objectives reuse these tensors instead of rescanning the large ASR vocab.
    base_log_normalizer = torch.logsumexp(base_logits, dim=-1)
    candidate_logits = base_logits.gather(1, candidate_ids)
    candidate_probabilities = torch.exp(
        candidate_logits - base_log_normalizer[:, None]
    )
    rank_width = min(100, base_logits.shape[-1])
    top_values, top_ids = base_logits.topk(rank_width, dim=-1)
    top1 = top_values[:, :1]
    gap = (top1 - candidate_logits).clamp_min(0.0)
    matches = top_ids[:, :, None].eq(candidate_ids[:, None, :])
    rank_values = torch.arange(
        1, rank_width + 1, dtype=torch.float32, device=device
    )[None, :, None]
    found_rank = torch.where(matches, rank_values, rank_width + 1.0).amin(dim=1)
    rank_normalized = torch.log1p(found_rank) / math.log1p(rank_width + 1.0)
    scalar_features = torch.stack(
        (
            candidate_probabilities,
            gap,
            rank_normalized,
            progress,
            is_start,
            is_end,
        ),
        dim=-1,
    )

    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        target_embeddings = embed_tokens(target_tensor).detach().float()
        if prepared["context_ids"]:
            context_tensor = torch.tensor(
                prepared["context_ids"], dtype=torch.long, device=device
            )
            context_embeddings = embed_tokens(context_tensor).detach().float()
        else:
            context_embeddings = target_embeddings.new_zeros(
                (0, target_embeddings.shape[-1])
            )
    history_features = torch.cat((context_embeddings, target_embeddings), dim=0)
    history_segment_ids = torch.cat(
        (
            torch.zeros(len(prepared["context_ids"]), dtype=torch.long, device=device),
            torch.ones(steps, dtype=torch.long, device=device),
        )
    )
    positions = torch.arange(steps, device=device)
    blocked_target = positions[None, :] >= positions[:, None]
    history_attention_mask = torch.cat(
        (
            torch.zeros(
                steps,
                len(prepared["context_ids"]),
                dtype=torch.bool,
                device=device,
            ),
            blocked_target,
        ),
        dim=1,
    )
    return (
        {
            "decoder_states": prepared["decoder_states"],
            "audio_features": prepared["audio_features"],
            "history_features": history_features,
            "history_segment_ids": history_segment_ids,
            "history_attention_mask": history_attention_mask,
            "hotword_means": prepared["hotword_means"],
            "current_token_embeddings": current_embeddings,
            "prefix_embeddings": prefix_embeddings,
            "scalar_features": scalar_features,
            "candidate_token_ids": candidate_ids,
            "candidate_token_indexes": token_indexes,
            "base_logits": base_logits,
            "base_log_normalizer": base_log_normalizer,
            "base_candidate_logits": candidate_logits,
            "base_candidate_ranks": found_rank,
            "base_top1_logits": top_values[:, 0],
            "base_top1_ids": top_ids[:, 0],
        },
        positive_candidates,
    )


def _mark_subsequence(
    mask: torch.Tensor, sequence: list[int], needle: list[int]
) -> None:
    if not needle or len(needle) > len(sequence):
        return
    width = len(needle)
    for start in range(len(sequence) - width + 1):
        if sequence[start : start + width] == needle:
            mask[start : start + width] = True


def hotword_training_masks(
    prepared: dict[str, Any], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Mark reference hotword tokens and words missed by Nano free decoding."""

    target_ids = [int(value) for value in prepared["target_ids"]]
    reference_key = normalize_for_match(prepared["reference"])
    base_key = normalize_for_match(prepared["base_prediction"])
    positive = torch.zeros(len(target_ids), dtype=torch.bool, device=device)
    missed_word = torch.zeros_like(positive)
    positive_words = 0
    missed_words = 0
    for word, word_ids in zip(
        prepared["hotwords"], prepared["hotword_token_ids"], strict=True
    ):
        key = normalize_for_match(word)
        if not key or key not in reference_key:
            continue
        positive_words += 1
        word_mask = torch.zeros_like(positive)
        _mark_subsequence(word_mask, target_ids, word_ids)
        positive |= word_mask
        if key not in base_key:
            missed_words += 1
            missed_word |= word_mask
    return positive, missed_word, positive_words, missed_words


def hard_first_sample_loss(
    *,
    prepared: dict[str, Any],
    attention_input: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Compute deployment-aware hard-first SFT loss without dense delta logits."""

    targets = torch.tensor(prepared["target_ids"], dtype=torch.long, device=device)
    base_logits = attention_input["base_logits"]
    candidate_ids = attention_input["candidate_token_ids"]
    candidate_base_logits = base_logits.gather(1, candidate_ids)
    base_normalizer = torch.logsumexp(base_logits, dim=-1)
    base_target_logits = base_logits.gather(1, targets[:, None]).squeeze(1)

    if args.base_topk_training_gate > 0:
        width = min(int(args.base_topk_training_gate), int(base_logits.shape[-1]))
        top_ids = base_logits.topk(width, dim=-1).indices
        candidate_allowed = candidate_ids[:, :, None].eq(
            top_ids[:, None, :]
        ).any(dim=-1)
    else:
        candidate_allowed = torch.ones_like(candidate_ids, dtype=torch.bool)
    candidate_bias = output["candidate_bias"].float().masked_fill(
        ~candidate_allowed, 0.0
    )

    token_losses = sparse_biased_nll(
        base_log_normalizer=base_normalizer,
        base_target_logits=base_target_logits,
        candidate_token_ids=candidate_ids,
        base_candidate_logits=candidate_base_logits,
        candidate_bias=candidate_bias,
        target_token_ids=targets,
        reduction="none",
    )
    base_loss = (base_normalizer - base_target_logits).mean()
    biased_loss = token_losses.mean()

    positive_mask, missed_word_mask, positive_words, missed_words = (
        hotword_training_masks(prepared, device=device)
    )
    base_wrong = base_logits.argmax(dim=-1).ne(targets)
    missed_token_mask = positive_mask & base_wrong
    hard_mask = missed_word_mask | missed_token_mask
    candidate_target = candidate_ids.eq(targets[:, None])
    target_gate_allowed = (candidate_target & candidate_allowed).any(dim=-1)
    eligible_hard = hard_mask & target_gate_allowed

    token_weights = torch.ones_like(targets, dtype=torch.float32)
    token_weights = torch.where(
        positive_mask,
        torch.full_like(token_weights, args.positive_token_weight),
        token_weights,
    )
    token_weights = torch.where(
        missed_word_mask,
        torch.full_like(token_weights, args.base_miss_word_weight),
        token_weights,
    )
    token_weights = torch.where(
        missed_token_mask,
        torch.full_like(token_weights, args.base_miss_token_weight),
        token_weights,
    )
    weighted_loss = (token_losses * token_weights).sum() / token_weights.sum()

    effective_bias, _ = coalesce_candidate_bias(candidate_ids, candidate_bias)
    target_bias = torch.where(
        candidate_target, effective_bias, torch.zeros_like(effective_bias)
    ).amax(dim=-1)
    final_target_logits = base_target_logits + target_bias
    base_top1_logits = base_logits.max(dim=-1).values.detach()
    if eligible_hard.any():
        margin_loss = F.relu(
            base_top1_logits[eligible_hard]
            + args.hard_margin
            - final_target_logits[eligible_hard]
        ).mean()
    else:
        margin_loss = candidate_bias.sum() * 0.0

    intervention_target = candidate_target & eligible_hard[:, None]
    negative_candidates = candidate_allowed & ~intervention_target
    if negative_candidates.any():
        active_negative_loss = candidate_bias[negative_candidates].square().mean()
    else:
        active_negative_loss = candidate_bias.sum() * 0.0

    base_correct = ~base_wrong
    override_candidates = (
        candidate_allowed & base_correct[:, None] & ~candidate_target
    )
    override_excess = F.relu(
        candidate_base_logits
        + candidate_bias
        - base_target_logits[:, None]
        + args.base_correct_override_margin
    )
    if override_candidates.any():
        override_loss = override_excess[override_candidates].square().mean()
        override_count = int(
            override_excess[override_candidates].detach().gt(0).sum().item()
        )
    else:
        override_loss = candidate_bias.sum() * 0.0
        override_count = 0

    if candidate_allowed.any():
        bias_penalty = candidate_bias[candidate_allowed].square().mean()
    else:
        bias_penalty = candidate_bias.sum() * 0.0
    loss = (
        weighted_loss
        + args.base_preservation_loss_weight * biased_loss
        + args.hard_margin_loss_weight * margin_loss
        + args.active_negative_loss_weight * active_negative_loss
        + args.base_correct_override_loss_weight * override_loss
        + args.bias_l2_weight * bias_penalty
    )
    return loss, {
        "base_loss": base_loss,
        "biased_loss": biased_loss,
        "weighted_loss": weighted_loss,
        "margin_loss": margin_loss,
        "active_negative_loss": active_negative_loss,
        "base_correct_override_loss": override_loss,
        "bias_penalty": bias_penalty,
        "positive_words": positive_words,
        "base_miss_words": missed_words,
        "positive_token_positions": int(positive_mask.sum().item()),
        "base_miss_token_positions": int(hard_mask.sum().item()),
        "eligible_base_miss_positions": int(eligible_hard.sum().item()),
        "base_correct_positions": int(base_correct.sum().item()),
        "base_correct_override_candidates": override_count,
        "allowed_candidate_rows": int(candidate_allowed.sum().item()),
    }


def scheduler_factor(
    step: int, *, warmup_steps: int, total_steps: int, minimum_ratio: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def save_checkpoint(
    path: Path,
    *,
    fusion,
    optimizer,
    scheduler,
    optimizer_step: int,
    samples_seen: int,
    args: argparse.Namespace,
    selection: dict[str, Any],
) -> None:
    module = fusion.module if isinstance(fusion, DDP) else fusion
    payload = {
        "config": vars(module.config),
        "model_state": {
            name: value.detach().cpu() for name, value in module.state_dict().items()
        },
        "adapter_state": {
            name: value.detach().cpu()
            for name, value in module.state_dict().items()
            if is_backbone_adapter_key(name)
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "optimizer_step": optimizer_step,
        "samples_seen": samples_seen,
        "selection_manifest": str(args.selection_manifest),
        "selected_hours": float(selection["selected_hours"]),
        "target_name": "Fun-ASR-Nano-2512",
        "training_scope": (
            "all_attention_fusion_parameters"
            if args.train_all_adapter
            else "six_backbone_adapters_only"
        ),
        "frozen_cross_attention": not args.train_all_adapter,
        "hard_first_objective": (
            {
                "positive_token_weight": args.positive_token_weight,
                "base_miss_word_weight": args.base_miss_word_weight,
                "base_miss_token_weight": args.base_miss_token_weight,
                "hard_margin": args.hard_margin,
                "hard_margin_loss_weight": args.hard_margin_loss_weight,
                "base_preservation_loss_weight": args.base_preservation_loss_weight,
                "active_negative_loss_weight": args.active_negative_loss_weight,
                "base_correct_override_loss_weight": (
                    args.base_correct_override_loss_weight
                ),
                "base_correct_override_margin": args.base_correct_override_margin,
                "base_topk_training_gate": args.base_topk_training_gate,
                "nano_base_prediction_mined_online": (
                    args.mine_nano_base_prediction
                ),
            }
            if args.hard_first_objective
            else None
        ),
        "initialization_only": False,
        "ready_for_target_inference": True,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be positive")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be positive")
    if args.audio_workers < 0:
        raise ValueError("--audio-workers cannot be negative")
    if args.prefetch_batches < 0:
        raise ValueError("--prefetch-batches cannot be negative")
    if args.prefetch_batches and not args.audio_workers:
        raise ValueError("--prefetch-batches requires --audio-workers")
    if args.hard_first_objective and not args.mine_nano_base_prediction:
        raise ValueError(
            "--hard-first-objective requires --mine-nano-base-prediction so "
            "whole-word weights come from the frozen Nano Base"
        )
    if args.base_topk_training_gate < 0:
        raise ValueError("--base-topk-training-gate cannot be negative")
    rank, local_rank, world_size, device = distributed_context()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    set_seed(args.seed, rank)
    shards, selection = load_selection(args.selection_manifest)
    selected_rows = int(selection["selected_rows"])
    slots_per_rank = math.ceil(selected_rows / world_size) * args.epochs
    if args.smoke:
        slots_per_rank = min(
            slots_per_rank,
            args.micro_batch_size * args.gradient_accumulation,
        )
        args.max_optimizer_steps = 1
        args.log_every = 1
        args.save_every = 1
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.nano_model_code is not None:
        model_code_dir = str(args.nano_model_code.resolve(strict=True).parent)
        if model_code_dir not in sys.path:
            sys.path.insert(0, model_code_dir)
    from funasr import AutoModel

    load_started = time.time()
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
    set_attention = getattr(model.llm, "set_attn_implementation", None)
    if callable(set_attention):
        set_attention(args.attention_backend)
    tokenizer = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    if rank == 0:
        print(f"[load] Nano ready in {time.time() - load_started:.1f}s", flush=True)

    initialization = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=True
    )
    config = AttentionFusionConfig(**initialization["config"])
    if config.audio_dim != 1024 or config.text_dim != 1024:
        raise ValueError(f"Nano initialization dimensions are wrong: {config}")
    fusion_module = FrozenFeatureAttentionFusion(config)
    fusion_module.load_state_dict(initialization["model_state"], strict=True)
    fusion_module.requires_grad_(args.train_all_adapter)
    if not args.train_all_adapter:
        fusion_module.requires_grad_(False)
        for name, parameter in fusion_module.named_parameters():
            if is_backbone_adapter_key(name):
                parameter.requires_grad_(True)
    fusion_module.to(device=device, dtype=torch.float32).train()
    trainable = {
        name: parameter.numel()
        for name, parameter in fusion_module.named_parameters()
        if parameter.requires_grad
    }
    frozen_core = {
        name: parameter.numel()
        for name, parameter in fusion_module.named_parameters()
        if not parameter.requires_grad
    }
    if not args.train_all_adapter and set(
        name.split(".", 1)[0] + "." for name in trainable
    ) != set(BACKBONE_ADAPTER_PREFIXES):
        raise RuntimeError(f"unexpected trainable parameter scope: {sorted(trainable)}")

    fusion = (
        DDP(fusion_module, device_ids=[local_rank], broadcast_buffers=False)
        if world_size > 1
        else fusion_module
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in fusion.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    micro_batches_per_rank = math.ceil(slots_per_rank / args.micro_batch_size)
    total_steps = math.ceil(
        micro_batches_per_rank / args.gradient_accumulation
    )
    if args.max_optimizer_steps:
        total_steps = min(total_steps, args.max_optimizer_steps)
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))
    minimum_ratio = args.minimum_learning_rate / args.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            minimum_ratio=minimum_ratio,
        ),
    )

    if rank == 0:
        audit = {
            "model_dir": str(args.model_dir),
            "initial_checkpoint": str(args.initial_checkpoint),
            "world_size": world_size,
            "selected_rows": selected_rows,
            "selected_hours": selection["selected_hours"],
            "epochs": args.epochs,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_global_batch": (
                world_size * args.micro_batch_size * args.gradient_accumulation
            ),
            "planned_optimizer_steps": total_steps,
            "trainable_parameters": sum(trainable.values()),
            "frozen_fusion_core_parameters": sum(frozen_core.values()),
            "frozen_nano_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_names": trainable,
            "frozen_cross_attention": all(
                not parameter.requires_grad
                for name, parameter in fusion_module.named_parameters()
                if "cross_attention" in name
            ),
            "optimizer_only_contains_trainable_adapters": {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            == {id(parameter) for parameter in fusion.parameters() if parameter.requires_grad},
            "training_scope": (
                "all_attention_fusion_parameters"
                if args.train_all_adapter
                else "six_backbone_adapters_only"
            ),
            "hard_first_objective": args.hard_first_objective,
            "nano_base_prediction_mined_online": args.mine_nano_base_prediction,
            "base_topk_training_gate": args.base_topk_training_gate,
        }
        (args.output_dir / "parameter_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_config = {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            **{key: audit[key] for key in (
                "world_size",
                "selected_rows",
                "selected_hours",
                "effective_global_batch",
                "planned_optimizer_steps",
                "trainable_parameters",
            )},
        }
        (args.output_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[audit] trainable parameters={sum(trainable.values()):,}; "
            f"frozen Fusion core={sum(frozen_core.values()):,}; steps={total_steps}",
            flush=True,
        )

    iterator = assigned_rows(
        shards, rank=rank, world_size=world_size, epochs=args.epochs
    )
    if args.smoke:
        iterator = iter([next(iterator) for _ in range(slots_per_rank)])
    batches = row_micro_batches(iterator, args.micro_batch_size)
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    samples_seen = 0
    accumulation_valid = 0
    interval = {
        "raw": 0.0,
        "valid": 0.0,
        "failed": 0.0,
        "duration_seconds": 0.0,
        "base_loss": 0.0,
        "biased_loss": 0.0,
        "weighted_loss": 0.0,
        "margin_loss": 0.0,
        "active_negative_loss": 0.0,
        "base_correct_override_loss": 0.0,
        "bias_penalty": 0.0,
        "mean_bias": 0.0,
        "max_bias": 0.0,
        "positive_candidates": 0.0,
        "candidate_rows": 0.0,
        "allowed_candidate_rows": 0.0,
        "positive_words": 0.0,
        "base_miss_words": 0.0,
        "positive_token_positions": 0.0,
        "base_miss_token_positions": 0.0,
        "eligible_base_miss_positions": 0.0,
        "base_correct_positions": 0.0,
        "base_correct_override_candidates": 0.0,
    }
    interval_started = time.time()
    log_path = args.output_dir / "training_log.jsonl"
    embed_tokens = model.llm.get_input_embeddings()
    executor = (
        ThreadPoolExecutor(
            max_workers=args.audio_workers,
            thread_name_prefix="nano-audio",
        )
        if args.audio_workers > 0
        else None
    )
    if executor is not None and args.prefetch_batches > 0:
        batch_stream: Iterator[
            tuple[list[dict[str, Any] | None], list[Any] | None]
        ] = prefetch_nano_cpu_batches(
            batches,
            depth=args.prefetch_batches,
            executor=executor,
            args=args,
            model=model,
            tokenizer=tokenizer,
            frontend=frontend,
        )
    else:
        batch_stream = ((rows, None) for rows in batches)

    try:
        for micro_index, (rows, cpu_outcomes) in enumerate(
            batch_stream, start=1
        ):
            if micro_index > micro_batches_per_rank:
                break
            sync_now = (
                micro_index % args.gradient_accumulation == 0
                or micro_index == micro_batches_per_rank
            )
            sync_context = (
                contextlib.nullcontext()
                if sync_now or not isinstance(fusion, DDP)
                else fusion.no_sync()
            )
            with sync_context:
                prepared_batch, error_messages = prepare_nano_teacher_forcing_batch(
                    rows,
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    frontend=frontend,
                    device=device,
                    executor=(None if cpu_outcomes is not None else executor),
                    cpu_outcomes=cpu_outcomes,
                )
                attention_inputs: list[dict[str, torch.Tensor]] = []
                positive_counts: list[int] = []
                for prepared in prepared_batch:
                    attention_input, positive_candidates = build_attention_input(
                        prepared, embed_tokens=embed_tokens, device=device
                    )
                    attention_input["return_full_logits"] = False
                    attention_inputs.append(attention_input)
                    positive_counts.append(positive_candidates)

                if attention_inputs:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = fusion(attention_inputs)
                    sample_losses: list[torch.Tensor] = []
                    base_losses: list[torch.Tensor] = []
                    biased_losses: list[torch.Tensor] = []
                    weighted_losses: list[torch.Tensor] = []
                    margin_losses: list[torch.Tensor] = []
                    active_negative_losses: list[torch.Tensor] = []
                    override_losses: list[torch.Tensor] = []
                    bias_penalties: list[torch.Tensor] = []
                    mean_biases: list[float] = []
                    max_biases: list[float] = []
                    candidate_rows = 0
                    hard_counts = {
                        "allowed_candidate_rows": 0,
                        "positive_words": 0,
                        "base_miss_words": 0,
                        "positive_token_positions": 0,
                        "base_miss_token_positions": 0,
                        "eligible_base_miss_positions": 0,
                        "base_correct_positions": 0,
                        "base_correct_override_candidates": 0,
                    }
                    for prepared, attention_input, output in zip(
                        prepared_batch, attention_inputs, outputs
                    ):
                        if args.hard_first_objective:
                            sample_loss, sample_metrics = hard_first_sample_loss(
                                prepared=prepared,
                                attention_input=attention_input,
                                output=output,
                                args=args,
                                device=device,
                            )
                            base_loss = sample_metrics["base_loss"]
                            biased_loss = sample_metrics["biased_loss"]
                            weighted_loss = sample_metrics["weighted_loss"]
                            margin_loss = sample_metrics["margin_loss"]
                            active_negative_loss = sample_metrics[
                                "active_negative_loss"
                            ]
                            override_loss = sample_metrics[
                                "base_correct_override_loss"
                            ]
                            bias_penalty = sample_metrics["bias_penalty"]
                            for key in hard_counts:
                                hard_counts[key] += int(sample_metrics[key])
                        else:
                            targets = torch.tensor(
                                prepared["target_ids"],
                                dtype=torch.long,
                                device=device,
                            )
                            base_logits = attention_input["base_logits"]
                            base_log_normalizer = torch.logsumexp(
                                base_logits, dim=-1
                            )
                            base_target_logits = base_logits.gather(
                                1, targets[:, None]
                            ).squeeze(1)
                            base_loss = (
                                base_log_normalizer - base_target_logits
                            ).mean()
                            candidate_token_ids = attention_input[
                                "candidate_token_ids"
                            ]
                            biased_loss = sparse_biased_nll(
                                base_log_normalizer=base_log_normalizer,
                                base_target_logits=base_target_logits,
                                candidate_token_ids=candidate_token_ids,
                                base_candidate_logits=base_logits.gather(
                                    1, candidate_token_ids
                                ),
                                candidate_bias=output["candidate_bias"],
                                target_token_ids=targets,
                            )
                            bias_penalty = (
                                output["candidate_bias"].float().square().mean()
                            )
                            weighted_loss = biased_loss
                            margin_loss = biased_loss * 0.0
                            active_negative_loss = biased_loss * 0.0
                            override_loss = biased_loss * 0.0
                            sample_loss = (
                                biased_loss + args.bias_l2_weight * bias_penalty
                            )
                            hard_counts["allowed_candidate_rows"] += int(
                                output["candidate_bias"].numel()
                            )
                        sample_losses.append(sample_loss)
                        base_losses.append(base_loss)
                        biased_losses.append(biased_loss)
                        weighted_losses.append(weighted_loss)
                        margin_losses.append(margin_loss)
                        active_negative_losses.append(active_negative_loss)
                        override_losses.append(override_loss)
                        bias_penalties.append(bias_penalty)
                        biases = output["candidate_bias"].detach().float()
                        mean_biases.append(float(biases.mean().item()))
                        max_biases.append(float(biases.max().item()))
                        candidate_rows += biases.numel()
                    loss = torch.stack(sample_losses).mean()
                    loss = loss / args.gradient_accumulation
                    valid = len(prepared_batch)
                    duration = sum(
                        float(prepared["duration"])
                        for prepared in prepared_batch
                    )
                    base_loss_sum = torch.stack(base_losses).sum()
                    biased_loss_sum = torch.stack(biased_losses).sum()
                    weighted_loss_sum = torch.stack(weighted_losses).sum()
                    margin_loss_sum = torch.stack(margin_losses).sum()
                    active_negative_loss_sum = torch.stack(
                        active_negative_losses
                    ).sum()
                    override_loss_sum = torch.stack(override_losses).sum()
                    bias_penalty_sum = torch.stack(bias_penalties).sum()
                    positive_candidates = sum(positive_counts)
                    mean_bias_sum = sum(mean_biases)
                    max_bias = max(max_biases)
                else:
                    loss = fusion([]) / args.gradient_accumulation
                    valid = 0
                    duration = 0.0
                    base_loss_sum = torch.tensor(0.0, device=device)
                    biased_loss_sum = torch.tensor(0.0, device=device)
                    weighted_loss_sum = torch.tensor(0.0, device=device)
                    margin_loss_sum = torch.tensor(0.0, device=device)
                    active_negative_loss_sum = torch.tensor(0.0, device=device)
                    override_loss_sum = torch.tensor(0.0, device=device)
                    bias_penalty_sum = torch.tensor(0.0, device=device)
                    positive_candidates = 0
                    candidate_rows = 0
                    mean_bias_sum = 0.0
                    max_bias = 0.0
                    hard_counts = {
                        "allowed_candidate_rows": 0,
                        "positive_words": 0,
                        "base_miss_words": 0,
                        "positive_token_positions": 0,
                        "base_miss_token_positions": 0,
                        "eligible_base_miss_positions": 0,
                        "base_correct_positions": 0,
                        "base_correct_override_candidates": 0,
                    }
                loss.backward()

            raw_count = len(rows)
            samples_seen += world_size * raw_count
            accumulation_valid += valid
            interval["raw"] += raw_count
            interval["valid"] += valid
            interval["failed"] += raw_count - valid
            interval["duration_seconds"] += duration
            interval["base_loss"] += float(base_loss_sum.detach().item())
            interval["biased_loss"] += float(biased_loss_sum.detach().item())
            interval["weighted_loss"] += float(weighted_loss_sum.detach().item())
            interval["margin_loss"] += float(margin_loss_sum.detach().item())
            interval["active_negative_loss"] += float(
                active_negative_loss_sum.detach().item()
            )
            interval["base_correct_override_loss"] += float(
                override_loss_sum.detach().item()
            )
            interval["bias_penalty"] += float(bias_penalty_sum.detach().item())
            interval["mean_bias"] += mean_bias_sum
            interval["max_bias"] = max(interval["max_bias"], max_bias)
            interval["positive_candidates"] += positive_candidates
            interval["candidate_rows"] += candidate_rows
            for key, value in hard_counts.items():
                interval[key] += value
            if error_messages and rank == 0:
                for error_message in error_messages[:3]:
                    print(f"[sample-error] {error_message}", flush=True)
            if not sync_now:
                continue

            global_accumulation_valid = torch.tensor(
                accumulation_valid, dtype=torch.int32, device=device
            )
            if world_size > 1:
                dist.all_reduce(global_accumulation_valid, op=dist.ReduceOp.SUM)
            if not global_accumulation_valid.item():
                raise RuntimeError(
                    "all ranks failed every sample in one gradient accumulation; "
                    "aborting instead of writing a zero-update checkpoint"
                )
            accumulation_valid = 0

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in fusion.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step % args.log_every == 0:
                keys = list(interval)
                packed = torch.tensor(
                    [interval[key] for key in keys],
                    dtype=torch.float64,
                    device=device,
                )
                if world_size > 1:
                    max_index = keys.index("max_bias")
                    max_value = packed[max_index].clone()
                    packed[max_index] = 0
                    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                    dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
                    packed[max_index] = max_value
                values = dict(zip(keys, packed.tolist()))
                peak = torch.tensor(
                    torch.cuda.max_memory_allocated(device) / 2**20,
                    dtype=torch.float64,
                    device=device,
                )
                if world_size > 1:
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                if rank == 0:
                    valid_count = max(values["valid"], 1.0)
                    elapsed = max(time.time() - interval_started, 1e-6)
                    log_row = {
                        "optimizer_step": optimizer_step,
                        "global_samples_seen": min(
                            samples_seen, selected_rows * args.epochs
                        ),
                        "interval_raw": int(values["raw"]),
                        "interval_valid": int(values["valid"]),
                        "interval_failed": int(values["failed"]),
                        "processed_hours": values["duration_seconds"] / 3600.0,
                        "base_loss": values["base_loss"] / valid_count,
                        "biased_loss": values["biased_loss"] / valid_count,
                        "weighted_loss": values["weighted_loss"] / valid_count,
                        "margin_loss": values["margin_loss"] / valid_count,
                        "active_negative_loss": (
                            values["active_negative_loss"] / valid_count
                        ),
                        "base_correct_override_loss": (
                            values["base_correct_override_loss"] / valid_count
                        ),
                        "bias_penalty": values["bias_penalty"] / valid_count,
                        "mean_bias": values["mean_bias"] / valid_count,
                        "max_bias": values["max_bias"],
                        "positive_candidate_rate": values["positive_candidates"]
                        / max(values["candidate_rows"], 1.0),
                        "allowed_candidate_rate": values["allowed_candidate_rows"]
                        / max(values["candidate_rows"], 1.0),
                        "positive_words": int(values["positive_words"]),
                        "base_miss_words": int(values["base_miss_words"]),
                        "positive_token_positions": int(
                            values["positive_token_positions"]
                        ),
                        "base_miss_token_positions": int(
                            values["base_miss_token_positions"]
                        ),
                        "eligible_base_miss_positions": int(
                            values["eligible_base_miss_positions"]
                        ),
                        "base_correct_positions": int(
                            values["base_correct_positions"]
                        ),
                        "base_correct_override_candidates": int(
                            values["base_correct_override_candidates"]
                        ),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "samples_per_second": values["raw"] / elapsed,
                        "gpu_peak_memory_mib": float(peak.item()),
                    }
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(log_row, ensure_ascii=False) + "\n"
                        )
                    print(
                        "[train] "
                        f"step={optimizer_step}/{total_steps} "
                        f"valid={log_row['interval_valid']}/{log_row['interval_raw']} "
                        f"base={log_row['base_loss']:.4f} "
                        f"biased={log_row['biased_loss']:.4f} "
                        f"weighted={log_row['weighted_loss']:.4f} "
                        f"hard={log_row['eligible_base_miss_positions']} "
                        f"bias={log_row['mean_bias']:.3f}/{log_row['max_bias']:.3f} "
                        f"speed={log_row['samples_per_second']:.2f}/s "
                        f"peak={log_row['gpu_peak_memory_mib']:.0f}MiB",
                        flush=True,
                    )
                interval = {key: 0.0 for key in interval}
                interval_started = time.time()
                torch.cuda.reset_peak_memory_stats(device)

            if optimizer_step % args.save_every == 0 and rank == 0:
                save_checkpoint(
                    args.output_dir / "checkpoint_latest.pt",
                    fusion=fusion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_step=optimizer_step,
                    samples_seen=min(samples_seen, selected_rows * args.epochs),
                    args=args,
                    selection=selection,
                )

            stop = torch.zeros(1, dtype=torch.int32, device=device)
            if args.max_optimizer_steps and optimizer_step >= args.max_optimizer_steps:
                stop.fill_(1)
            if rank == 0 and args.stop_file and args.stop_file.exists():
                stop.fill_(1)
            if world_size > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if rank == 0:
        final_name = (
            "checkpoint_final.pt"
            if optimizer_step >= total_steps and not args.smoke
            else "checkpoint_latest.pt"
        )
        save_checkpoint(
            args.output_dir / final_name,
            fusion=fusion,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_step=optimizer_step,
            samples_seen=min(samples_seen, selected_rows * args.epochs),
            args=args,
            selection=selection,
        )
        (args.output_dir / "status.txt").write_text(
            f"COMPLETE optimizer_step={optimizer_step} checkpoint={final_name}\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
