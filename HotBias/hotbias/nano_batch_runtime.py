from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Sequence

import torch
from transformers import LogitsProcessorList


@dataclass
class PreparedNanoGenerationBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    audio_features: list[torch.Tensor]


@dataclass
class _FusionRequest:
    sample: dict[str, torch.Tensor]
    ready: threading.Event
    output: dict[str, torch.Tensor] | None = None
    error: BaseException | None = None


class _CoalescingFusionProxy:
    """Coalesce independent one-row Adapter calls into small tensor batches."""

    def __init__(
        self,
        fusion: Any,
        *,
        group_size: int,
        timeout_seconds: float = 0.01,
    ) -> None:
        self._fusion = fusion
        self._group_size = max(1, int(group_size))
        self._timeout_seconds = float(timeout_seconds)
        self._lock = threading.Lock()
        self._pending: list[_FusionRequest] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fusion, name)

    def set_group_size(self, value: int) -> None:
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot resize Fusion group with pending requests")
            self._group_size = max(1, int(value))

    @staticmethod
    def _raw_sample(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Static projections are row-shaped. Re-projecting their compact raw
        # inputs lets the Fusion module execute one padded batch instead of a
        # sequence of tiny kernels, which is faster at concurrent inference.
        output = {
            key: value
            for key, value in sample.items()
            if not key.startswith("projected_")
        }
        output["return_full_logits"] = False
        return output

    def _claim(self, request: _FusionRequest, *, force: bool) -> list[_FusionRequest]:
        with self._lock:
            position = next(
                (
                    index
                    for index, pending in enumerate(self._pending)
                    if pending is request
                ),
                None,
            )
            if position is None:
                return []
            if not force and len(self._pending) < self._group_size:
                return []
            count = min(self._group_size, len(self._pending))
            if force and position >= count:
                selected = list(range(count - 1)) + [position]
                group = [self._pending[index] for index in selected]
                selected_ids = {id(item) for item in group}
                self._pending = [
                    item for item in self._pending if id(item) not in selected_ids
                ]
            else:
                group = self._pending[:count]
                del self._pending[:count]
            return group

    def _execute(self, group: list[_FusionRequest]) -> None:
        try:
            outputs = self._fusion(
                [self._raw_sample(request.sample) for request in group]
            )
            if not isinstance(outputs, list) or len(outputs) != len(group):
                raise RuntimeError("coalesced Fusion returned an invalid batch")
            for request, output in zip(group, outputs):
                request.output = output
        except BaseException as error:
            for request in group:
                request.error = error
        finally:
            for request in group:
                request.ready.set()

    def __call__(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        if len(samples) != 1:
            return self._fusion(samples)
        request = _FusionRequest(sample=samples[0], ready=threading.Event())
        with self._lock:
            self._pending.append(request)
        group = self._claim(request, force=False)
        if group:
            self._execute(group)
        elif not request.ready.wait(self._timeout_seconds):
            group = self._claim(request, force=True)
            if group:
                self._execute(group)
        request.ready.wait()
        if request.error is not None:
            raise request.error
        if request.output is None:
            raise RuntimeError("coalesced Fusion request completed without output")
        return [request.output]


class BatchedStatefulLogitsProcessor:
    """Batch frozen decoder work while retaining one state machine per row."""

    def __init__(
        self,
        processors: Sequence[Any],
        hidden_capture: Any,
        *,
        eos_ids: set[int] | None = None,
    ) -> None:
        if not processors:
            raise ValueError("at least one row processor is required")
        self.processors = list(processors)
        self.hidden_capture = hidden_capture
        self.eos_ids = set(eos_ids or ())
        self.finished = [False] * len(self.processors)
        self._original_fusions: list[Any] = []
        self._fusion_proxy: _CoalescingFusionProxy | None = None
        worker_count = min(8, len(self.processors))
        fusions = [getattr(processor, "fusion", None) for processor in self.processors]
        if fusions and fusions[0] is not None and all(
            fusion is fusions[0] for fusion in fusions
        ):
            proxy = _CoalescingFusionProxy(fusions[0], group_size=worker_count)
            self._fusion_proxy = proxy
            self._original_fusions = fusions
            for processor in self.processors:
                processor.fusion = proxy
        self.executor = (
            ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="nano-adapter-row",
            )
            if len(self.processors) > 1
            else None
        )

    @staticmethod
    def _run_row(
        processor: Any,
        hidden: torch.Tensor,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> None:
        processor.hidden_capture.value = hidden
        processor(input_ids, scores)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        hidden = self.hidden_capture.value
        if hidden is None:
            raise RuntimeError("decoder hidden hook was not ready")
        if input_ids.shape[0] != len(self.processors):
            raise RuntimeError("decoder batch and stateful processors do not match")
        if scores.shape[0] != len(self.processors) or hidden.shape[0] != len(
            self.processors
        ):
            raise RuntimeError("decoder hidden/scores batch does not match processors")
        if input_ids.shape[1] > 0 and self.eos_ids:
            last_tokens = input_ids[:, -1].tolist()
            for index, token_id in enumerate(last_tokens):
                if int(token_id) in self.eos_ids:
                    self.finished[index] = True
        active = [
            index for index, is_finished in enumerate(self.finished) if not is_finished
        ]
        if not active:
            return scores
        if self._fusion_proxy is not None:
            self._fusion_proxy.set_group_size(min(8, len(active)))
        arguments = [
            (
                self.processors[index],
                hidden[index : index + 1],
                input_ids[index : index + 1],
                scores[index : index + 1],
            )
            for index in active
        ]
        if self.executor is None:
            self._run_row(*arguments[0])
        else:
            futures = [
                self.executor.submit(self._run_row, *arguments_row)
                for arguments_row in arguments
            ]
            for future in futures:
                future.result()
        return scores

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        if self._original_fusions:
            for processor, fusion in zip(self.processors, self._original_fusions):
                processor.fusion = fusion
            self._original_fusions = []
            self._fusion_proxy = None


def _load_generation_row(
    *,
    model,
    tokenizer,
    frontend,
    prompt: str,
    audio_path: Path,
) -> dict[str, torch.Tensor]:
    dialog = model.generate_chatml(prompt, str(audio_path))
    return model.data_load_speech(
        model.data_template(dialog),
        tokenizer,
        frontend,
        meta_data={},
        teacherforcing=False,
        batch_size=1,
        bf16=False,
        llm_dtype="bf16",
        disable_pbar=True,
    )


def prepare_nano_generation_batch(
    *,
    model,
    tokenizer,
    frontend,
    prompts: Sequence[str],
    audio_paths: Sequence[Path],
    device: torch.device,
    audio_workers: int = 0,
) -> PreparedNanoGenerationBatch:
    """Prepare several Nano multimodal prompts with one batched audio forward."""

    if not prompts or len(prompts) != len(audio_paths):
        raise ValueError("prompts and audio paths must be non-empty and aligned")
    arguments = [
        {
            "model": model,
            "tokenizer": tokenizer,
            "frontend": frontend,
            "prompt": prompt,
            "audio_path": Path(audio_path),
        }
        for prompt, audio_path in zip(prompts, audio_paths)
    ]
    if audio_workers > 1:
        with ThreadPoolExecutor(
            max_workers=audio_workers, thread_name_prefix="nano-infer-audio"
        ) as executor:
            batches = list(executor.map(lambda kwargs: _load_generation_row(**kwargs), arguments))
    else:
        batches = [_load_generation_row(**kwargs) for kwargs in arguments]

    speech_rows = [batch["speech"][0] for batch in batches]
    speech_lengths = torch.tensor(
        [int(batch["speech_lengths"][0, 0].item()) for batch in batches],
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

    source_rows = [batch["source_ids"][0].long() for batch in batches]
    maximum_length = max(int(row.numel()) for row in source_rows)
    padded_ids = torch.zeros(
        len(source_rows), maximum_length, dtype=torch.long
    )
    attention_mask = torch.zeros_like(padded_ids)
    offsets: list[int] = []
    for index, row in enumerate(source_rows):
        offset = maximum_length - int(row.numel())
        offsets.append(offset)
        clean = row.clone()
        clean[clean < 0] = 0
        padded_ids[index, offset:] = clean
        attention_mask[index, offset:] = 1
    if device.type == "cuda":
        padded_ids = padded_ids.pin_memory()
        attention_mask = attention_mask.pin_memory()
    padded_ids = padded_ids.to(device=device, non_blocking=True)
    attention_mask = attention_mask.to(device=device, non_blocking=True)
    embed_tokens = model.llm.get_input_embeddings()
    with torch.no_grad():
        inputs_embeds = embed_tokens(padded_ids)
        for index, batch in enumerate(batches):
            for turn_id in range(batch["fbank_beg"].shape[1]):
                begin = int(batch["fbank_beg"][0, turn_id].item())
                if begin <= 0:
                    continue
                fake_length = int(batch["fake_token_len"][0, turn_id].item())
                available = int(adaptor_out_lens[index].item())
                length = min(fake_length, available)
                begin += offsets[index]
                inputs_embeds[index, begin : begin + length] = adaptor_out[
                    index, :length
                ].to(inputs_embeds.dtype)

    audio_features = [
        adaptor_out[index, : int(adaptor_out_lens[index].item())]
        .detach()
        .float()
        for index in range(len(batches))
    ]
    return PreparedNanoGenerationBatch(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        audio_features=audio_features,
    )


def _configured_eos_ids(llm) -> set[int]:
    value = llm.generation_config.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def _trim_tokens(
    token_ids: Sequence[int], *, eos_ids: set[int], pad_id: int | None
) -> list[int]:
    output: list[int] = []
    for token_id in token_ids:
        value = int(token_id)
        if pad_id is not None and value == int(pad_id) and output:
            break
        output.append(value)
        if value in eos_ids:
            break
    return output


def generate_nano_batch(
    *,
    llm,
    tokenizer,
    prepared: PreparedNanoGenerationBatch,
    max_new_tokens: int,
    hidden_capture: Any | None = None,
    row_processors: Sequence[Any] | None = None,
) -> tuple[list[list[int]], list[str]]:
    """Greedy batched generation for Base or independent stateful adapters."""

    if row_processors is not None and len(row_processors) != prepared.inputs_embeds.shape[0]:
        raise ValueError("row processors must align with the prepared batch")
    hook_handle = None
    processors = None
    batched_processor = None
    if row_processors is not None:
        if hidden_capture is None:
            raise ValueError("adapter generation requires a hidden capture")
        batched_processor = BatchedStatefulLogitsProcessor(
            row_processors,
            hidden_capture,
            eos_ids=_configured_eos_ids(llm),
        )
        processors = LogitsProcessorList([batched_processor])
        hook_handle = llm.model.register_forward_hook(hidden_capture.hook)
    pad_id = llm.config.pad_token_id
    if pad_id is None:
        pad_id = llm.config.eos_token_id
    try:
        kwargs: dict[str, Any] = {}
        if processors is not None:
            kwargs["logits_processor"] = processors
        with torch.no_grad(), torch.autocast(
            device_type=prepared.inputs_embeds.device.type,
            dtype=torch.bfloat16,
            enabled=prepared.inputs_embeds.device.type == "cuda",
        ):
            generated = llm.generate(
                inputs_embeds=prepared.inputs_embeds.to(torch.bfloat16),
                attention_mask=prepared.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                output_router_logits=False,
                **kwargs,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if batched_processor is not None:
            batched_processor.close()
    eos_ids = _configured_eos_ids(llm)
    token_rows = [
        _trim_tokens(row.tolist(), eos_ids=eos_ids, pad_id=pad_id)
        for row in generated
    ]
    return token_rows, [
        tokenizer.decode(row, skip_special_tokens=True) for row in token_rows
    ]
