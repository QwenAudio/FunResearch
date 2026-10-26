"""Minimal offline vLLM backend used by the Route 1 positive pipeline."""

from __future__ import annotations

from typing import Any

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a meticulous multimodal data annotator. Report only content "
    "supported by the provided evidence."
)


class VllmOmniBatchRunner:
    def __init__(
        self,
        model_path: str,
        tp: int,
        max_num_seqs: int,
        max_model_len: int,
        submit_batch_size: int,
    ) -> None:
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_path, use_fast=False
        )
        self.submit_batch_size = submit_batch_size
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1},
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=1234,
        )

    def build_input(
        self,
        media_type: str,
        media_path: str,
        prompt: str,
        use_audio_in_video: bool,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": media_type, media_type: media_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=use_audio_in_video
        )
        value: dict[str, Any] = {
            "prompt": prompt_text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
        }
        if audios is not None:
            value["multi_modal_data"]["audio"] = audios
        if images is not None:
            value["multi_modal_data"]["image"] = images
        if videos is not None:
            value["multi_modal_data"]["video"] = videos
        return value

    def generate(
        self,
        requests: list[tuple[str, str, str, bool]],
        max_tokens: int,
    ) -> list[str]:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        results: list[str] = []
        for start in range(0, len(requests), self.submit_batch_size):
            chunk = requests[start : start + self.submit_batch_size]
            outputs = self.llm.generate(
                [self.build_input(*request) for request in chunk],
                sampling_params=params,
            )
            results.extend(result.outputs[0].text.strip() for result in outputs)
        return results

    def build_av_input(
        self, video_path: str, audio_path: str, prompt: str
    ) -> dict[str, Any]:
        """Attach the extracted WAV explicitly because some MP4s are silent."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        value: dict[str, Any] = {
            "prompt": prompt_text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {"use_audio_in_video": False},
        }
        if audios is not None:
            value["multi_modal_data"]["audio"] = audios
        if images is not None:
            value["multi_modal_data"]["image"] = images
        if videos is not None:
            value["multi_modal_data"]["video"] = videos
        return value

    def generate_av(
        self,
        requests: list[tuple[str, str, str]],
        max_tokens: int,
    ) -> list[str]:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        results: list[str] = []
        for start in range(0, len(requests), self.submit_batch_size):
            chunk = requests[start : start + self.submit_batch_size]
            outputs = self.llm.generate(
                [self.build_av_input(*request) for request in chunk],
                sampling_params=params,
            )
            results.extend(result.outputs[0].text.strip() for result in outputs)
        return results
