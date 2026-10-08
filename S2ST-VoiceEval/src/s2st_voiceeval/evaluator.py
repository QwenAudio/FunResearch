"""Load an exported model and score full utterances without padding or cropping."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from torchaudio.functional import resample
from transformers import WavLMConfig, WavLMModel

from ._model import MicrosoftUniSpeechWavLMECAPA, validate_official_wavlm_large_config

FORMAT = "s2st-voiceeval.full.v1"
SAMPLE_RATE = 16000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model(model: str | Path, revision: str | None = None,
                  local_files_only: bool = False) -> Path:
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if isinstance(model, Path) or path.is_absolute() or str(model).startswith((".", "~")):
        raise FileNotFoundError(f"Model directory not found: {path}")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=str(model), revision=revision, local_files_only=local_files_only,
        allow_patterns=["config.json", "model.safetensors", "README.md", "*LICENSE*"],
    ))


def read_audio(path: str | Path, *, allow_silence: bool = False,
               max_duration: float = 300.0) -> torch.Tensor:
    """Match the paper: average channels, resample to 16 kHz, no amplitude norm."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    info = sf.info(str(path))
    duration = info.frames / info.samplerate
    if not 0.25 <= duration <= max_duration:
        raise ValueError(
            f"{path}: expected 0.25–{max_duration:g} seconds, got {duration:.3f}. "
            "Provide an utterance rather than a full recording."
        )
    array, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(array.T.copy()).mean(dim=0)
    if not torch.isfinite(wave).all():
        raise ValueError(f"{path}: audio contains NaN or infinite samples")
    if not allow_silence and wave.square().mean() <= 1e-10:
        raise ValueError(f"{path}: the reference recording is silent")
    if rate != SAMPLE_RATE:
        wave = resample(wave, rate, SAMPLE_RATE)
    return wave.contiguous()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Evaluation used float64 cosine on float32 model embeddings.
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return float(np.clip(a @ b / denominator, -1.0, 1.0))


class VoiceEvaluator:
    """Reusable scorer taking a local export directory or a Hub repo ID.

    CUDA auto precision uses BF16 when available, matching paper extraction.
    CPU uses FP32. Each utterance is encoded separately, preserving its length.
    """

    def __init__(self, model: str | Path, *, device: str = "auto",
                 precision: str = "auto", revision: str | None = None,
                 local_files_only: bool = False, verify_checksum: bool = True):
        self.model_dir = resolve_model(model, revision, local_files_only)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: cpu, cuda, cuda:0, ...")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select device='cpu'")
        if precision not in ("auto", "fp32", "bf16"):
            raise ValueError("precision must be auto, fp32, or bf16")
        bf16_supported = False
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                bf16_supported = torch.cuda.is_bf16_supported()
        self.precision = (
            "bf16" if precision == "auto" and bf16_supported
            else "fp32" if precision == "auto" else precision
        )
        if self.precision == "bf16" and not bf16_supported:
            raise ValueError("BF16 requires a supported CUDA GPU; select fp32")

        config_path = self.model_dir / "config.json"
        weight_path = self.model_dir / "model.safetensors"
        if not config_path.is_file() or not weight_path.is_file():
            raise FileNotFoundError(
                f"{self.model_dir} must contain config.json and model.safetensors. "
                "Use the exported S2ST-VoiceEval model, not a training checkpoint."
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if self.config.get("format") != FORMAT:
            raise ValueError(f"Unsupported model format; expected {FORMAT}")
        if self.config.get("sample_rate") != SAMPLE_RATE:
            raise ValueError("This evaluator requires a 16 kHz model configuration")
        expected = self.config.get("weights_sha256")
        if verify_checksum:
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Model configuration is missing weights_sha256")
            if sha256_file(weight_path) != expected:
                raise ValueError("Model checksum mismatch; download the weights again")
        wavlm_config = WavLMConfig.from_dict(self.config["wavlm_config"])
        validate_official_wavlm_large_config(wavlm_config)
        with torch.device("meta"):
            encoder = MicrosoftUniSpeechWavLMECAPA(
                WavLMModel(wavlm_config), **self.config["speaker_config"],
            )
        state = load_file(str(weight_path), device="cpu")
        encoder.load_state_dict(state, strict=True, assign=True)
        self.encoder = encoder.to(self.device).eval().requires_grad_(False)
        del state

    @torch.inference_mode()
    def _embed(self, audio: str | Path, *, reference: bool) -> np.ndarray:
        wave = read_audio(audio, allow_silence=not reference)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16" else nullcontext()
        )
        with context:
            embedding = self.encoder(wave.unsqueeze(0).to(self.device))[0].float()
        if not torch.isfinite(embedding).all():
            raise RuntimeError(f"Non-finite embedding for {audio}")
        return embedding.cpu().numpy()

    def embed(self, audio: str | Path) -> np.ndarray:
        """Return one unnormalized 256-dimensional float32 embedding."""
        return self._embed(audio, reference=False)

    def score(self, source_audio: str | Path, candidate_audio: str | Path) -> float:
        """Return cosine similarity in [-1, 1]; higher means greater similarity."""
        return self.score_many(source_audio, [candidate_audio])[0]

    def score_many(self, source_audio: str | Path,
                   candidate_audios: Sequence[str | Path]) -> list[float]:
        """Return scores in input order; encode the reference only once."""
        if isinstance(candidate_audios, (str, Path)):
            raise TypeError("candidate_audios must be a sequence of audio paths")
        candidates = list(candidate_audios)
        if not candidates:
            raise ValueError("Provide at least one candidate recording")
        reference = self._embed(source_audio, reference=True)
        return [cosine(reference, self._embed(path, reference=False)) for path in candidates]

    def rank(self, source_audio: str | Path,
             candidate_audios: Sequence[str | Path]) -> list[dict]:
        """Return best-first records with path, score and zero-based input index.

        Exact score ties preserve input order. Rank is a display position, not a
        tie-breaking claim about which voice is better.
        """
        if isinstance(candidate_audios, (str, Path)):
            raise TypeError("candidate_audios must be a sequence of audio paths")
        candidates = list(candidate_audios)
        scores = self.score_many(source_audio, candidates)
        return [
            {"rank": rank, "index": index, "candidate": str(candidates[index]),
             "score": scores[index]}
            for rank, index in enumerate(
                sorted(range(len(scores)), key=lambda index: -scores[index]), 1
            )
        ]
