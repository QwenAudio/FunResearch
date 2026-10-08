#!/usr/bin/env python3
"""Merge a trusted XLSV v2 delta with its exact base into a portable export."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from s2st_voiceeval._model import MicrosoftUniSpeechWavLMECAPA
from s2st_voiceeval.evaluator import FORMAT, sha256_file


def export(checkpoint: Path, base_checkpoint: Path, wavlm_config: Path,
           output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_sha = sha256_file(checkpoint)
    base_sha = sha256_file(base_checkpoint)
    config_file = wavlm_config / "config.json" if wavlm_config.is_dir() else wavlm_config
    configuration_sha = sha256_file(config_file)
    # This one known metadata type occurs in older project checkpoints.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if saved.get("checkpoint_format") != "xlsv.trainable_delta.v2":
        raise ValueError("Expected a version-2 XLSV trainable delta")
    provenance = saved["encoder_provenance"]
    if provenance["base_sha256"] != base_sha:
        raise ValueError("Base checkpoint does not match the delta's recorded SHA256")
    if provenance["configuration_sha256"] != configuration_sha:
        raise ValueError("WavLM configuration does not match the delta's recorded SHA256")
    config = saved["config"]["model"]
    if config["architecture"] != "microsoft_unispeech_wavlm_ecapa":
        raise ValueError("Expected Microsoft WavLM + ECAPA architecture")
    speaker_config = {
        "embedding_dim": int(config["embedding_dim"]),
        "channels": int(config["ecapa_channels"]),
        "normalize_waveforms": bool(config["normalize_waveforms"]),
        "normalize_embeddings": bool(config["normalize_embeddings"]),
    }
    model, converted = MicrosoftUniSpeechWavLMECAPA.from_official_checkpoint(
        base_checkpoint, config_file,
        last_n_layers=int(config["trainable_transformer_layers"]), **speaker_config,
    )
    del converted
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            expected.update(f"{name}.{key}" for key, _ in module.named_buffers(recurse=False))
    actual = set(saved["encoder"])
    if actual != expected:
        raise ValueError(
            f"Incomplete delta: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    model.load_state_dict(saved["encoder"], strict=False)
    model.eval()
    # All tensors, including frozen frontend parameters and BN statistics, are exported.
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    backbone_config = model.wavlm.config.to_dict()
    backbone_config.pop("_name_or_path", None)
    metadata = {
        "format": FORMAT,
        "model_name": "S2ST-VoiceEval BIA+ALA+Rank",
        "sample_rate": 16000,
        "embedding_dim": model.embedding_dim,
        "speaker_config": speaker_config,
        "wavlm_config": backbone_config,
        "paper_inference_precision": "cuda-bf16",
        "preprocessing": {
            "channels": "arithmetic mean",
            "resampling": "torchaudio.functional.resample defaults",
            "waveform_normalization": False,
            "utterance_handling": "full utterances, individually encoded; no padding or cropping",
            "cosine_dtype": "float64",
        },
        "provenance": {
            "source_checkpoint_sha256": source_sha,
            "base_checkpoint_sha256": base_sha,
            "source_configuration_sha256": configuration_sha,
            "encoder_update": int(saved["step"]),
            "source": "https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".export-") as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        weights = staging / "model.safetensors"
        save_file(state, str(weights), metadata={"format": "pt"})
        metadata["weights_sha256"] = sha256_file(weights)
        (staging / "config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (staging / "SHA256SUMS").write_text(
            f"{metadata['weights_sha256']}  model.safetensors\n"
            f"{sha256_file(staging / 'config.json')}  config.json\n", encoding="utf-8",
        )
        package_root = Path(__file__).resolve().parents[1]
        shutil.copy2(package_root / "MODEL_CARD.md", staging / "README.md")
        shutil.copy2(package_root / "NOTICE.md", staging / "NOTICE.md")
        shutil.copy2(package_root / "src/s2st_voiceeval/UPSTREAM_LICENSE", staging / "UPSTREAM_LICENSE")
        if sha256_file(checkpoint) != source_sha:
            raise RuntimeError("Source checkpoint changed during export")
        staging.rename(output)
    return {"format": FORMAT, "tensor_count": len(state),
            "weights_bytes": (output / "model.safetensors").stat().st_size,
            "weights_sha256": metadata["weights_sha256"],
            "source_checkpoint_sha256": source_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--wavlm-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.base_checkpoint,
                            args.wavlm_config, args.output), indent=2))


if __name__ == "__main__":
    main()
