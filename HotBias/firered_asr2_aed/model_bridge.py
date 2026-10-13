"""Frozen FireRedASR2-AED model bridge used by the HotBias adapter.

The model and the FireRedASR2S source tree are supplied by the caller. Nothing
in this module downloads weights or imports a private experiment directory.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types

import torch


def _source_directory(firered_repo: Path | None) -> Path:
    if firered_repo is None:
        spec = importlib.util.find_spec("fireredasr2s")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("Install FireRedASR2S or pass --firered-repo to its source checkout")
        package = Path(next(iter(spec.submodule_search_locations)))
    else:
        repository = Path(firered_repo).expanduser().resolve()
        package = repository / "fireredasr2s"
        if not package.is_dir():
            package = repository
    source = package / "fireredasr2"
    if not (source / "models/fireredasr_aed.py").is_file():
        raise FileNotFoundError(f"FireRedASR2S AED implementation not found under {source}")
    return source


def _component(source: Path, module: str, name: str):
    # Load only AED modules: the official package's aggregate imports can pull
    # optional VAD, LID, or LLM dependencies into this small training workflow.
    alias = "_hotbias_firered2_aed"
    package = sys.modules.get(alias)
    if package is None:
        package = types.ModuleType(alias)
        package.__path__ = [str(source)]
        package.__package__ = alias
        sys.modules[alias] = package
    elif package.__path__ != [str(source)]:
        raise RuntimeError("Only one FireRedASR2S source checkout may be loaded per process")
    return getattr(importlib.import_module(alias + "." + module), name)


class FrozenFireRed2Bridge:
    def __init__(self, model_dir: str | Path, firered_repo: str | Path | None = None, device: str = "cpu"):
        model_dir = Path(model_dir).expanduser().resolve()
        required = ("model.pth.tar", "cmvn.ark", "dict.txt", "train_bpe1000.model")
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"FireRedASR2-AED model directory {model_dir} lacks {missing}")
        source = _source_directory(None if firered_repo is None else Path(firered_repo))
        payload = torch.load(model_dir / "model.pth.tar", map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not {"args", "model_state_dict"} <= payload.keys():
            raise ValueError("Expected an official FireRedASR2-AED model.pth.tar")
        args = payload["args"]
        model_type = _component(source, "models.fireredasr_aed", "FireRedAsrAed")
        self.model = model_type.from_args(args)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
        if tuple(self.model.ctc.ctc_lo.weight.shape) != (8667, 1280):
            raise ValueError("This adapter expects the official 8667-token, 1280-dim FireRedASR2-AED")
        self.model.requires_grad_(False).eval().to(device)
        self.device = torch.device(device)
        feat_type = _component(source, "data.asr_feat", "ASRFeatExtractor")
        tok_type = _component(source, "tokenizer.aed_tokenizer", "ChineseCharEnglishSpmTokenizer")
        self.extractor = feat_type(str(model_dir / "cmvn.ark"))
        self.tokenizer = tok_type(str(model_dir / "dict.txt"), str(model_dir / "train_bpe1000.model"))
        self.args = args

    def tokenize(self, text: str):
        tokens, ids = self.tokenizer.tokenize(text)
        return {"tokens": tokens, "ids": ids,
                "unknown_positions": [i for i, value in enumerate(ids) if value == self.tokenizer.dict.unkid]}

    @torch.no_grad()
    def encode(self, audio_path: str | Path):
        import soundfile
        info = soundfile.info(audio_path)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError("Audio must be mono 16 kHz for training and inference")
        if info.duration > 60:
            raise ValueError("Audio over 60 s requires explicit segmentation")
        feats, lengths, durations, paths, _uttids = self.extractor(
            [str(audio_path)], [Path(audio_path).stem]
        )
        if feats is None or len(paths) != 1:
            raise RuntimeError("FireRed feature extraction failed")
        encoder, encoder_lengths, mask = self.model.encoder(
            feats.to(self.device), lengths.to(self.device)
        )
        ctc_log_probs = self.model.ctc(encoder)
        if not bool(torch.isfinite(encoder).all() and torch.isfinite(ctc_log_probs).all()):
            raise FloatingPointError("FireRed encoder or CTC head produced non-finite values")
        return {"encoder_outputs": encoder, "encoder_lengths": encoder_lengths,
                "encoder_mask": mask, "ctc_log_probs": ctc_log_probs,
                "duration_seconds": durations[0]}

    @torch.no_grad()
    def teacher_force(self, encoded, target_ids):
        decoder = self.model.decoder
        inputs = torch.tensor([[self.model.sos_id] + list(target_ids)], device=self.device, dtype=torch.long)
        target_mask = decoder.ignored_target_position_is_0(inputs, decoder.pad_id)
        embeddings = decoder.tgt_word_emb(inputs)
        hidden = decoder.dropout(embeddings * decoder.scale + decoder.positional_encoding(inputs))
        for layer in decoder.layer_stack:
            hidden = layer(hidden, encoded["encoder_outputs"], target_mask, encoded["encoder_mask"], cache=None)
        hidden = decoder.layer_norm_out(hidden)
        logits = decoder.tgt_word_prj(hidden)
        size = inputs.shape[1]
        future = torch.arange(size, device=self.device)[None, :] > torch.arange(size, device=self.device)[:, None]
        return {"base_logits": logits[0], "decoder_states": hidden[0],
                "audio_features": encoded["encoder_outputs"][0, :int(encoded["encoder_lengths"][0])],
                "history_features": embeddings[0], "history_attention_mask": future,
                "history_segment_ids": torch.ones(size, dtype=torch.long, device=self.device),
                "input_ids": inputs[0],
                "label_ids": torch.tensor(list(target_ids) + [self.model.eos_id], device=self.device)}
