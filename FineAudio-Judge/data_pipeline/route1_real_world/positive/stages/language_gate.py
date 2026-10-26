#!/usr/bin/env python3
"""Whisper transcription plus independent VoxLingua window verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def normalize_language(value: str) -> str:
    code = value.split(":", 1)[0].strip().lower()
    return {"cmn": "zh", "zho": "zh", "eng": "en"}.get(code, code)


def reliable(segment: dict[str, Any], cfg: dict[str, Any]) -> bool:
    return (
        float(segment.get("avg_logprob") or -99) >= float(cfg["whisper_min_avg_logprob"])
        and float(segment.get("no_speech_prob") or 1) <= float(cfg["whisper_max_no_speech_probability"])
        and float(segment.get("end") or 0) - float(segment.get("start") or 0) >= float(cfg["min_window_seconds"])
        and bool(str(segment.get("text") or "").strip())
    )


def classify_window(classifier: Any, waveform: torch.Tensor) -> dict[str, Any]:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    with torch.no_grad():
        probability, score, index, label = classifier.classify_batch(waveform)
    label_value = label[0] if isinstance(label, (list, tuple)) else label
    if isinstance(label_value, bytes):
        label_value = label_value.decode("utf-8")
    values = probability.detach().float().cpu().reshape(-1)
    if values.numel() and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
        values = torch.softmax(values, dim=0)
    probability_value = (
        float(values.max())
        if values.numel()
        else float(score.reshape(-1)[0].detach().cpu())
    )
    return {
        "language": normalize_language(str(label_value)),
        "label": str(label_value),
        "probability": probability_value,
        "class_index": int(index.reshape(-1)[0].detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import librosa
    import soundfile as sf
    import whisper
    from speechbrain.inference.classifiers import EncoderClassifier

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = config["language_gate"]
    manifest = read_jsonl(args.run_root / "data" / args.dataset / "source_manifest.jsonl")
    whisper_path = args.run_root / "outputs" / f"{args.dataset}_whisper.jsonl"
    gate_path = args.run_root / "outputs" / f"{args.dataset}_language_gate.jsonl"
    if args.overwrite:
        whisper_path.unlink(missing_ok=True)
        gate_path.unlink(missing_ok=True)
    done = {str(row["id"]) for row in read_jsonl(gate_path)}

    whisper_model = whisper.load_model(cfg["whisper_model"], device=args.device)
    savedir = args.run_root / "models" / "voxlingua107"
    classifier = EncoderClassifier.from_hparams(
        source=cfg["voxlingua_model"],
        savedir=str(savedir),
        run_opts={"device": args.device},
    )
    allowed = set(cfg["allowed_languages"])
    threshold = float(cfg["voxlingua_min_probability"])

    for source in manifest:
        sample_id = str(source["id"])
        if sample_id in done:
            continue
        result = whisper_model.transcribe(
            source["audio_path"], task="transcribe", temperature=0.0,
            condition_on_previous_text=False, word_timestamps=True, verbose=False,
        )
        segments = []
        for segment in result.get("segments") or []:
            row = {
                "start": segment.get("start"), "end": segment.get("end"),
                "text": segment.get("text"), "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
                "words": segment.get("words") or [],
            }
            row["reliable"] = reliable(row, cfg)
            segments.append(row)
        whisper_row = {
            "id": sample_id, "dataset": source.get("dataset"),
            "audio_path": source["audio_path"], "model": cfg["whisper_model"],
            "language": normalize_language(str(result.get("language") or "uncertain")),
            "text": str(result.get("text") or "").strip(), "segments": segments,
        }
        append_jsonl(whisper_path, whisper_row)

        waveform, sample_rate = sf.read(source["audio_path"], dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        windows = []
        for segment in segments:
            if not segment["reliable"]:
                continue
            start, end = float(segment["start"]), float(segment["end"])
            clip = torch.from_numpy(waveform[int(start * sample_rate):int(end * sample_rate)]).float()
            if clip.numel() < int(float(cfg["min_window_seconds"]) * sample_rate):
                continue
            prediction = classify_window(classifier, clip.to(args.device))
            windows.append({"start": start, "end": end, **prediction})

        reliable_seconds = sum(float(row["end"]) - float(row["start"]) for row in segments if row["reliable"])
        confident_languages = {
            row["language"] for row in windows if float(row["probability"]) >= threshold
        }
        disallowed = sorted(confident_languages - allowed)
        allowed_detected = sorted(confident_languages & allowed)
        whisper_language = whisper_row["language"]
        voxlingua_mode = str(cfg.get("voxlingua_mode") or "audit")
        if reliable_seconds == 0:
            decision = "pass_no_reliable_speech"
        elif whisper_language not in allowed:
            decision = "reject_whisper_language"
        elif voxlingua_mode == "hard_any_window" and disallowed:
            decision = "reject_disallowed_language"
        elif voxlingua_mode == "hard_any_window" and not allowed_detected and windows:
            decision = "reject_voxlingua_uncertain"
        else:
            decision = "pass_zh_en"
        append_jsonl(gate_path, {
            "id": sample_id, "dataset": source.get("dataset"),
            "audio_path": source["audio_path"],
            "whisper_overall_language": whisper_language,
            "whisper_transcript": whisper_row["text"], "whisper_segments": segments,
            "reliable_speech_seconds": reliable_seconds,
            "voxlingua_windows": windows,
            "allowed_languages_detected": allowed_detected,
            "disallowed_languages_detected": disallowed,
            "language_policy_decision": decision,
            "language_threshold": threshold,
            "voxlingua_mode": voxlingua_mode,
            "provenance": {"whisper_model": cfg["whisper_model"], "voxlingua_model": cfg["voxlingua_model"]},
        })
        print(json.dumps({"id": sample_id, "decision": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
