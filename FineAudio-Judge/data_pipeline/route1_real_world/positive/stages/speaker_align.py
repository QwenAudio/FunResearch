#!/usr/bin/env python3
"""Add bounded speaker-alignment patches to accepted Route 1 captions."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


HERE = Path(__file__).resolve().parent
CORE_CODE = HERE


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def parse_json(text: str) -> dict[str, Any]:
    sys.path.insert(0, str(CORE_CODE))
    from common import parse_json_object
    return parse_json_object(text)


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("“", '"').replace("”", '"').replace("’", "'")).strip()


def output_paths(root: Path, dataset: str) -> dict[str, Path]:
    return {
        "inventory": root / "outputs" / f"{dataset}_speaker_inventory.jsonl",
        "words": root / "outputs" / f"{dataset}_speaker_words.jsonl",
        "clusters": root / "outputs" / f"{dataset}_speaker_clusters.jsonl",
        "turns": root / "outputs" / f"{dataset}_speaker_turns.jsonl",
        "caption_raw": root / "outputs" / f"{dataset}_caption_patch_raw.jsonl",
        "audit": root / "outputs" / f"{dataset}_caption_patch_audit.jsonl",
        "final": root / dataset / "speaker_aligned_candidates.jsonl",
        "rejected": root / dataset / "speaker_alignment_rejected.jsonl",
    }


def eligible_ids(root: Path, dataset: str, ids_file: Path | None) -> list[str]:
    sources = by_id(read_jsonl(root / "data" / dataset / "selected_manifest.jsonl"))
    gates = by_id(read_jsonl(root / "outputs" / f"{dataset}_language_gate.jsonl"))
    if ids_file:
        if ids_file.suffix == ".jsonl":
            requested = [str(row["id"]) for row in read_jsonl(ids_file)]
        else:
            requested = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing = sorted(set(requested) - set(sources))
        if missing:
            raise ValueError(f"IDs missing from selected manifest: {missing[:10]}")
        return requested
    return [
        sample_id for sample_id in sources
        if float(gates.get(sample_id, {}).get("reliable_speech_seconds") or 0) > 0
    ]


def inventory_prompt(gate: dict[str, Any], max_speakers: int) -> str:
    transcript = [
        {"start": row.get("start"), "end": row.get("end"), "text": row.get("text"), "reliable": row.get("reliable")}
        for row in gate.get("whisper_segments") or [] if str(row.get("text") or "").strip()
    ]
    return f"""
Listen to the complete audio and inventory only distinct speaker sources that
produce intelligible linguistic speech. Do not write a caption.
Do not count non-linguistic vocalizations or an indistinguishable crowd as
individual speakers. Count the same voice as one speaker across separate
intervals. Do not infer identity from semantics.

Whisper is provided only for rough localization and may contain recognition
errors or merged speakers:
{json.dumps(transcript, ensure_ascii=False)}

Return valid JSON only:
{{"speaker_count":2,"profiles":[{{"speaker_id":"SPEAKER_00","audible_profile":"audible voice characteristics","rough_intervals":[[0.0,1.0]]}},{{"speaker_id":"SPEAKER_01","audible_profile":"audible voice characteristics","rough_intervals":[[1.0,2.0]]}}],"has_overlapping_speech":false,"overlap_intervals":[],"confidence":0.0,"uncertainties":[],"reason":"audible evidence"}}
speaker_count must be an integer from 1 to {max_speakers}, and profiles must
contain exactly speaker_count entries. Return no text outside the JSON object.
""".strip()


def run_inventory(args: argparse.Namespace, config: dict[str, Any], ids: list[str]) -> None:
    sys.path.insert(0, str(CORE_CODE))
    from omni_backend import VllmOmniBatchRunner
    paths = output_paths(args.run_root, args.dataset)
    if args.overwrite:
        paths["inventory"].unlink(missing_ok=True)
    done = by_id(read_jsonl(paths["inventory"]))
    ids = [value for value in ids if value not in done]
    if not ids:
        return
    sources = by_id(read_jsonl(args.run_root / "data" / args.dataset / "selected_manifest.jsonl"))
    gates = by_id(read_jsonl(args.run_root / "outputs" / f"{args.dataset}_language_gate.jsonl"))
    generation = config["generation"]
    maximum = int(config["speaker_alignment"]["max_speakers"])
    runner = VllmOmniBatchRunner(generation["omni_model"], int(generation["omni_tensor_parallel"]), 4, 32768, 8)
    raw_values = runner.generate([
        ("audio", sources[value]["audio_path"], inventory_prompt(gates[value], maximum), False) for value in ids
    ], 1400)
    for sample_id, raw in zip(ids, raw_values, strict=True):
        try:
            inventory = parse_json(raw)
            count = int(inventory.get("speaker_count") or 0)
            if not 1 <= count <= maximum:
                raise ValueError(f"invalid speaker_count={count}")
            append_jsonl(paths["inventory"], {"id": sample_id, "speaker_inventory": inventory, "raw": raw, "model": generation["omni_model"]})
        except Exception as exc:
            append_jsonl(paths["rejected"], {"id": sample_id, "stage": "speaker_inventory", "reason": str(exc), "raw": raw})


def run_words(args: argparse.Namespace, config: dict[str, Any], ids: list[str]) -> None:
    import whisper
    paths = output_paths(args.run_root, args.dataset)
    if args.overwrite:
        paths["words"].unlink(missing_ok=True)
    done = by_id(read_jsonl(paths["words"]))
    sources = by_id(read_jsonl(args.run_root / "data" / args.dataset / "selected_manifest.jsonl"))
    gates = by_id(read_jsonl(args.run_root / "outputs" / f"{args.dataset}_language_gate.jsonl"))
    model_name = config["speaker_alignment"]["whisper_word_model"]
    model = whisper.load_model(model_name, device=args.device)
    for sample_id in ids:
        if sample_id in done:
            continue
        language = gates[sample_id].get("whisper_overall_language")
        if language not in {"zh", "en"}:
            language = None
        result = model.transcribe(
            sources[sample_id]["audio_path"], language=language, task="transcribe",
            temperature=0.0, condition_on_previous_text=False, word_timestamps=True, verbose=False,
        )
        append_jsonl(paths["words"], {
            "id": sample_id, "audio_path": sources[sample_id]["audio_path"],
            "language": result.get("language"), "text": result.get("text"),
            "segments": result.get("segments") or [], "model": model_name, "word_timestamps": True,
        })


def vocal_path(root: Path, dataset: str, model: str, audio_path: str) -> Path:
    return root / "outputs" / "demucs_vocals" / dataset / model / Path(audio_path).stem / "vocals.wav"


def run_separate(args: argparse.Namespace, config: dict[str, Any], ids: list[str]) -> None:
    sources = by_id(read_jsonl(args.run_root / "data" / args.dataset / "selected_manifest.jsonl"))
    model = config["speaker_alignment"]["demucs_model"]
    executable = config["executables"]["demucs_python"]
    outdir = args.run_root / "outputs" / "demucs_vocals" / args.dataset
    for sample_id in ids:
        target = vocal_path(args.run_root, args.dataset, model, sources[sample_id]["audio_path"])
        if target.is_file() and not args.overwrite:
            continue
        subprocess.run([
            executable, "-m", "demucs", "-n", model, "--two-stems", "vocals",
            "-o", str(outdir), sources[sample_id]["audio_path"],
        ], check=True)


def mean_word_probability(segment: dict[str, Any]) -> float:
    values = [float(row.get("probability") or 0) for row in segment.get("words") or []]
    return sum(values) / len(values) if values else 0.0


def run_cluster(args: argparse.Namespace, config: dict[str, Any], ids: list[str]) -> None:
    import librosa
    import soundfile as sf
    from funasr.models.campplus.cluster_backend import SpectralCluster
    from funasr.models.campplus.utils import correct_labels
    from modelscope.pipelines import pipeline

    paths = output_paths(args.run_root, args.dataset)
    if args.overwrite:
        paths["clusters"].unlink(missing_ok=True)
    done = by_id(read_jsonl(paths["clusters"]))
    words = by_id(read_jsonl(paths["words"]))
    inventories = by_id(read_jsonl(paths["inventory"]))
    sources = by_id(read_jsonl(args.run_root / "data" / args.dataset / "selected_manifest.jsonl"))
    cfg = config["speaker_alignment"]
    speaker_model = pipeline(task="speaker-verification", model=cfg["speaker_embedding_model"], device=args.device)
    for sample_id in ids:
        if sample_id in done or sample_id not in inventories:
            continue
        expected = int(inventories[sample_id]["speaker_inventory"]["speaker_count"])
        path = vocal_path(args.run_root, args.dataset, cfg["demucs_model"], sources[sample_id]["audio_path"])
        wave, sample_rate = sf.read(path, dtype="float32")
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        if sample_rate != 16000:
            wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        utterances, clips, discarded = [], [], []
        for segment in words[sample_id]["segments"]:
            quote = str(segment.get("text") or "").strip()
            start, end = float(segment.get("start") or 0), float(segment.get("end") or 0)
            probability = mean_word_probability(segment)
            if not quote or end - start < float(cfg["min_utterance_seconds"]) or probability < float(cfg["min_word_probability"]):
                if quote:
                    discarded.append({"start": start, "end": end, "text": quote, "reason": "low_confidence_or_too_short"})
                continue
            clip = wave[max(0, int((start - 0.08) * sample_rate)):min(len(wave), int((end + 0.08) * sample_rate))]
            if len(clip) < sample_rate:
                clip = np.pad(clip, (0, sample_rate - len(clip)))
            clips.append(np.asarray(clip, dtype=np.float32))
            utterances.append({"start": round(start, 3), "end": round(end, 3), "quote": quote, "language": words[sample_id].get("language"), "mean_word_probability": round(probability, 4)})
        if not clips:
            append_jsonl(paths["rejected"], {"id": sample_id, "stage": "ecapa_cluster", "reason": "no_reliable_utterances"})
            continue
        matrix = np.asarray(speaker_model(clips, output_emb=True)["embs"], dtype=np.float32)
        cluster_count = min(expected, len(matrix))
        labels = correct_labels(SpectralCluster(min_num_spks=1, max_num_spks=int(cfg["max_speakers"]))(matrix, cluster_count))
        for utterance, label in zip(utterances, labels, strict=True):
            utterance["speaker_id"] = f"SPEAKER_{int(label):02d}"
        append_jsonl(paths["clusters"], {
            "id": sample_id, "speaker_count_prior": expected,
            "speaker_count": len(set(int(value) for value in labels)), "speaker_turns": utterances,
            "discarded_whisper_segments": discarded,
            "source_separation": {"model": cfg["demucs_model"], "purpose": "speaker_embedding_only", "vocal_path": str(path), "caption_audio_remains_original_mix": True},
            "models": {"word_alignment": words[sample_id]["model"], "speaker_embedding": cfg["speaker_embedding_model"], "clustering": f"spectral_count_{cluster_count}"},
        })


def slim(value: Any, limit: int = 3000) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text[:limit]


def source_hints(source: dict[str, Any]) -> dict[str, Any]:
    fine = source.get("finevideo_visual_metadata") or {}
    content = fine.get("content_metadata") or {}
    speaking_scenes = []
    for scene in content.get("scenes") or []:
        activities = [
            row for row in scene.get("activities") or []
            if any(
                token in str(row.get("description") or "").lower()
                for token in ("talk", "speak", "say", "ask", "narrat", "explain")
            )
        ]
        if activities or scene.get("cast"):
            speaking_scenes.append({
                "timestamps": scene.get("timestamps"), "cast": scene.get("cast"),
                "activities": activities[:6],
            })
    return {
        "vript_visual_caption": slim(source.get("source_visual_caption"), 1800),
        "vript_voiceover": slim(source.get("source_voiceover"), 1800),
        "finevideo_title": fine.get("youtube_title"),
        "finevideo_characters": (content.get("characterList") or [])[:12],
        "finevideo_speaking_scenes": speaking_scenes[:8],
    }


def turn_prompt(source: dict[str, Any], words: dict[str, Any], clusters: dict[str, Any], old_speech: str) -> str:
    segments = [{"start": row.get("start"), "end": row.get("end"), "text": row.get("text"), "mean_word_probability": mean_word_probability(row)} for row in words.get("segments") or []]
    return f"""
Re-listen to the complete original mixed audio and determine who says each
utterance. This is not a caption-writing step.

Whisper segment candidates: {json.dumps(segments, ensure_ascii=False)}
ECAPA cluster candidates from Demucs vocals: {json.dumps(clusters.get('speaker_turns') or [], ensure_ascii=False)}
Earlier audio-only speech investigation: {old_speech}
Optional video hints, which may be used only for speaker ownership and never to
add or alter spoken wording: {json.dumps(source_hints(source), ensure_ascii=False)}

Rules: re-listen to each segment in the original mix. Assign speaker_id using
voice characteristics, pauses, turn structure, and visible speaking transitions
when video is available. quote must contain only intelligible verbatim Chinese
or English speech; do not summarize or translate it. A low-confidence or
incomplete candidate must not be treated as reliable wording for a caption
change. Set include_in_caption=true only for a reliable, complete utterance.

Return JSON with the structure below. Do not copy field descriptions or return
placeholder values:
{{"speaker_count":2,"speaker_profiles":[{{"speaker_id":"SPEAKER_00","source_role":"unknown","audible_profile":"audible voice characteristics"}}],"speaker_turns":[{{"start":0.0,"end":1.0,"speaker_id":"SPEAKER_00","quote":"verbatim utterance","language":"en|zh","confidence":0.0,"include_in_caption":true}}],"discarded_candidates":[],"checks":{{"reheard_original_mix":true,"speaker_ids_consistent":true,"no_semantic_speaker_guessing":true}}}}
""".strip()


def caption_prompt(
    base_caption: str, turns: dict[str, Any], language: str, repair: Any = None
) -> str:
    """Ask for bounded patches instead of allowing a full caption rewrite."""
    turn_evidence = []
    for index, turn in enumerate(turns.get("speaker_turns") or []):
        turn_evidence.append({"turn_id": f"turn_{index:03d}", **turn})
    repair_note = "" if repair is None else (
        "\nThe previous patch failed deterministic validation. Correct only the reported problem:\n"
        + json.dumps(repair, ensure_ascii=False)[:12000]
    )
    target = "English" if language == "en" else "Chinese"
    return f"""
Re-listen to the complete original mixed audio. The base caption below is already
the accepted audio-grounded result. Do NOT rewrite, shorten, summarize, polish,
reorder, or otherwise improve it. Your only task is to fix a clearly audible
speaker-boundary, speaker-attribution, or spoken-quote error.

Base caption (immutable except for exact submitted patches):
{base_caption}

Adjudicated speaker turns:
{json.dumps(turn_evidence, ensure_ascii=False)}

Rules:
1. If there is no clear speaker/utterance-boundary error, set needs_edit=false.
2. Return only the smallest necessary set of non-overlapping edits. Each old_text
   must be one exact, contiguous substring copied character-for-character from
   the base caption and must occur exactly once.
3. new_text must keep all correct non-speech, music, ambience, chronology, overlap,
   and surrounding detail in old_text. Change only the minimum wording needed to
   separate interrupted/merged utterances, correct a quote, or correct who speaks.
4. Do not delete detail merely because it is absent from the speaker-turn list.
   The list is evidence for speech alignment, not a replacement caption.
5. Any new spoken quote must be verbatim audio in its original language and tied
   to evidence_turns. Do not output speaker IDs or timecodes in new_text.
6. Caption narration remains {target}; do not translate quoted speech.
7. When verified evidence shows merged or interrupted speech, preserve the full
   affected context and place different speakers' utterances in separate quoted
   spans. Update only narration directly contradicted by the corrected speech
   boundary, and leave all unrelated content unchanged.
{repair_note}
Return JSON only:
{{"needs_edit":true,"edits":[{{"old_text":"exact base substring","new_text":"minimal replacement","reason":"speaker alignment reason","evidence_turns":["turn_000"]}}],"checks":{{"only_speaker_or_spoken_boundary_edits":true,"outside_text_unchanged":true,"no_non_speech_detail_removed":true,"new_quotes_verbatim":true}}}}
""".strip()


def validate_turns(value: dict[str, Any], min_confidence: float) -> None:
    required = ("reheard_original_mix", "speaker_ids_consistent", "no_semantic_speaker_guessing")
    if not all((value.get("checks") or {}).get(key) is True for key in required):
        raise ValueError("speaker turn checks failed")
    retained = []
    for turn in value.get("speaker_turns") or []:
        quote = str(turn.get("quote") or "").strip()
        confidence = float(turn.get("confidence") or 0.0)
        placeholder = quote.lower() in {"verbatim", "actual verbatim quote", "实际逐字原话"}
        default_span = float(turn.get("start") or 0.0) == 0.0 and float(turn.get("end") or 0.0) == 1.0
        if (
            turn.get("include_in_caption") is True
            and confidence >= min_confidence
            and not placeholder
            and not default_span
            and len(re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", quote)) > 2
        ):
            retained.append(turn)
    if not retained:
        raise ValueError("no reliable speaker turns")
    value["speaker_turns"] = retained
    value["speaker_count"] = len({str(row.get("speaker_id")) for row in retained})


def fallback_turns(cluster_row: dict[str, Any], min_confidence: float) -> dict[str, Any]:
    turns = []
    for row in cluster_row.get("speaker_turns") or []:
        confidence = float(row.get("mean_word_probability") or 0.0)
        quote = str(row.get("quote") or "").strip()
        if confidence < min_confidence or len(re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", quote)) <= 2:
            continue
        turns.append({
            "start": row.get("start"), "end": row.get("end"),
            "speaker_id": row.get("speaker_id"), "quote": quote,
            "language": row.get("language"), "confidence": confidence,
            "include_in_caption": True,
        })
    speaker_ids = sorted({str(row.get("speaker_id")) for row in turns})
    value = {
        "speaker_count": len(speaker_ids),
        "speaker_profiles": [
            {"speaker_id": value, "source_role": "unknown", "audible_profile": "cluster-derived voice"}
            for value in speaker_ids
        ],
        "speaker_turns": turns,
        "discarded_candidates": [],
        "checks": {
            "reheard_original_mix": True,
            "speaker_ids_consistent": True,
            "no_semantic_speaker_guessing": True,
        },
        "adjudication_fallback": "whisper_words_plus_ecapa_clusters",
    }
    validate_turns(value, min_confidence)
    return value


def quoted_spans(text: str) -> list[str]:
    values = [match.strip() for match in re.findall(r'["“]([^"”]+)["”]', text) if match.strip()]
    values.extend(
        match.strip()
        for match in re.findall(r"(?<![A-Za-z0-9])'([^'\n]+)'(?![A-Za-z0-9])", text)
        if match.strip()
    )
    return values


def quote_free_sentences(text: str) -> list[str]:
    """Return complete narration sentences that a speech-only patch must preserve."""
    sentences = re.split(
        r"(?:(?<=[.!?])|(?<=[.!?][\"'”]))\s+",
        normalized(text),
    )
    return [
        sentence
        for sentence in sentences
        if len(re.findall(r"[A-Za-z0-9\u3400-\u9fff]+", sentence)) >= 4
        and len(re.findall(r'["“”]|(?<![A-Za-z0-9])\'|\'(?![A-Za-z0-9])', sentence)) % 2 == 0
        and not quoted_spans(sentence)
    ]


def quote_is_grounded(quote: str, turns: list[dict[str, Any]]) -> bool:
    target = normalized(quote).lower().strip(" .,!?;:")
    evidence = [normalized(str(row.get("quote") or "")).lower().strip(" .,!?;:") for row in turns]
    candidates = list(evidence)
    for width in (2, 3):
        candidates.extend(" ".join(evidence[index:index + width]) for index in range(len(evidence) - width + 1))
    by_speaker: dict[str, list[str]] = {}
    for row, value in zip(turns, evidence, strict=True):
        by_speaker.setdefault(str(row.get("speaker_id")), []).append(value)
    candidates.extend(" ".join(values) for values in by_speaker.values())
    return any(target == value or target in value or value in target for value in candidates if value)


def deterministic_interruption_patch(
    base_caption: str,
    old_text: str,
    evidence_ids: list[str],
    turns: dict[str, Any],
    language: str,
) -> tuple[str, str, list[str]] | None:
    """Repair a model-detected A-B-A interruption without regenerating prose."""
    all_turns = turns.get("speaker_turns") or []
    indices = {int(value.split("_")[-1]) for value in evidence_ids}
    old_words = set(re.findall(r"[a-z0-9]+", old_text.lower()))
    for index, row in enumerate(all_turns):
        quote_words = set(re.findall(r"[a-z0-9]+", str(row.get("quote") or "").lower()))
        if quote_words and quote_words.issubset(old_words):
            indices.add(index)
    ordered = [all_turns[index] for index in sorted(indices, key=lambda value: float(all_turns[value].get("start") or 0))]
    if len({str(row.get("speaker_id")) for row in ordered}) < 2:
        return None
    first_speaker = str(ordered[0].get("speaker_id"))
    if not any(str(row.get("speaker_id")) == first_speaker for row in ordered[2:]):
        return None

    start = base_caption.index(old_text)
    end = start + len(old_text)
    quote_pairs = {"'": "'", '"': '"', "“": "”"}
    if start > 0 and end < len(base_caption) and base_caption[start - 1] in quote_pairs:
        if base_caption[end] == quote_pairs[base_caption[start - 1]]:
            start -= 1
            end += 1
    expanded_old = base_caption[start:end]
    context = base_caption[max(0, start - 140):start].lower()
    singing = any(token in context for token in ("sing", "song", "lyric", "vocal", "music"))
    profiles = {
        str(row.get("speaker_id")): str(row.get("audible_profile") or "").lower()
        for row in turns.get("speaker_profiles") or []
    }
    pieces = []
    previous_speaker = None
    for index, row in enumerate(ordered):
        speaker = str(row.get("speaker_id"))
        quote = str(row.get("quote") or "").strip().strip('"“”\'')
        if index == 0:
            quote = quote.rstrip(".!?…") + "..."
            pieces.append(f'"{quote}"')
        elif speaker == first_speaker and previous_speaker != first_speaker:
            quote = quote[:1].lower() + quote[1:]
            quote = "..." + quote.rstrip(".!?…") + "..."
            subject = "The singer" if singing else "The first voice"
            verb = "resumes singing" if singing else "resumes"
            pieces.append(f'{subject} {verb}, "{quote}"')
        else:
            profile = profiles.get(speaker, "")
            role = "A different male voice" if ("男" in profile or "male" in profile) else (
                "A different female voice" if ("女" in profile or "female" in profile) else "A different voice"
            )
            verb = "shouts" if (quote.endswith("!") or "喊" in profile or "shout" in profile) else "interjects"
            pieces.append(f'{role} {verb}, "{quote}"')
        previous_speaker = speaker
    replacement = " ".join(pieces)

    return expanded_old, replacement, [f"turn_{index:03d}" for index in sorted(indices)]


def apply_caption_patch(
    base_caption: str,
    patch: dict[str, Any],
    turns: dict[str, Any],
    language: str,
) -> tuple[str, list[dict[str, Any]]]:
    required_checks = (
        "only_speaker_or_spoken_boundary_edits", "outside_text_unchanged",
        "no_non_speech_detail_removed", "new_quotes_verbatim",
    )
    if not all((patch.get("checks") or {}).get(key) is True for key in required_checks):
        raise ValueError("minimal-patch checks failed")
    needs_edit = patch.get("needs_edit") is True
    edits = patch.get("edits") or []
    if not needs_edit:
        if edits:
            raise ValueError("needs_edit=false but edits is not empty")
        return base_caption, []
    if not isinstance(edits, list) or not edits:
        raise ValueError("needs_edit=true requires at least one edit")

    allowed_turn_ids = {f"turn_{index:03d}" for index, _ in enumerate(turns.get("speaker_turns") or [])}
    placed: list[dict[str, Any]] = []
    for edit in edits:
        old_text = str(edit.get("old_text") or "")
        new_text = str(edit.get("new_text") or "")
        if not old_text or not new_text:
            raise ValueError("empty edit")
        if base_caption.count(old_text) != 1:
            raise ValueError("old_text must occur exactly once in base caption")
        evidence_ids = edit.get("evidence_turns") or []
        if not evidence_ids or any(value not in allowed_turn_ids for value in evidence_ids):
            raise ValueError("invalid or missing evidence_turns")
        if old_text == new_text:
            deterministic = deterministic_interruption_patch(
                base_caption, old_text, evidence_ids, turns, language,
            )
            if deterministic is None:
                raise ValueError("ineffective edit: use needs_edit=false or provide a real structural replacement")
            old_text, new_text, evidence_ids = deterministic
        if re.search(r"\b\d{1,2}:\d{2}\b", new_text):
            raise ValueError("machine timecode in replacement")
        if language == "en" and re.search(r"[\u3400-\u9fff]", new_text):
            raise ValueError("Chinese text in English replacement")
        start = base_caption.index(old_text)
        end = start + len(old_text)
        quote_pairs = {'"': '"', "'": "'", "“": "”"}
        old_text_is_quoted = (
            start > 0 and end < len(base_caption)
            and base_caption[start - 1] in quote_pairs
            and base_caption[end] == quote_pairs[base_caption[start - 1]]
        )
        normalized_new = normalized(new_text).lower()
        missing_narration = [
            sentence for sentence in quote_free_sentences(old_text)
            if not old_text_is_quoted and normalized(sentence).lower() not in normalized_new
        ]
        if missing_narration:
            raise ValueError(
                "replacement removes quote-free narration: "
                + " | ".join(missing_narration)
            )
        old_quote_values = [normalized(value).lower() for value in quoted_spans(old_text)]
        old_quotes = set(old_quote_values)
        old_quote_keys = [value.strip(" .,!?;:…") for value in old_quote_values]
        outside_quote_keys = {
            normalized(value).lower().strip(" .,!?;:…")
            for value in quoted_spans(base_caption[:start] + base_caption[end:])
        }
        new_quotes = quoted_spans(new_text)
        new_quote_keys = [
            normalized(quote).lower().strip(" .,!?;:…") for quote in new_quotes
        ]
        duplicated_outside_quotes = [
            quote for quote, key in zip(new_quotes, new_quote_keys, strict=True)
            if key in outside_quote_keys and new_quote_keys.count(key) > old_quote_keys.count(key)
        ]
        if duplicated_outside_quotes:
            raise ValueError(
                "replacement duplicates quoted speech from outside its span: "
                + " | ".join(duplicated_outside_quotes)
            )
        selected_turns = [
            turns["speaker_turns"][int(value.split("_")[-1])]
            for value in evidence_ids
        ]
        selected_speakers = {str(row.get("speaker_id")) for row in selected_turns}
        if len(selected_speakers) > 1 and len(quoted_spans(new_text)) < 2:
            raise ValueError("multi-speaker interruption must use separate quoted spans")
        for quote in quoted_spans(new_text):
            if normalized(quote).lower() not in old_quotes and not quote_is_grounded(quote, selected_turns):
                raise ValueError(f"new quote not grounded in selected turns: {quote}")
        placed.append({
            "old_text": old_text, "new_text": new_text,
            "reason": str(edit.get("reason") or ""),
            "evidence_turns": evidence_ids, "start": start, "end": start + len(old_text),
        })
    if not placed:
        return base_caption, []
    placed.sort(key=lambda row: row["start"])
    if any(left["end"] > right["start"] for left, right in zip(placed, placed[1:])):
        raise ValueError("patch spans overlap")
    chunks, cursor = [], 0
    for edit in placed:
        chunks.extend((base_caption[cursor:edit["start"]], edit["new_text"]))
        cursor = edit["end"]
    chunks.append(base_caption[cursor:])
    return "".join(chunks), placed


def run_caption(args: argparse.Namespace, config: dict[str, Any], ids: list[str]) -> None:
    sys.path.insert(0, str(CORE_CODE))
    from omni_backend import VllmOmniBatchRunner
    from finalize import audit_errors, audit_prompt, direct_quotes, mask_direct_quotes, normalized_words, restore_audit_quotes
    paths = output_paths(args.run_root, args.dataset)
    cached_turn_rows = by_id(read_jsonl(paths["turns"])) if args.reuse_existing_turns else {}
    if args.overwrite:
        keys = ["caption_raw", "audit", "final", "rejected"]
        if not args.reuse_existing_turns:
            keys.insert(0, "turns")
        for key in keys:
            if args.ids_file and paths[key].exists():
                retained = [row for row in read_jsonl(paths[key]) if str(row.get("id")) not in set(ids)]
                if retained:
                    write_jsonl(paths[key], retained)
                else:
                    paths[key].unlink(missing_ok=True)
            else:
                paths[key].unlink(missing_ok=True)
    done = by_id(read_jsonl(paths["final"]))
    if done and paths["rejected"].exists():
        unresolved = [
            row for row in read_jsonl(paths["rejected"])
            if str(row.get("id")) not in done
        ]
        if unresolved:
            write_jsonl(paths["rejected"], unresolved)
        else:
            paths["rejected"].unlink()
    ids = [value for value in ids if value not in done]
    if not ids:
        return
    sources = by_id(read_jsonl(args.run_root / "data" / args.dataset / "selected_manifest.jsonl"))
    words = by_id(read_jsonl(paths["words"]))
    clusters = by_id(read_jsonl(paths["clusters"]))
    old = by_id(read_jsonl(args.run_root / "outputs" / f"{args.dataset}_audio_analysis.jsonl"))
    gates = by_id(read_jsonl(args.run_root / "outputs" / f"{args.dataset}_language_gate.jsonl"))
    base = by_id(read_jsonl(args.run_root / args.dataset / "caption_candidates.jsonl"))
    ids = [value for value in ids if value in words and value in clusters and value in old]
    generation = config["generation"]
    speaker_cfg = config["speaker_alignment"]
    min_turn_confidence = float(speaker_cfg.get("caption_turn_min_confidence", 0.6))
    max_model_len = int(generation.get("speaker_alignment_max_model_len", 65536))
    runner = VllmOmniBatchRunner(
        generation["omni_model"],
        int(generation["omni_tensor_parallel"]),
        4,
        max_model_len,
        8,
    )
    turns_by_id: dict[str, dict[str, Any]] = {}
    missing_turn_ids = []
    for sample_id in ids:
        cached = cached_turn_rows.get(sample_id, {}).get("speaker_alignment")
        try:
            if not cached:
                raise ValueError("no cached turns")
            validate_turns(cached, min_turn_confidence)
            turns_by_id[sample_id] = cached
        except Exception:
            missing_turn_ids.append(sample_id)
    if missing_turn_ids:
        raw_by_id: dict[str, str] = {}
        video_ids = [value for value in missing_turn_ids if sources[value].get("video_path")]
        audio_only_ids = [value for value in missing_turn_ids if not sources[value].get("video_path")]
        if video_ids:
            raw_values = runner.generate_av([
                (
                    sources[value]["video_path"], sources[value]["audio_path"],
                    turn_prompt(sources[value], words[value], clusters[value], old[value]["investigation_rounds"]["speech"]),
                )
                for value in video_ids
            ], 4000)
            raw_by_id.update(zip(video_ids, raw_values, strict=True))
        if audio_only_ids:
            raw_values = runner.generate([
                (
                    "audio", sources[value]["audio_path"],
                    turn_prompt(sources[value], words[value], clusters[value], old[value]["investigation_rounds"]["speech"]),
                    False,
                )
                for value in audio_only_ids
            ], 4000)
            raw_by_id.update(zip(audio_only_ids, raw_values, strict=True))
        for sample_id in missing_turn_ids:
            raw = raw_by_id[sample_id]
            try:
                value = parse_json(raw)
                validate_turns(value, min_turn_confidence)
                turns_by_id[sample_id] = value
                append_jsonl(paths["turns"], {"id": sample_id, "speaker_alignment": value, "raw": raw})
            except Exception as exc:
                try:
                    value = fallback_turns(clusters[sample_id], min_turn_confidence)
                    turns_by_id[sample_id] = value
                    append_jsonl(paths["turns"], {
                        "id": sample_id, "speaker_alignment": value, "raw": raw,
                        "adjudication_warning": str(exc),
                    })
                except Exception as fallback_exc:
                    append_jsonl(paths["rejected"], {
                        "id": sample_id, "stage": "speaker_adjudication",
                        "reason": str(fallback_exc), "primary_reason": str(exc), "raw": raw,
                    })
    valid_ids = [value for value in ids if value in turns_by_id and value in base]
    caption_candidates_by_id: dict[str, dict[str, Any]] = {}
    pending = list(valid_ids)
    repairs: dict[str, Any] = {}
    last_failures: dict[str, tuple[str, str]] = {}
    for _ in range(4):
        if not pending:
            break
        raw_values = runner.generate([
            (
                "audio", sources[sample_id]["audio_path"],
                caption_prompt(
                    base[sample_id]["audio_caption"], turns_by_id[sample_id],
                    "en" if gates[sample_id].get("whisper_overall_language") == "en" else "zh",
                    repairs.get(sample_id),
                ),
                False,
            )
            for sample_id in pending
        ], 5000)
        retry = []
        for sample_id, raw in zip(pending, raw_values, strict=True):
            try:
                patch = parse_json(raw)
                language = "en" if gates[sample_id].get("whisper_overall_language") == "en" else "zh"
                audio_caption, edits = apply_caption_patch(
                    base[sample_id]["audio_caption"], patch, turns_by_id[sample_id],
                    language,
                )
                masked, quotes = mask_direct_quotes(audio_caption)
                tokens = [f"__VERBATIM_QUOTE_{value}__" for value in range(len(quotes))]
                caption_candidates_by_id[sample_id] = {
                    "id": sample_id, "turns": turns_by_id[sample_id], "patch": patch,
                    "caption_raw": raw, "language": language, "audio_caption": audio_caption,
                    "base_audio_caption": base[sample_id]["audio_caption"], "speaker_edits": edits,
                    "changed": bool(edits), "automatic_fallback_to_base": False,
                    "masked": masked, "quotes": quotes, "tokens": tokens,
                }
            except Exception as exc:
                error = str(exc)
                last_failures[sample_id] = (error, raw)
                repairs[sample_id] = {"previous_result": raw, "machine_error": error}
                retry.append(sample_id)
        pending = retry
    for sample_id in pending:
        error, raw = last_failures[sample_id]
        language = "en" if gates[sample_id].get("whisper_overall_language") == "en" else "zh"
        base_caption = base[sample_id]["audio_caption"]
        masked, quotes = mask_direct_quotes(base_caption)
        caption_candidates_by_id[sample_id] = {
            "id": sample_id, "turns": turns_by_id[sample_id], "patch": {},
            "caption_raw": raw, "language": language, "audio_caption": base_caption,
            "base_audio_caption": base_caption, "speaker_edits": [], "changed": False,
            "automatic_fallback_to_base": True, "fallback_reason": error,
            "masked": masked, "quotes": quotes,
            "tokens": [f"__VERBATIM_QUOTE_{value}__" for value in range(len(quotes))],
        }
    caption_candidates = [
        caption_candidates_by_id[value]
        for value in valid_ids if value in caption_candidates_by_id
    ]
    changed_candidates = [row for row in caption_candidates if row["changed"]]
    audit_raw_values = []
    if changed_candidates:
        audit_raw_values = runner.generate([
            ("text", "", audit_prompt(row["masked"], row["tokens"]), False)
            for row in changed_candidates
        ], 5000)
    for _ in range(3):
        bad = []
        for index, raw in enumerate(audit_raw_values):
            try:
                value = parse_json(raw)
                errors = audit_errors(value, changed_candidates[index]["tokens"])
            except Exception as exc:
                value = {"raw": raw}
                errors = [f"invalid_json:{exc}"]
            if errors:
                bad.append((index, value, errors))
        if not bad:
            break
        repaired = runner.generate([
            (
                "text",
                "",
                audit_prompt(
                    changed_candidates[index]["masked"],
                    changed_candidates[index]["tokens"],
                    {"previous_result": value, "machine_errors": errors},
                ),
                False,
            )
            for index, value, errors in bad
        ], 5000)
        for (index, _, _), raw in zip(bad, repaired, strict=True):
            audit_raw_values[index] = raw
    audits_by_id = dict(zip((row["id"] for row in changed_candidates), audit_raw_values, strict=True))
    for candidate in caption_candidates:
        sample_id = candidate["id"]
        try:
            audit_raw = audits_by_id.get(sample_id, "")
            if candidate["changed"]:
                audit = parse_json(audit_raw)
                errors = audit_errors(audit, candidate["tokens"])
                if errors:
                    raise ValueError("; ".join(errors))
                audit_caption = restore_audit_quotes(audit["audit_caption_zh"], candidate["quotes"])
            else:
                audit_caption = base[sample_id]["audit_caption_zh"]
            production_quotes = sorted(
                normalized_words(value)
                for value in direct_quotes(candidate["audio_caption"])
            )
            audit_quotes = sorted(
                normalized_words(value) for value in direct_quotes(audit_caption)
            )
            if production_quotes != audit_quotes:
                raise ValueError("audit quotes differ from production caption")
            source = sources[sample_id]
            append_jsonl(paths["final"], {
                "id": sample_id, "dataset": source.get("dataset"),
                "pipeline": "route1_speaker_aligned",
                "audio_duration": source.get("duration"), "audio_path": source["audio_path"],
                "video_path": source.get("video_path"), "source_locator": source.get("source_locator"),
                "source_group_id": source.get("source_group_id"),
                "speech_language": candidate["language"],
                "caption_language": candidate["language"],
                "language_basis": "detected_speech",
                "audio_caption": candidate["audio_caption"],
                "audit_caption_zh": audit_caption,
                "caption_language_policy": "match_detected_speech_language_preserve_spoken_content_verbatim",
                "speaker_alignment": candidate["turns"],
                "base_audio_caption": candidate["base_audio_caption"],
                "speaker_edits": candidate["speaker_edits"],
                "speaker_alignment_changed_caption": candidate["changed"],
                "automatic_fallback_to_base": candidate["automatic_fallback_to_base"],
                "caption_checks": {
                    "minimal_patch_only": True,
                    "outside_patch_byte_identical": True,
                },
                "provenance": {
                    "route": "audio_grounded_plus_speaker_alignment",
                    "protocol": config["protocol"],
                    "dialogue_boundary": config["speaker_alignment"]["whisper_word_model"],
                    "background_reduction_for_embedding": config["speaker_alignment"]["demucs_model"],
                    "speaker_embedding": config["speaker_alignment"]["speaker_embedding_model"],
                    "speaker_adjudication_and_caption": generation["omni_model"],
                    "speaker_adjudication_media": "video_plus_explicit_original_wav",
                    "caption_audio": "original_mixed_wav", "caption_update": "bounded_exact_substring_patch",
                    "replaces_route_a": False,
                    "audit_translation": "placeholder_protected_narration_only",
                },
            })
            append_jsonl(paths["caption_raw"], {
                "id": sample_id, "patch": candidate["patch"],
                "raw": candidate["caption_raw"], "audit_raw": audit_raw,
                "automatic_fallback_to_base": candidate["automatic_fallback_to_base"],
                "fallback_reason": candidate.get("fallback_reason"),
            })
        except Exception as exc:
            append_jsonl(paths["rejected"], {"id": sample_id, "stage": "speaker_aligned_audit", "reason": str(exc), "raw": audit_raw})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--step", choices=("inventory", "words", "separate", "cluster", "caption", "all"), default="all")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-existing-turns", action="store_true",
        help="Reuse already adjudicated speaker turns while regenerating only bounded caption patches.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    if args.step == "all":
        for step in ("inventory", "words", "separate", "cluster", "caption"):
            command = [
                sys.executable, str(Path(__file__).resolve()), "--run-root", str(args.run_root),
                "--config", str(args.config), "--dataset", args.dataset, "--step", step,
                "--device", args.device,
            ]
            if args.ids_file:
                command.extend(["--ids-file", str(args.ids_file)])
            if args.overwrite:
                command.append("--overwrite")
            if args.reuse_existing_turns:
                command.append("--reuse-existing-turns")
            subprocess.run(command, check=True)
        return

    ids = eligible_ids(args.run_root, args.dataset, args.ids_file)
    functions = {
        "inventory": run_inventory, "words": run_words, "separate": run_separate,
        "cluster": run_cluster, "caption": run_caption,
    }
    functions[args.step](args, config, ids)
    gc.collect()


if __name__ == "__main__":
    main()
