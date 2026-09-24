#!/usr/bin/env python3
"""Download pinned official files and build validated per-recording inputs."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from typing import Any
import wave

REPO_ID = "speechcolab/GigaSpeechBench"
REVISION = "680d3057641b7507a1ef14974407c7b0a7964e64"
DOMAINS = ("AGR", "AIT", "ART", "BIO", "ECM", "ENG", "ENT", "FIN", "HUM", "LAW", "MED", "MIL")
BASELINES = ("FunASR-Realtime", "Whisper-Large-v3")
EXPECTED = {"CH": 524, "EN": 387}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite different prepared data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def index_audios(value: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for row in value["audios"]:
        aid = row["aid"]
        if not re.fullmatch(r"[A-Z]{3}-(?:CH|EN)#[a-zA-Z0-9_-]+", aid):
            raise ValueError(f"unsafe or unsupported recording ID: {aid!r}")
        if aid in result:
            raise ValueError(f"duplicate recording ID: {aid}")
        result[aid] = row
    return result


def segment_index(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for segment in row["segments"]:
        sid = segment["sid"]
        parts = sid.rsplit("#", 2)
        if len(parts) != 3 or parts[0] != row["aid"]:
            raise ValueError(f"segment/recording ID mismatch: {sid}")
        begin, end = Decimal(segment["begin_time"]), Decimal(segment["end_time"])
        if not begin.is_finite() or not end.is_finite() or not 0 <= begin < end:
            raise ValueError(f"invalid interval: {sid}")
        if (Decimal(parts[1]), Decimal(parts[2])) != (begin, end):
            raise ValueError(f"SID/time mismatch: {sid}")
        if sid in result:
            raise ValueError(f"duplicate segment ID: {sid}")
        if not isinstance(segment["text"], str):
            raise ValueError(f"invalid transcript: {sid}")
        result[sid] = segment
    if not result:
        raise ValueError(f"empty recording: {row['aid']}")
    return result


def aligned_segments(reference: dict[str, Any], hypothesis: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if reference["aid"] != hypothesis["aid"]:
        raise ValueError("recording IDs differ")
    refs, hyps = segment_index(reference), segment_index(hypothesis)
    if refs.keys() != hyps.keys():
        raise ValueError(f"segment coverage mismatch: {reference['aid']}")
    prepared_refs, prepared_hyps = [], []
    for index, (sid, ref) in enumerate(refs.items()):
        hyp = hyps[sid]
        for key in ("begin_time", "end_time"):
            if Decimal(ref[key]) != Decimal(hyp[key]):
                raise ValueError(f"reference/baseline interval mismatch: {sid}")
        base = {"id": index, "sid": sid, "start": float(ref["begin_time"]), "end": float(ref["end_time"])}
        # Reference text, entities and speaker labels never enter the hyp file.
        prepared_hyps.append({**base, "text_original": hyp["text"], "text_working": hyp["text"], "text_final": hyp["text"]})
        prepared_refs.append({**base, "text_original": ref["text"], "text_working": ref["text"], "text_final": ref["text"], "entities": ref.get("entities", [])})
    return prepared_refs, prepared_hyps


def extract_audio(archive: Path) -> Path:
    destination = archive.parent / "extracted"
    marker = destination / ".archive.sha256"
    digest = sha256(archive)
    if marker.is_file():
        if marker.read_text().strip() != digest:
            raise ValueError(f"archive changed; choose a new snapshot directory: {archive}")
        return destination
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"unsafe archive member: {member.name}")
            if not (member.isfile() or member.isdir()) or member.issym() or member.islnk():
                raise ValueError(f"unsupported archive member: {member.name}")
            if member.isfile() and target.suffix.lower() != ".wav":
                raise ValueError(f"unexpected non-WAV archive member: {member.name}")
        handle.extractall(destination, members=members, filter="data")
    marker.write_text(digest + "\n", encoding="utf-8")
    return destination


def find_audio(domain_root: Path, aid: str) -> Path | None:
    candidates = [domain_root / "extracted" / "audio" / f"{aid}.wav",
                  domain_root / "extracted" / "audio" / "audio" / f"{aid}.wav",
                  domain_root / "extracted" / f"{aid}.wav",
                  domain_root / "audio" / f"{aid}.wav",
                  domain_root / "audio" / "audio" / f"{aid}.wav"]
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) > 1:
        raise ValueError(f"ambiguous audio location: {aid}")
    return next(iter(found), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="HF snapshot destination/root (not Vertical-Domain itself)")
    parser.add_argument("--output", type=Path, required=True, help="prepared Vertical-Domain root")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--languages", nargs="+", choices=EXPECTED, default=list(EXPECTED))
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=[BASELINES[0]])
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--audio-domains", nargs="+", default=[])
    audio_group.add_argument("--all-audio", action="store_true")
    args = parser.parse_args()
    snapshot, output = args.snapshot.resolve(), args.output.resolve()
    if snapshot == output or snapshot.is_relative_to(output) or output.is_relative_to(snapshot):
        raise SystemExit("snapshot and prepared output must be separate non-nested directories")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("use an immutable 40-character HF revision")
    domains = [f"{domain}-{language}" for language in args.languages for domain in DOMAINS]
    audio_domains = domains if args.all_audio else args.audio_domains
    if set(audio_domains) - set(domains):
        raise SystemExit("audio domains must belong to the requested languages")
    files = [f"Vertical-Domain/data/{d}/metadata.json" for d in domains]
    files += [f"Vertical-Domain/results/{b}.json" for b in args.baselines]
    files += [f"Vertical-Domain/data/{d}/audio.tar.gz" for d in audio_domains]
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=args.revision,
                          allow_patterns=files, local_dir=snapshot, max_workers=4)
    source_files = {}
    for name in files:
        path = snapshot / name
        if not path.is_file():
            raise SystemExit(f"missing official file: {name}; use --download")
        source_files[name] = sha256(path)
    for domain in audio_domains:
        extract_audio(snapshot / f"Vertical-Domain/data/{domain}/audio.tar.gz")
    hypotheses = {b: index_audios(load(snapshot / f"Vertical-Domain/results/{b}.json")) for b in args.baselines}
    counts = dict.fromkeys(args.languages, 0)
    manifest = []
    for domain in domains:
        source = snapshot / "Vertical-Domain/data" / domain
        references = index_audios(load(source / "metadata.json"))
        if any(not aid.startswith(domain + "#") for aid in references):
            raise ValueError(f"metadata domain mismatch: {domain}")
        for baseline in args.baselines:
            actual = {aid for aid in hypotheses[baseline] if aid.startswith(domain + "#")}
            if actual != references.keys():
                raise ValueError(f"recording coverage mismatch: {domain}/{baseline}")
        counts[domain[-2:]] += len(references)
        for aid, reference in references.items():
            audio = find_audio(source, aid)
            if domain in audio_domains and audio is None:
                raise ValueError(f"downloaded archive missing recording: {aid}")
            duration = max(float(s["end_time"]) for s in reference["segments"])
            audio_hash = None
            if audio:
                with wave.open(str(audio), "rb") as reader:
                    if reader.getcomptype() != "NONE":
                        raise ValueError(f"audio must be PCM WAV: {aid}")
                    if reader.getnframes() / reader.getframerate() + 0.1 < duration:
                        raise ValueError(f"audio shorter than segment timestamps: {aid}")
                audio_hash = sha256(audio)
            for baseline in args.baselines:
                refs, hyps = aligned_segments(reference, hypotheses[baseline][aid])
                model_root = output / domain / baseline
                common = {"dataset_name": f"gigaspeechbench_hf_Vertical-Domain_{domain}", "recording_id": aid,
                          "audio": {"path": f"audio/{aid}.wav", "duration_s": duration}}
                hyp_path, ref_path = model_root / "hyp" / f"{aid}.json", model_root / "ref" / f"{aid}.json"
                write_json(hyp_path, {**common, "source": {"repo_id": REPO_ID, "revision": args.revision, "model": baseline}, "segments": hyps})
                write_json(ref_path, {**common, "segments": refs})
                if audio:
                    target = model_root / "audio" / f"{aid}.wav"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() or target.is_symlink():
                        if target.resolve() != audio:
                            raise ValueError(f"refusing to replace audio link: {target}")
                    else:
                        target.symlink_to(os.path.relpath(audio, target.parent))
                manifest.append({"run_id": aid, "baseline": baseline, "segments": len(hyps), "audio_available": audio is not None,
                                 "audio_sha256": audio_hash, "hyp_sha256": sha256(hyp_path), "ref_sha256": sha256(ref_path)})
    if counts != {lang: EXPECTED[lang] for lang in args.languages}:
        raise ValueError(f"recording counts differ from frozen cohort: {counts}")
    report = {"repo_id": REPO_ID, "revision_declared": args.revision, "download_performed": args.download,
              "files_sha256": source_files, "recordings": counts, "baselines": args.baselines,
              "alignment": "recording ID + exact SID + decimal timestamps", "items": manifest}
    output.mkdir(parents=True, exist_ok=True)
    # Immutable manifests allow later audio additions without altering past provenance.
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:12]
    write_json(output / f"PREPARATION.{digest}.json", report)
    print(json.dumps({"recordings": counts, "segment_baseline_pairs": sum(r['segments'] for r in manifest),
                      "audio_recording_baseline_pairs": sum(r['audio_available'] for r in manifest),
                      "manifest": str(output / f"PREPARATION.{digest}.json")}, indent=2))


if __name__ == "__main__":
    main()
