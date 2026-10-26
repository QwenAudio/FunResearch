#!/usr/bin/env python3
"""Select additional speech clips unused by the Route 2 positives."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys


POSITIVE_STAGES = Path(__file__).resolve().parents[2] / "positive" / "stages"
sys.path.insert(0, str(POSITIVE_STAGES))
import build_speech_pool as speech_builder  # noqa: E402

norm_text = speech_builder.norm_text


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_key(seed: int, row: dict) -> str:
    value = f"{seed}:{row['source_dataset']}:{row['uid']}"
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-speech", type=Path, required=True)
    parser.add_argument("--zh-labels", type=Path, required=True)
    parser.add_argument("--zh-audio-root", type=Path, required=True)
    parser.add_argument("--paraspeechcaps-parquet", type=Path, required=True)
    parser.add_argument("--ears-audio-root", type=Path, required=True)
    parser.add_argument("--expresso-audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zh-count", type=int, default=800)
    parser.add_argument("--en-count", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--exclude-source-manifest", type=Path, action="append", default=[],
        help="Optional JSONL denylists for source IDs and paths; may be repeated.",
    )
    args = parser.parse_args()

    speech_builder.ZH_LABELS = args.zh_labels
    speech_builder.ZH_AUDIO = args.zh_audio_root
    speech_builder.PSC = args.paraspeechcaps_parquet
    speech_builder.EARS_AUDIO = args.ears_audio_root
    speech_builder.EXPRESSO_AUDIO = args.expresso_audio_root

    positive = read_jsonl(args.positive_speech)
    used_paths = {row["audio_path"] for row in positive}
    used_text = {norm_text(row["text"]) for row in positive}
    denied_ids, denied_paths = speech_builder.load_exclusions(args.exclude_source_manifest)

    # The positive selector is prefix-stable for a fixed seed. Selecting larger
    # pools with the original seed exposes additional candidates, which are then
    # filtered against every positive path and transcript.
    # 1,174 is the complete set passing the frozen Chinese quality gates
    # (speaker cap, unique transcript, duration, labels and profanity checks).
    zh = speech_builder.select_zh(20260902, 1174, allow_partial=True)
    for alternate_seed in range(20260903, 20260909):
        zh.extend(speech_builder.select_zh(alternate_seed, 800, allow_partial=True))
    # A different deterministic rank exposes unused EARS/EXPRESSO clips while
    # keeping exactly the same strict label, duration and per-speaker gates.
    # Two ranks provide margin without relaxing the quality rules.
    en = speech_builder.select_en(20260903, 600, 200, allow_partial=True) + speech_builder.select_en(
        20260904, 600, 200, allow_partial=True
    )
    candidates = [
        row for row in zh + en
        if row["audio_path"] not in used_paths
        and row["uid"] not in denied_ids
        and str(Path(row["audio_path"]).resolve()) not in denied_paths
    ]
    # Replacement allocation is longest-first, so retain the longest valid
    # unused sources first; the hash provides deterministic tie breaking.
    candidates.sort(key=lambda row: (-float(row["duration"]), stable_key(args.seed, row)))
    output = []
    for language, wanted in (("zh", args.zh_count), ("en", args.en_count)):
        language_candidates = [row for row in candidates if row["language"] == language]
        if language == "en":
            # Preserve as much of the original 3:1 EARS/EXPRESSO preference as
            # the globally unused pool allows. The original 600 EARS positives
            # leave at most 149 clips under the strict speaker cap.
            ears = [row for row in language_candidates if row["source_dataset"].endswith("/EARS")]
            expresso = [row for row in language_candidates if row["source_dataset"].endswith("/EXPRESSO")]
            language_candidates = ears + expresso
        selected = []
        local_paths = set()
        for row in language_candidates:
            if row["audio_path"] in local_paths:
                continue
            selected.append(row)
            local_paths.add(row["audio_path"])
            if len(selected) >= wanted:
                break
        if len(selected) != wanted:
            raise RuntimeError(f"only {len(selected)}/{wanted} unused {language} speech clips")
        output.extend(selected)
    random.Random(args.seed).shuffle(output)
    if len({row["audio_path"] for row in output}) != len(output):
        raise RuntimeError("duplicate speech paths in negative pool")
    if used_paths & {row["audio_path"] for row in output}:
        raise RuntimeError("positive speech leaked into negative pool")
    if any("instructttseval" in (row["source_dataset"] + row["audio_path"]).casefold() for row in output):
        raise RuntimeError("InstructTTSEval leakage")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    summary = {
        "rows": len(output),
        "languages": dict(Counter(row["language"] for row in output)),
        "sources": dict(Counter(row["source_dataset"] for row in output)),
        "unique_audio_paths": len({row["audio_path"] for row in output}),
        "positive_path_overlap": 0,
        "unique_transcripts": len({norm_text(row["text"]) for row in output}),
        "positive_transcript_overlap": len(used_text & {norm_text(row["text"]) for row in output}),
        "excluded_source_ids": len(denied_ids),
        "excluded_source_paths": len(denied_paths),
        "instructttseval_rows": 0,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
