#!/usr/bin/env python3
"""Build a deterministic Route 2 speech manifest at a user-selected scale."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


PROFANITY = re.compile(
    r"\b(fuck|fucking|shit|bitch|cunt|wanker|wankers|asshole)\b|"
    r"操你|草你|他妈的|妈的|傻逼|煞笔|狗日的",
    re.IGNORECASE,
)

ZH_LABELS: Path
ZH_AUDIO: Path
PSC: Path
EARS_AUDIO: Path
EXPRESSO_AUDIO: Path
EXCLUDED_IDS: set[str] = set()
EXCLUDED_PATHS: set[str] = set()


def load_exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    source_ids: set[str] = set()
    source_paths: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key in ("source_id", "uid", "video_id", "original_source_id", "replacement_source_id"):
                if value.get(key) is not None:
                    source_ids.add(str(value[key]))
            if value.get("source_path") is not None:
                source_paths.add(str(Path(str(value["source_path"])).resolve()))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                visit(json.loads(line))
    return source_ids, source_paths


def stable_rank(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:16], 16)


def norm_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def zh_phrase(gender: str, selected: list[dict[str, str]], text: str) -> tuple[str, str]:
    person = "一名女性" if gender == "女" else "一名男性"
    parts = []
    for item in selected:
        name, value = item["name"], item["value"]
        if name == "speed":
            parts.append({"快": "语速较快", "中": "语速适中", "慢": "语速缓慢"}.get(value, f"语速{value}"))
        elif name == "pitch":
            parts.append({"高": "音调较高", "中": "音调适中", "低": "音调较低"}.get(value, f"音调{value}"))
        elif name == "volume":
            parts.append({"高": "音量较高", "中": "音量适中", "低": "音量较低"}.get(value, f"音量{value}"))
    description = person + "以" + "、".join(parts) + "的方式说话"
    return description, f"{description}：“{text}。”"


def english_phrase(gender: str, selected: list[dict[str, str]], text: str) -> tuple[str, str]:
    person = "A woman" if gender == "female" else "A man"
    accents = {"american", "russian", "ukrainian", "german", "mandarin", "spanish"}
    values = []
    for item in selected:
        value = item["value"].strip()
        lowered = value.casefold()
        if lowered in accents:
            rendered = f"an {value.title()} accent"
        elif lowered == "high-pitched":
            rendered = "a high pitch"
        elif lowered == "medium-pitched":
            rendered = "a medium pitch"
        elif lowered == "low-pitched":
            rendered = "a low pitch"
        elif lowered.endswith(" speed"):
            rendered = f"a {lowered.removesuffix(' speed')} pace"
        elif item["name"] == "situational_style":
            rendered = f"a {value} delivery"
        else:
            rendered = f"a {value} voice"
        item["caption_text"] = rendered
        values.append(rendered)
    description = f"{person} speaks with " + ", ".join(values[:-1])
    if len(values) > 1:
        description += (" and " if len(values) == 2 else ", and ") + values[-1]
    else:
        description += values[0]
    return description, f'{description}, saying, "{text}"'


def select_zh(seed: int, count: int, *, allow_partial: bool = False) -> list[dict]:
    candidates = []
    speaker_use = Counter()
    text_seen = set()
    with ZH_LABELS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row["Key"]
            if not uid.startswith("aishell_") or row["Gender"] not in {"男", "女"}:
                continue
            text = row["Transcript"].strip()
            if not text or PROFANITY.search(text):
                continue
            speaker_id = uid.removeprefix("aishell_")[:7]
            audio_path = ZH_AUDIO / speaker_id / f"{uid.removeprefix('aishell_')}.wav"
            if uid in EXCLUDED_IDS or str(audio_path.resolve()) in EXCLUDED_PATHS:
                continue
            if not audio_path.exists():
                continue
            available = [
                {"name": "speed", "value": row["Speed"].strip()},
                {"name": "pitch", "value": row["Pitch"].strip()},
                {"name": "volume", "value": row["Energy"].strip()},
            ]
            if any(not item["value"] for item in available):
                continue
            candidates.append((uid, speaker_id, audio_path, text, row, available))
    candidates.sort(key=lambda item: stable_rank(seed, item[0]))
    output = []
    for uid, speaker_id, audio_path, text, row, available in candidates:
        if len(output) >= count:
            break
        normalized = norm_text(text)
        if normalized in text_seen or speaker_use[speaker_id] >= 8:
            continue
        duration = float(sf.info(audio_path).duration)
        if not 2.0 <= duration <= 15.0:
            continue
        rng = random.Random(stable_rank(seed + 17, uid))
        attr_count = 2 if rng.random() < 0.70 else 3
        selected = rng.sample(available, attr_count)
        description, prompt = zh_phrase(row["Gender"], selected, text)
        output.append({
            "uid": uid,
            "language": "zh",
            "source_dataset": "SpeechCraft/AISHELL-3",
            "audio_path": str(audio_path),
            "duration": round(duration, 4),
            "text": text,
            "speaker_id": speaker_id,
            "speaker": "女性" if row["Gender"] == "女" else "男性",
            "speaker_description": description,
            "style_prompt": prompt,
            "speech_caption_phrase": description,
            "selected_caption_attrs": selected,
            "all_source_attrs": {
                "gender": row["Gender"], "age": row["Age"], "speed": row["Speed"],
                "pitch": row["Pitch"], "volume": row["Energy"], "emotion": row["Emotion"],
            },
            "primary_attrs": {
                "gender": "female" if row["Gender"] == "女" else "male",
                **{item["name"]: item["value"] for item in selected},
            },
            "annotation_format": "SpeechCraft structured CSV labels; deterministic short caption",
        })
        text_seen.add(normalized)
        speaker_use[speaker_id] += 1
    if len(output) != count and not allow_partial:
        raise RuntimeError(f"Only selected {len(output)}/{count} Chinese clips")
    return output


def strict_ears(path: str) -> bool:
    name = Path(path).name
    if re.fullmatch(r"emo_.+_sentences\.wav", name):
        return "neutral" not in name
    if name.startswith(("rainbow_", "sentences_")):
        return any(f"_{style}.wav" in name for style in ("fast", "slow", "highpitch", "lowpitch", "loud", "whisper"))
    return False


def strict_expresso(path: str) -> bool:
    if not path.startswith("audio_48khz/read/"):
        return False
    return any(f"/{style}/" in path for style in ("confused", "enunciated", "happy", "laughing", "sad", "whisper"))


def select_en(
    seed: int, ears_count: int, expresso_count: int, *, allow_partial: bool = False
) -> list[dict]:
    table = pq.read_table(PSC)
    pools = {"ears": [], "expresso": []}
    for row in table.to_pylist():
        source = row["source"]
        relative = row["relative_audio_path"]
        if source == "ears" and strict_ears(relative):
            audio_path = EARS_AUDIO / relative
        elif source == "expresso" and strict_expresso(relative):
            audio_path = EXPRESSO_AUDIO / relative
        else:
            continue
        uid = f"paraspeechcaps_{source}_{relative}"
        if uid in EXCLUDED_IDS or str(audio_path.resolve()) in EXCLUDED_PATHS:
            continue
        text = (row.get("transcription") or "").strip()
        duration = float(row.get("duration") or 0.0)
        if not text or PROFANITY.search(text) or not 2.0 <= duration <= 15.0 or not audio_path.exists():
            continue
        pools[source].append((row, audio_path))

    output = []
    for source, wanted, speaker_cap in (("ears", ears_count, 7), ("expresso", expresso_count, 50)):
        candidates = sorted(pools[source], key=lambda item: stable_rank(seed, source + ":" + item[0]["relative_audio_path"]))
        speaker_use = Counter()
        audio_seen = set()
        for row, audio_path in candidates:
            if sum(item["source_dataset"].endswith(source.upper() if source == "ears" else "EXPRESSO") for item in output) >= wanted:
                break
            speaker_id = row["speakerid"]
            if speaker_use[speaker_id] >= speaker_cap or str(audio_path) in audio_seen:
                continue
            rng = random.Random(stable_rank(seed + 23, row["relative_audio_path"]))
            situational = [x for x in (row.get("situational_tags") or []) if x]
            intrinsic = [x for x in (row.get("intrinsic_tags") or []) if x and x.lower() not in {"male", "female"}]
            basics = [x for x in (row.get("basic_tags") or []) if x and "environment" not in x.lower() and x.lower() not in {"male", "female"}]
            choices = []
            if situational:
                choices.append({"name": "situational_style", "value": rng.choice(situational)})
            remainder = [
                *({"name": "intrinsic_style", "value": x} for x in intrinsic),
                *({"name": "basic_style", "value": x} for x in basics),
            ]
            rng.shuffle(remainder)
            attr_count = 2 if rng.random() < 0.70 else 3
            used_values = {item["value"] for item in choices}
            for item in remainder:
                if item["value"] in used_values:
                    continue
                choices.append(item); used_values.add(item["value"])
                if len(choices) >= attr_count:
                    break
            if len(choices) < 2:
                continue
            choices = choices[:attr_count]
            gender = row.get("gender") if row.get("gender") in {"male", "female"} else "unknown"
            description, prompt = english_phrase(gender, choices, row["transcription"].strip())
            output.append({
                "uid": f"paraspeechcaps_{source}_{row['relative_audio_path']}",
                "language": "en",
                "source_dataset": "ParaSpeechCaps-Base/EARS" if source == "ears" else "ParaSpeechCaps-Base/EXPRESSO",
                "audio_path": str(audio_path),
                "duration": round(float(sf.info(audio_path).duration), 4),
                "text": row["transcription"].strip(),
                "speaker_id": speaker_id,
                "speaker": "woman" if gender == "female" else ("man" if gender == "male" else "speaker"),
                "speaker_description": description,
                "style_prompt": prompt,
                "speech_caption_phrase": description,
                "selected_caption_attrs": choices,
                "all_source_attrs": {
                    "intrinsic_tags": row.get("intrinsic_tags"),
                    "situational_tags": row.get("situational_tags"),
                    "basic_tags": row.get("basic_tags"),
                    "all_tags": row.get("all_tags"),
                },
                "primary_attrs": {"gender": gender, **{item["name"]: item["value"] for item in choices}},
                "original_description": row.get("text_description"),
                "annotation_format": "ParaSpeechCaps-Base structured tag arrays plus natural descriptions",
            })
            speaker_use[speaker_id] += 1
            audio_seen.add(str(audio_path))
        actual = sum(item["source_dataset"].endswith(source.upper() if source == "ears" else "EXPRESSO") for item in output)
        if actual != wanted and not allow_partial:
            raise RuntimeError(f"Only selected {actual}/{wanted} {source} clips")
    return output


def main() -> None:
    global ZH_LABELS, ZH_AUDIO, PSC, EARS_AUDIO, EXPRESSO_AUDIO, EXCLUDED_IDS, EXCLUDED_PATHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--zh-labels", type=Path, required=True)
    parser.add_argument("--zh-audio-root", type=Path, required=True)
    parser.add_argument("--paraspeechcaps-parquet", type=Path, required=True)
    parser.add_argument("--ears-audio-root", type=Path, required=True)
    parser.add_argument("--expresso-audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--zh-count", type=int, required=True)
    parser.add_argument("--ears-count", type=int, required=True)
    parser.add_argument("--expresso-count", type=int, required=True)
    parser.add_argument("--exclude-source-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    ZH_LABELS = args.zh_labels
    ZH_AUDIO = args.zh_audio_root
    PSC = args.paraspeechcaps_parquet
    EARS_AUDIO = args.ears_audio_root
    EXPRESSO_AUDIO = args.expresso_audio_root
    EXCLUDED_IDS, EXCLUDED_PATHS = load_exclusions(args.exclude_source_manifest)
    rows = select_zh(args.seed, args.zh_count) + select_en(
        args.seed, args.ears_count, args.expresso_count
    )
    expected = args.zh_count + args.ears_count + args.expresso_count
    if len({row["audio_path"] for row in rows}) != expected:
        raise RuntimeError("speech audio paths are not unique")
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "rows": len(rows),
        "languages": Counter(row["language"] for row in rows),
        "sources": Counter(row["source_dataset"] for row in rows),
        "unique_audio_paths": len({row["audio_path"] for row in rows}),
        "unique_transcripts": len({norm_text(row["text"]) for row in rows}),
        "unique_speakers": len({(row["source_dataset"], row["speaker_id"]) for row in rows}),
        "unique_speakers_by_source": {
            source: len({row["speaker_id"] for row in rows if row["source_dataset"] == source})
            for source in sorted({row["source_dataset"] for row in rows})
        },
        "attribute_counts": Counter(len(row["selected_caption_attrs"]) for row in rows),
        "excluded_source_ids": len(EXCLUDED_IDS),
        "excluded_source_paths": len(EXCLUDED_PATHS),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
