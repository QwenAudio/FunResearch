#!/usr/bin/env python3
"""Generate reproducible random Path2 scene graphs without semantic themes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import soundfile as sf


EFFECT_ROOT = Path(os.environ.get("FINEAUDIO_LEGACY_EFFECT_ROOT", "."))
SPEECH_INDEX = Path(os.environ.get("FINEAUDIO_LEGACY_SPEECH_INDEX", "."))
WAVCAPS_ROOT = Path(os.environ.get("FINEAUDIO_LEGACY_WAVCAPS_ROOT", "."))
PROFANITY = re.compile(
    r"\b(fuck(?:ing|ed|s)?|shit(?:ty|s)?|bitch(?:es)?|cunts?|wankers?|assholes?)\b|"
    r"操你|草你|他妈的|妈的|傻逼|煞笔|狗日的",
    re.IGNORECASE,
)

EFFECT_LABELS = {
    "alarm": "警报声", "applause": "掌声", "church_bell": "钟声",
    "coin_dropping": "硬币落下声", "drill": "电钻声", "duck": "鸭叫声",
    "engine": "发动机声", "fireworks": "焰火爆响", "helicopter": "直升机声",
    "jackhammer": "风镐声", "motorcycle": "摩托车声", "pour": "倾倒液体声",
    "printer": "打印机声", "sawing": "锯切声", "siren": "警笛声",
    "splash_splatter": "水花飞溅声", "splinter": "木材劈裂声",
    "train": "列车声", "water_tap_faucet": "水龙头流水声",
    "whoosh_swoosh_swish": "快速掠过声",
}
AMBIENCE_LABELS = {
    "rain_on_surface": "雨水敲击表面", "waves_surf": "连续海浪声",
    "waterfall": "瀑布声", "wind_noise_microphone": "阵风声",
    "traffic_noise_roadway_noise": "道路交通底噪",
}
MUSIC_LABELS = {"choir": "合唱音乐", "synthetic_singing": "合成歌声"}

# Review-oriented pool: prefer recognizable, sustained everyday sounds. Rare or
# very short transition effects stay in the source library, but are excluded
# from this batch sampler.
COMMON_EFFECTS = (
    "alarm", "applause", "church_bell", "engine", "motorcycle", "pour",
    "printer", "siren", "splash_splatter", "train", "water_tap_faucet",
)
COMMON_AMBIENCE = (
    "rain_on_surface", "waves_surf", "traffic_noise_roadway_noise",
)

# Events in one family are often difficult to distinguish after mixing.  A
# scene may use at most one member of each family, including its ambience.
ACOUSTIC_CONFUSION_FAMILIES = {
    "water": {
        "rain_on_surface", "waves_surf", "waterfall", "pour",
        "splash_splatter", "water_tap_faucet",
    },
    "alert": {"alarm", "siren"},
    "motor_transport": {"traffic_noise_roadway_noise", "engine", "motorcycle", "train"},
}

# These clips are only category/onset-offset labeled. One selected interval may
# contain multiple acoustic pulses, so they must never be copied as N counted
# occurrences unless a separate atomic-occurrence audit has passed.
NON_COUNTABLE_SOURCE_DATASETS = {"AudioSet-Strong"}

GENDER_ZH = {"male": "男性", "female": "女性"}
AGE_ZH = {"young": "青年", "middle_aged": "中年", "old": "年长"}
PITCH_ZH = {"low": "低沉", "mid": "中等音高", "high": "较高", "variable": "音调有起伏"}
SPEED_ZH = {"slow": "沉稳缓慢", "normal": "自然", "fast": "较快"}
EMOTION_ZH = {
    "serious": "严肃", "calm": "平静", "sad": "悲伤", "helpless": "无奈",
    "anxious": "焦虑", "angry": "愤怒", "happy": "愉快", "excited": "激动",
    "fear": "恐惧", "frustrated": "沮丧", "confident": "自信",
}


def speech_description(row: dict[str, Any]) -> tuple[str, str]:
    primary = row.get("primary_attrs") or {}
    gender = GENDER_ZH.get(str(primary.get("gender", "")).lower(), "说话者")
    age = AGE_ZH.get(str(primary.get("age", "")).lower(), "")
    identity = f"{age}{gender}" if gender != "说话者" else gender
    pitch = PITCH_ZH.get(str(primary.get("pitch", "")).lower())
    speed = SPEED_ZH.get(str(primary.get("speed", "")).lower())
    emotions = []
    for value in re.split(r"[,;/]", str(primary.get("emotion", ""))):
        mapped = EMOTION_ZH.get(value.strip().lower())
        if mapped and mapped not in emotions:
            emotions.append(mapped)
    # Keep the rendering conversational: choose at most two salient traits
    # instead of mechanically verbalizing every metadata attribute.
    traits: list[str] = []
    if emotions:
        traits.append(emotions[0])
    if speed and speed != "自然":
        traits.append("语速较快" if speed == "较快" else "语速沉稳缓慢")
    if not traits and pitch:
        pitch_trait = {
            "低沉": "嗓音低沉",
            "较高": "嗓音偏高",
            "音调有起伏": "语调起伏明显",
        }.get(pitch)
        if pitch_trait:
            traits.append(pitch_trait)
    if not traits:
        return identity, f"一名{identity}说"
    if len(traits) == 1:
        trait = traits[0]
        if trait.startswith("语速") or trait.startswith("嗓音"):
            return identity, f"一名{identity}{trait}地说"
        return identity, f"一名{identity}{trait}地说"
    return identity, f"一名{identity}{traits[0]}、{traits[1]}地说"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audio_duration(path: Path) -> float:
    return float(sf.info(path).duration)


def choose_asset(root: Path, folder: str, rng: random.Random, min_duration: float = 0.0) -> Path:
    values = [
        path for path in sorted((root / folder).glob("*.wav"))
        if audio_duration(path) >= min_duration
    ]
    if not values:
        raise FileNotFoundError(root / folder)
    return rng.choice(values)


def source_event(
    event_id: str,
    role: str,
    label: str,
    path: Path,
    start: float,
    duration: float,
    gain_db: float,
    *,
    loop: bool = False,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "role": role,
        "label": label,
        "source_dataset": "AudioSet-Strong",
        "source_id": path.stem,
        "source_path": str(path),
        "start": round(start, 2),
        "duration": round(duration, 2),
        "gain_db": gain_db,
        "loop": loop,
        "caption_phrase": f"{label}出现",
    }


def speech_pool(language: str) -> list[dict[str, Any]]:
    result = []
    for row in read_jsonl(SPEECH_INDEX):
        path = Path(row.get("audio_path", ""))
        if (
            row.get("language") != language
            or row.get("exclude_from_positive")
            or row.get("discard_sample")
            or row.get("verification_status") not in {"verified", "pending"}
            or not path.exists()
            or PROFANITY.search(str(row.get("text", "")))
        ):
            continue
        duration = audio_duration(path)
        if 1.2 <= duration <= 15.0 and 2 <= len(str(row["text"])) <= 140:
            result.append({
                **row,
                "duration": duration,
                "speech_quality_tier": (
                    "human_verified"
                    if row.get("verification_status") == "verified"
                    else "pending_auto_gated"
                ),
            })
    return result


def foreground_occurrence_count(events: list[dict[str, Any]]) -> int:
    return sum(
        len(event.get("occurrences", [None]))
        for event in events if event["role"] != "ambience"
    )


def acoustic_family(folder: str) -> str | None:
    return next(
        (name for name, members in ACOUSTIC_CONFUSION_FAMILIES.items()
         if folder in members),
        None,
    )


def choose_distinct_effects(
    count: int, rng: random.Random, blocked_families: set[str]
) -> list[str]:
    candidates = list(COMMON_EFFECTS)
    rng.shuffle(candidates)
    chosen: list[str] = []
    used_families = set(blocked_families)
    for folder in candidates:
        family = acoustic_family(folder)
        if family and family in used_families:
            continue
        chosen.append(folder)
        if family:
            used_families.add(family)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Not enough acoustically distinct effects for count={count}")


def reschedule_foreground(
    events: list[dict[str, Any]], duration: float, rng: random.Random,
    requested_overlap_count: int,
    requested_co_onset_count: int,
    max_gap: float,
) -> None:
    """Place foreground with irregular gaps and zero to two bounded overlaps."""
    items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for event in events:
        if event["role"] == "ambience":
            continue
        placements = event.get("occurrences")
        if placements:
            items.extend((event, placement) for placement in placements)
        else:
            items.append((event, event))
    rng.shuffle(items)

    # Keep two occurrences of the same repeated event apart when possible.
    for index in range(1, len(items)):
        if items[index][0]["event_id"] == items[index - 1][0]["event_id"]:
            swap = next(
                (candidate for candidate in range(index + 1, len(items))
                 if items[candidate][0]["event_id"] != items[index][0]["event_id"]),
                None,
            )
            if swap is not None:
                items[index], items[swap] = items[swap], items[index]

    overlap_candidates = [
        index for index in range(1, len(items))
        if items[index - 1][0]["role"] != "speech"
        and items[index][0]["role"] != "speech"
    ]
    rng.shuffle(overlap_candidates)
    overlap_indices: list[int] = []
    if requested_overlap_count:
        for candidate in overlap_candidates:
            if any(abs(candidate - existing) <= 1 for existing in overlap_indices):
                continue
            overlap_indices.append(candidate)
            if len(overlap_indices) == requested_overlap_count:
                break
    if len(overlap_indices) != requested_overlap_count:
        raise ValueError(
            f"Unable to place {requested_overlap_count} non-adjacent overlaps"
        )
    co_onset_indices = set(rng.sample(
        overlap_indices, k=min(requested_co_onset_count, len(overlap_indices))
    ))
    overlap_by_index = {}
    for index in overlap_indices:
        if index in co_onset_indices:
            items[index][1]["duration"] = round(min(
                float(items[index][1]["duration"]), rng.uniform(4.2, 5.0)
            ), 2)
            overlap_by_index[index] = min(
                float(items[index - 1][1]["duration"]),
                float(items[index][1]["duration"]),
            )
            for event, _ in (items[index - 1], items[index]):
                event["gain_db"] = float(event.get("gain_db", 0.0)) - 2.0
        else:
            overlap_by_index[index] = round(rng.uniform(
                0.55,
                min(2.6, float(items[index - 1][1]["duration"]) * 0.42),
            ), 2)

    durations = [float(placement["duration"]) for _, placement in items]
    gap_slots = len(items) + 1 - len(overlap_indices)
    total_gap = duration - sum(durations) + sum(overlap_by_index.values())
    min_gap = 0.2
    if not gap_slots * min_gap <= total_gap <= gap_slots * max_gap:
        raise ValueError(
            f"Unable to schedule foreground: total_gap={total_gap:.3f}, slots={gap_slots}"
        )

    gaps: list[float] | None = None
    remaining = total_gap - gap_slots * min_gap
    for _ in range(500):
        weights = [rng.gammavariate(1.15, 1.0) for _ in range(gap_slots)]
        candidate = [min_gap + remaining * weight / sum(weights) for weight in weights]
        if max(candidate) <= max_gap and (
            gap_slots < 3 or max(candidate) - min(candidate) >= 0.9
        ):
            gaps = candidate
            break
    if gaps is None:
        raise ValueError("Unable to sample irregular foreground gaps")

    gap_iter = iter(gaps)
    cursor = next(gap_iter)
    previous_end = 0.0
    previous_start = 0.0
    for index, (_, placement) in enumerate(items):
        if index:
            if index in co_onset_indices:
                cursor = previous_start
            elif index in overlap_by_index:
                cursor = previous_end - overlap_by_index[index]
            else:
                cursor = previous_end + next(gap_iter)
        placement["start"] = round(cursor, 2)
        previous_start = cursor
        previous_end = max(previous_end, cursor + float(placement["duration"]))


def generate_one(
    index: int,
    rng: random.Random,
    speeches: dict[str, list[dict[str, Any]]],
    requested_overlap_count: int,
    requested_co_onset_count: int,
    background_mode: str,
) -> dict[str, Any]:
    duration = float(rng.randint(44, 54))
    events: list[dict[str, Any]] = []
    next_id = 1

    # Background is optional.  Foreground density is raised when it is absent
    # so the final signal still cannot contain a large silent block.
    ambience_family = None
    if background_mode == "continuous":
        folder = rng.choice(COMMON_AMBIENCE)
        ambience_family = acoustic_family(folder)
        path = choose_asset(EFFECT_ROOT, folder, rng)
        events.append(source_event(
            f"E{next_id}", "ambience", AMBIENCE_LABELS[folder], path,
            0.0, duration, -16.0, loop=True,
        ))
        events[-1]["acoustic_confusion_family"] = acoustic_family(folder)
        next_id += 1

    # Two of every three samples include speech; alternate zh/en to keep balance.
    include_speech = index % 3 != 0
    if include_speech:
        language = "zh" if index % 2 == 0 else "en"
        speech = rng.choice(speeches[language])
        prompt = speech.get("rp_prompt") or speech.get("dsd_prompt") or ""
        speaker_identity, speaker_description = speech_description(speech)
        events.append({
            "event_id": f"E{next_id}", "role": "speech",
            "label": f"{speaker_identity}说话声", "source_dataset": "InstructTTSEval reviewed",
            "source_id": speech["uid"], "source_path": speech["audio_path"],
            "start": 0.0, "duration": round(speech["duration"], 3), "gain_db": 0.0, "language": language,
            "speaker": speaker_identity, "speaker_description": speaker_description,
            "style_prompt": prompt, "primary_attrs": speech.get("primary_attrs", {}),
            "secondary_attrs": speech.get("secondary_attrs", {}), "text": speech["text"],
            "caption_phrase": speaker_description,
        })
        next_id += 1

    # Optional music is explicit, but does not impose a coherent scene theme.
    if index % 2 == 1:
        folder = rng.choice(sorted(MUSIC_LABELS))
        path = choose_asset(EFFECT_ROOT, folder, rng)
        length = round(rng.uniform(7.5, 12.5), 2)
        events.append(source_event(f"E{next_id}", "music", MUSIC_LABELS[folder], path, 0.0, length, -12.0, loop=True))
        next_id += 1

    blocked_families = {family for family in (ambience_family,) if family}
    current_foreground_count = sum(event["role"] != "ambience" for event in events)
    if background_mode == "none":
        target_options = tuple(range(6, min(8, current_foreground_count + 6) + 1))
        target_weights = tuple(1 for _ in target_options)
    else:
        target_options = (4, 5, 6) if current_foreground_count else (4, 5)
        target_weights = (3, 5, 2) if current_foreground_count else (3, 5)
    target_foreground_count = rng.choices(
        target_options, weights=target_weights, k=1
    )[0]
    effect_count = max(2, target_foreground_count - current_foreground_count)
    folders = choose_distinct_effects(effect_count, rng, blocked_families)
    for position, folder in enumerate(folders):
        group_id = f"E{next_id}"
        path = choose_asset(EFFECT_ROOT, folder, rng, min_duration=4.2)
        base_length = round(min(rng.uniform(4.2, 8.8), audio_duration(path)), 2)
        event = source_event(group_id, "sound_effect", EFFECT_LABELS[folder], path, 0.0, base_length, -4.0)
        event["count_control_allowed"] = False
        event["count_control_reason"] = "category interval may contain multiple acoustic onsets"
        event["acoustic_confusion_family"] = acoustic_family(folder)
        events.append(event)
        next_id += 1

    max_foreground_gap = 2.0 if background_mode == "none" else 6.0
    reschedule_foreground(
        events, duration, rng, requested_overlap_count,
        requested_co_onset_count, max_foreground_gap,
    )

    return {
        "id": f"path2_random_{index+1:04d}",
        "generation_mode": "random",
        "sampling_seed": rng.randrange(2**31),
        "duration": duration,
        "background_mode": background_mode,
        "mix_policy": {
            "preferred_sound_pool": "common_sustained",
            "foreground_occurrence_count_range": "4-6",
            "sound_effect_duration_seconds": "4.2-8.8",
            "max_foreground_layers": 2,
            "foreground_overlap_count_range": "0-2",
            "requested_foreground_overlap_count": requested_overlap_count,
            "requested_co_onset_count": requested_co_onset_count,
            "foreground_overlap_seconds": "0.55-2.6",
            "foreground_gap_seconds": "irregular_0.2-6.0",
            "relative_gain_db": {"ambience": -16, "music": -12, "speech": 0, "sound_effect": -4},
            "max_detected_silence_seconds": 2.5,
            "max_foreground_gap_seconds": max_foreground_gap,
            "count_control": "disabled_until_atomic_occurrence_verified",
            "acoustic_confusion_family_limit": 1,
        },
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "random_scene_graphs.jsonl")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    speeches = {lang: speech_pool(lang) for lang in ("zh", "en")}
    if not all(speeches.values()):
        raise RuntimeError("No eligible reviewed speech-prompt pairs")
    overlap_plan = (
        [0] * round(args.count * 0.3)
        + [2] * round(args.count * 0.2)
    )
    overlap_plan += [1] * (args.count - len(overlap_plan))
    rng.shuffle(overlap_plan)
    background_plan = (
        ["none"] * round(args.count * 0.3)
        + ["continuous"] * (args.count - round(args.count * 0.3))
    )
    rng.shuffle(background_plan)
    co_onset_plan = [0] * args.count
    overlap_samples = [index for index, count in enumerate(overlap_plan) if count]
    if overlap_samples:
        co_onset_plan[rng.choice(overlap_samples)] = 1
    rows = []
    failure_counts: Counter[str] = Counter()
    attempts_for_current = 0
    while len(rows) < args.count:
        attempts_for_current += 1
        try:
            rows.append(generate_one(
                len(rows), rng, speeches, overlap_plan[len(rows)],
                co_onset_plan[len(rows)], background_plan[len(rows)],
            ))
            attempts_for_current = 0
        except ValueError as exc:
            failure_counts[str(exc)] += 1
            if attempts_for_current >= 200:
                raise RuntimeError(
                    f"Unable to generate sample {len(rows)} after 200 attempts; "
                    f"failures={dict(failure_counts.most_common(5))}"
                ) from exc
            continue
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")
    print(json.dumps({
        "samples": len(rows), "zh_pool": len(speeches["zh"]),
        "en_pool": len(speeches["en"]), "overlap_plan": overlap_plan,
        "co_onset_plan": co_onset_plan,
        "background_plan": background_plan,
        "rejected_attempts": sum(failure_counts.values()),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
