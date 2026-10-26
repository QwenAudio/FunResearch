#!/usr/bin/env python3
"""Generate a diverse 1k-scale Path2 batch with globally low source reuse."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

from scene_graph_utils import (
    acoustic_family as legacy_acoustic_family,
    read_jsonl,
    reschedule_foreground,
    speech_description,
    speech_pool,
)


CONFUSION_FAMILIES = {
    "water": {"rain", "waves", "waterfall", "pour", "splash", "water_tap"},
    "siren_alarm": {"alarm", "siren"},
    "bells": {"bell", "church_bell"},
    "road_motor": {"traffic", "engine", "medium_engine", "motorcycle"},
    "applause": {"applause", "clapping", "cheering"},
    "office_machine": {"printer", "computer_keyboard"},
    "steady_appliance": {"mechanical_fan", "vacuum_cleaner"},
}

EXPANDED_FAMILY_KEYWORDS = {
    "water": (
        "water", "rain", "wave", "ocean", "stream", "waterfall", "pour",
        "splash", "trickle", "gush", "gurgling", "drip", "sink", "bathtub",
        "toilet", "liquid", "spray",
    ),
    "siren_alarm": ("siren", "alarm", "smoke_detector"),
    "bells": ("bell", "chime", "ding"),
    "road_motor": (
        "engine", "car", "truck", "bus", "motorcycle", "race_car", "idling",
        "accelerating", "vehicle_horn", "air_horn", "reversing_beeps",
    ),
    "rail": ("train", "railroad", "subway", "clickety_clack"),
    "aircraft": ("aircraft", "airplane", "jet_engine", "helicopter", "propeller"),
    "watercraft": ("motorboat", "boat", "ship", "sailboat", "rowboat", "foghorn"),
    "fire_explosion": ("fire", "explosion", "fireworks", "firecracker", "eruption"),
    "powered_tools": (
        "chainsaw", "power_tool", "lawn_mower", "drill", "jackhammer", "sawing",
        "sanding", "filing_rasp",
    ),
    "bird": ("bird", "chirp", "chicken", "rooster", "crow", "duck", "quack", "pigeon", "dove", "coo", "caw"),
    "dog": ("dog", "bark", "whimper"),
    "cat": ("cat", "purr", "caterwaul"),
    "insect": ("insect", "cricket", "bee", "wasp", "mosquito", "housefly"),
    "applause": ("applause", "clapping", "cheering"),
    "clock": ("clock", "tick_tock", "tick"),
    "phone": ("ringtone", "telephone", "busy_signal", "dial_tone"),
}

# Specific transport families must win over broad tokens such as ``engine``.
# The ordering also keeps ``train`` from being confused with the substring
# ``rain``: matching below is token/phrase based rather than arbitrary
# substring based.
EXPANDED_FAMILY_PRIORITY = (
    "aircraft", "rail", "watercraft", "road_motor", "water",
    "siren_alarm", "bells", "powered_tools",
)


def _keyword_matches_slug(keyword: str, slug: str) -> bool:
    if "_" in keyword:
        return keyword in slug
    return keyword in slug.split("_")


def family(slug: str) -> str | None:
    for name, members in CONFUSION_FAMILIES.items():
        if slug in members:
            return name
    ordered_names = list(EXPANDED_FAMILY_PRIORITY) + [
        name for name in EXPANDED_FAMILY_KEYWORDS
        if name not in EXPANDED_FAMILY_PRIORITY
    ]
    for name in ordered_names:
        if any(
            _keyword_matches_slug(keyword, slug)
            for keyword in EXPANDED_FAMILY_KEYWORDS[name]
        ):
            return name
    return legacy_acoustic_family(slug)


class UniqueAssetPool:
    def __init__(self, rows: list[dict[str, Any]], rng: random.Random):
        self.rng = rng
        self.by_role_slug: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            self.by_role_slug[row["role"]][row["slug"]].append(row)
        for by_slug in self.by_role_slug.values():
            for values in by_slug.values():
                rng.shuffle(values)
        self.all_by_role_slug = {
            role: {slug: list(values) for slug, values in by_slug.items()}
            for role, by_slug in self.by_role_slug.items()
        }
        self.used = Counter()
        self.slug_usage = Counter()
        self.used_source_ids: set[str] = set()
        self.transaction: list[tuple[dict[str, Any], bool]] | None = None

    def reserve_paths(self, paths: set[str]) -> None:
        """Remove recordings already used by a previous batch."""
        for role, by_slug in self.by_role_slug.items():
            for slug, values in by_slug.items():
                kept = []
                for row in values:
                    if row["source_path"] in paths:
                        self.used[row["source_path"]] += 1
                        self.slug_usage[(role, slug)] += 1
                        self.used_source_ids.add(str(row["video_id"]))
                    else:
                        kept.append(row)
                by_slug[slug] = kept

    def begin(self) -> None:
        if self.transaction is not None:
            raise RuntimeError("asset transaction already active")
        self.transaction = []

    def commit(self) -> None:
        self.transaction = None

    def rollback(self) -> None:
        if self.transaction is None:
            return
        for chosen, was_unique in reversed(self.transaction):
            self.used[chosen["source_path"]] -= 1
            self.slug_usage[(chosen["role"], chosen["slug"])] -= 1
            if was_unique:
                self.used_source_ids.remove(str(chosen["video_id"]))
                self.by_role_slug[chosen["role"]][chosen["slug"]].append(chosen)
        self.transaction = None

    def take(
        self, role: str, *, blocked_slugs: set[str] | None = None,
        blocked_families: set[str] | None = None,
    ) -> dict[str, Any]:
        blocked_slugs = blocked_slugs or set()
        blocked_families = blocked_families or set()
        by_slug = self.by_role_slug[role]
        eligible_slugs = [
            slug for slug in by_slug
            if slug not in blocked_slugs and family(slug) not in blocked_families
        ]
        self.rng.shuffle(eligible_slugs)
        eligible_slugs.sort(key=lambda slug: self.slug_usage[(role, slug)])
        for slug in eligible_slugs:
            candidates = by_slug[slug]
            while candidates:
                chosen = candidates.pop()
                source_id = str(chosen["video_id"])
                if source_id in self.used_source_ids:
                    continue
                self.used_source_ids.add(source_id)
                self.used[chosen["source_path"]] += 1
                self.slug_usage[(role, slug)] += 1
                if self.transaction is not None:
                    self.transaction.append((chosen, True))
                return chosen

        # Only after every suitable recording has been used once, reuse the
        # least-used recording.  This keeps exact reuse close to the theoretical
        # minimum when a batch requests more events than the unique pool holds.
        fallback = [
            row for slug, values in self.all_by_role_slug[role].items()
            if slug not in blocked_slugs and family(slug) not in blocked_families
            for row in values
        ]
        if not fallback:
            raise RuntimeError(
                f"No asset role={role}, blocked_slugs={sorted(blocked_slugs)}, "
                f"blocked_families={sorted(blocked_families)}"
            )
        minimum = min(self.used[row["source_path"]] for row in fallback)
        least_used = [row for row in fallback if self.used[row["source_path"]] == minimum]
        chosen = self.rng.choice(least_used)
        self.used[chosen["source_path"]] += 1
        self.slug_usage[(role, chosen["slug"])] += 1
        if self.transaction is not None:
            self.transaction.append((chosen, False))
        return chosen


class SpeechCycle:
    def __init__(self, rng: random.Random, speech_manifest: Path | None = None):
        self.rng = rng
        if speech_manifest is None:
            self.values = {lang: speech_pool(lang) for lang in ("zh", "en")}
        else:
            rows = read_jsonl(speech_manifest)
            self.values = {
                lang: [row for row in rows if row.get("language") == lang]
                for lang in ("zh", "en")
            }
            if any(not values for values in self.values.values()):
                raise ValueError(
                    f"official speech manifest lacks a language: {speech_manifest}"
                )
        for values in self.values.values():
            rng.shuffle(values)
        self.position = Counter()
        self.usage = Counter()
        self.snapshot: tuple[Counter, Counter] | None = None

    def begin(self) -> None:
        self.snapshot = (self.position.copy(), self.usage.copy())

    def commit(self) -> None:
        self.snapshot = None

    def rollback(self) -> None:
        if self.snapshot is not None:
            self.position, self.usage = self.snapshot
        self.snapshot = None

    def take(self, language: str) -> dict[str, Any]:
        values = self.values[language]
        position = self.position[language]
        if position and position % len(values) == 0:
            self.rng.shuffle(values)
        row = values[position % len(values)]
        self.position[language] += 1
        self.usage[row["audio_path"]] += 1
        return row


def asset_event(
    event_id: str, asset: dict[str, Any], start: float, duration: float,
    gain_db: float, *, loop: bool,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "role": asset["role"],
        "label": asset["label"],
        "source_dataset": asset["source_dataset"],
        "source_id": asset["video_id"],
        "source_path": asset["source_path"],
        "source_event_name": asset["event_name"],
        "source_event_start": asset["event_start"],
        "source_event_end": asset["event_end"],
        "start": round(start, 2),
        "duration": round(duration, 2),
        "gain_db": gain_db,
        "loop": loop,
        "asset_slug": asset["slug"],
        "acoustic_confusion_family": family(asset["slug"]),
        "atomic_count_verified": False,
        "count_control_allowed": False,
        "count_control_reason": "strong interval may contain multiple acoustic onsets",
        "caption_phrase": f"{asset['label']}出现",
    }


def make_scene(
    index: int, id_offset: int, rng: random.Random, assets: UniqueAssetPool,
    speeches: SpeechCycle, overlap_count: int, co_onset_count: int,
    background_mode: str, speech_language: str | None, include_music: bool,
) -> dict[str, Any]:
    duration = float(rng.randint(42, 58))
    events: list[dict[str, Any]] = []
    next_id = 1
    blocked_slugs: set[str] = set()
    blocked_families: set[str] = set()

    if background_mode == "continuous":
        asset = assets.take("ambience")
        events.append(asset_event(
            f"E{next_id}", asset, 0.0, duration,
            rng.uniform(-13.0, -9.5), loop=True,
        ))
        blocked_slugs.add(asset["slug"])
        if family(asset["slug"]):
            blocked_families.add(family(asset["slug"]))
        next_id += 1

    if speech_language:
        speech = speeches.take(speech_language)
        identity, description = speech_description(speech)
        events.append({
            "event_id": f"E{next_id}", "role": "speech",
            "label": f"{identity}说话声",
            "source_dataset": (
                "InstructTTSEval official"
                if speech.get("verification_status") == "official_unreviewed"
                else "InstructTTSEval reviewed subset (legacy fallback)"
            ),
            "source_id": speech["uid"], "source_path": speech["audio_path"],
            "start": 0.0, "duration": round(speech["duration"], 3),
            "gain_db": rng.uniform(-1.0, 1.0), "language": speech_language,
            "speaker": identity, "speaker_description": description,
            "style_prompt": speech.get("rp_prompt") or speech.get("dsd_prompt") or "",
            "primary_attrs": speech.get("primary_attrs", {}),
            "secondary_attrs": speech.get("secondary_attrs", {}),
            "speech_quality_tier": speech.get("speech_quality_tier"),
            "source_verification_status": speech.get("verification_status"),
            "text": speech["text"], "caption_phrase": description,
        })
        next_id += 1

    if include_music:
        asset = assets.take("music")
        length = min(float(asset["duration"]), rng.uniform(7.5, 12.5))
        events.append(asset_event(
            f"E{next_id}", asset, 0.0, length,
            rng.uniform(-11.5, -8.0), loop=False,
        ))
        next_id += 1

    current = sum(event["role"] != "ambience" for event in events)
    if background_mode == "none":
        target = rng.choices((6, 7), weights=(3, 2), k=1)[0]
        max_gap = 2.0
    else:
        target = rng.choices((4, 5), weights=(3, 2), k=1)[0]
        max_gap = 5.0
    effect_count = max(2, target - current)
    for _ in range(effect_count):
        asset = assets.take(
            "sound_effect", blocked_slugs=blocked_slugs,
            blocked_families=blocked_families,
        )
        event_duration = min(float(asset["duration"]), rng.uniform(4.2, 8.8))
        events.append(asset_event(
            f"E{next_id}", asset, 0.0, event_duration,
            rng.uniform(-6.0, -3.0), loop=False,
        ))
        blocked_slugs.add(asset["slug"])
        if family(asset["slug"]):
            blocked_families.add(family(asset["slug"]))
        next_id += 1

    reschedule_foreground(
        events, duration, rng, overlap_count, co_onset_count, max_gap,
    )
    return {
        "id": f"path2_natural_{id_offset + index + 1:06d}",
        "generation_mode": "large_diverse_random",
        "sampling_seed": rng.randrange(2**31),
        "duration": duration,
        "background_mode": background_mode,
        "mix_policy": {
            "source_selection": "globally_unique_audioset_source_when_available",
            "semantic_theme_required": False,
            "max_foreground_layers": 2,
            "requested_foreground_overlap_count": overlap_count,
            "requested_co_onset_count": co_onset_count,
            "max_foreground_gap_seconds": max_gap,
            "max_detected_silence_seconds": 2.5,
            "count_control": "disabled_until_atomic_occurrence_verified",
            "acoustic_confusion_family_limit": 1,
        },
        "events": events,
    }


def exact_plan(count: int, distribution: list[tuple[Any, float]], rng: random.Random) -> list[Any]:
    values: list[Any] = []
    for value, fraction in distribution[:-1]:
        values.extend([value] * round(count * fraction))
    values.extend([distribution[-1][0]] * (count - len(values)))
    rng.shuffle(values)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usage-report", type=Path, required=True)
    parser.add_argument("--id-offset", type=int, default=0)
    parser.add_argument("--exclude-scene-graphs", type=Path)
    parser.add_argument("--speech-manifest", type=Path)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    assets = UniqueAssetPool(read_jsonl(args.asset_manifest), rng)
    if args.exclude_scene_graphs:
        previous = read_jsonl(args.exclude_scene_graphs)
        assets.reserve_paths({
            event["source_path"] for row in previous for event in row["events"]
            if event["role"] != "speech"
        })
    speeches = SpeechCycle(rng, args.speech_manifest)

    overlaps = exact_plan(args.count, [(0, 0.30), (1, 0.50), (2, 0.20)], rng)
    backgrounds = exact_plan(
        args.count, [("none", 0.50), ("continuous", 0.50)], rng
    )
    music = exact_plan(args.count, [(False, 0.55), (True, 0.45)], rng)
    speech_plan = exact_plan(
        args.count, [(None, 0.50), ("zh", 0.25), ("en", 0.25)], rng
    )
    co_onsets = [0] * args.count
    eligible = [i for i, value in enumerate(overlaps) if value > 0]
    for index in rng.sample(eligible, k=round(args.count * 0.10)):
        co_onsets[index] = 1

    rows: list[dict[str, Any]] = []
    rejected = Counter()
    while len(rows) < args.count:
        index = len(rows)
        assets.begin()
        speeches.begin()
        try:
            rows.append(make_scene(
                index, args.id_offset, rng, assets, speeches,
                overlaps[index], co_onsets[index],
                backgrounds[index], speech_plan[index], music[index],
            ))
            assets.commit()
            speeches.commit()
        except ValueError as exc:
            assets.rollback()
            speeches.rollback()
            rejected[str(exc)] += 1
            if sum(rejected.values()) > args.count * 10:
                raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    all_source_paths = [
        event["source_path"] for row in rows for event in row["events"]
    ]
    usage = Counter(all_source_paths)
    report = {
        "samples": len(rows),
        "background_distribution": Counter(backgrounds),
        "overlap_distribution": Counter(overlaps),
        "co_onset_samples": sum(co_onsets),
        "music_samples": sum(music),
        "speech_distribution": Counter(speech_plan),
        "total_source_uses": len(all_source_paths),
        "unique_source_paths": len(usage),
        "max_source_reuse": max(usage.values()),
        "reused_source_count": sum(value > 1 for value in usage.values()),
        "speech_source_max_reuse": max(speeches.usage.values(), default=0),
        "speech_candidate_pool": {
            lang: len(values) for lang, values in speeches.values.items()
        },
        "speech_manifest": str(args.speech_manifest) if args.speech_manifest else None,
        "rejected_attempts": rejected,
        "scene_graphs": str(args.output),
    }
    args.usage_report.parent.mkdir(parents=True, exist_ok=True)
    args.usage_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
