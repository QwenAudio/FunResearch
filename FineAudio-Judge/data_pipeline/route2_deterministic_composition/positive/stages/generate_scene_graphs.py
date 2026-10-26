#!/usr/bin/env python3
"""Generate deterministic Route 2 scene graphs from explicit user quotas."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

from asset_pool import (
    UniqueAssetPool, asset_event, family,
)
from scene_graph_utils import read_jsonl, reschedule_foreground


class StrictUniqueAssetPool(UniqueAssetPool):
    """Never use the legacy pool's reuse fallback."""

    def __init__(self, rows: list[dict[str, Any]], rng: random.Random):
        super().__init__(rows, rng)
        self.diversity_usage = Counter()
        self.diversity_relaxations = Counter()
        self.diversity_relaxations_snapshot = None

    def begin(self):
        super().begin()
        self.diversity_relaxations_snapshot = self.diversity_relaxations.copy()

    def commit(self):
        super().commit()
        self.diversity_relaxations_snapshot = None

    @staticmethod
    def diversity_group(role: str, slug: str) -> tuple[str, str]:
        # Related engine/aircraft/bell classes should be spread across demos,
        # while unrelated slugs without a known confusion family remain their
        # own diversity groups.
        return role, family(slug) or slug

    def rollback(self):
        if self.transaction is not None:
            for chosen, _ in self.transaction:
                group = self.diversity_group(chosen["role"], chosen["slug"])
                self.diversity_usage[group] -= 1
        super().rollback()
        if self.diversity_relaxations_snapshot is not None:
            self.diversity_relaxations = self.diversity_relaxations_snapshot
        self.diversity_relaxations_snapshot = None

    def take(
        self, role: str, *, blocked_slugs=None, blocked_families=None,
        max_slug_reuse: int | None = None,
        max_family_reuse: int | None = None,
    ):
        blocked_slugs = blocked_slugs or set()
        blocked_families = blocked_families or set()
        by_slug = self.by_role_slug[role]
        eligible = [
            slug for slug in by_slug
            if slug not in blocked_slugs and family(slug) not in blocked_families
            and (max_slug_reuse is None or self.slug_usage[(role, slug)] < max_slug_reuse)
            and (
                max_family_reuse is None
                or self.diversity_usage[self.diversity_group(role, slug)] < max_family_reuse
            )
        ]
        relaxed = False
        if not eligible and (max_slug_reuse is not None or max_family_reuse is not None):
            # Cross-demo diversity is a showcase preference. Keep within-scene
            # confusion blocks and global source uniqueness, but do not make a
            # smaller user-owned pool unusable solely because this preference
            # cannot be met.
            eligible = [
                slug for slug in by_slug
                if slug not in blocked_slugs and family(slug) not in blocked_families
            ]
            relaxed = True
        self.rng.shuffle(eligible)
        eligible.sort(key=lambda slug: (
            self.diversity_usage[self.diversity_group(role, slug)],
            self.slug_usage[(role, slug)],
        ))
        for slug in eligible:
            while by_slug[slug]:
                chosen = by_slug[slug].pop()
                source_id = str(chosen["video_id"])
                if source_id in self.used_source_ids:
                    continue
                self.used_source_ids.add(source_id)
                self.used[chosen["source_path"]] += 1
                self.slug_usage[(role, slug)] += 1
                self.diversity_usage[self.diversity_group(role, slug)] += 1
                if self.transaction is not None:
                    self.transaction.append((chosen, True))
                if relaxed:
                    self.diversity_relaxations[role] += 1
                return chosen
        raise RuntimeError(f"unique asset pool exhausted for role={role}")


class StrictSpeechPool:
    def __init__(self, rows: list[dict[str, Any]], rng: random.Random):
        self.values = {lang: [row for row in rows if row["language"] == lang] for lang in ("zh", "en")}
        for values in self.values.values():
            rng.shuffle(values)
        self.position = Counter()
        self.usage = Counter()
        self.snapshot = None

    def begin(self):
        self.snapshot = (self.position.copy(), self.usage.copy())

    def commit(self):
        self.snapshot = None

    def rollback(self):
        if self.snapshot:
            self.position, self.usage = self.snapshot
        self.snapshot = None

    def take(self, language: str) -> dict[str, Any]:
        position = self.position[language]
        if position >= len(self.values[language]):
            raise RuntimeError(f"speech pool exhausted for language={language}")
        row = self.values[language][position]
        self.position[language] += 1
        self.usage[row["audio_path"]] += 1
        return row


def make_scene(index, rng, assets, speeches, spec, overlap_count, co_onset_count,
               background_mode, include_music, id_prefix, max_effect_slug_reuse,
               max_effect_family_reuse,
               foreground_no_background_range, foreground_with_background_range,
               duration_range):
    duration = float(rng.randint(*duration_range))
    events = []
    blocked_slugs, blocked_families = set(), set()
    next_id = 1
    if background_mode == "continuous":
        asset = assets.take("ambience")
        events.append(asset_event(f"E{next_id}", asset, 0.0, duration, rng.uniform(-13.0, -9.5), loop=True))
        blocked_slugs.add(asset["slug"])
        if family(asset["slug"]):
            blocked_families.add(family(asset["slug"]))
        next_id += 1

    speech_language = spec["speech_language"]
    if speech_language:
        speech = speeches.take(speech_language)
        events.append({
            "event_id": f"E{next_id}", "role": "speech",
            "label": "说话声" if speech_language == "zh" else "speech",
            "source_dataset": speech["source_dataset"], "source_id": speech["uid"],
            "source_path": speech["audio_path"], "start": 0.0,
            "duration": round(float(speech["duration"]), 3),
            "gain_db": rng.uniform(-1.0, 1.0), "language": speech_language,
            "speaker": speech["speaker"], "speaker_description": speech["speaker_description"],
            "style_prompt": speech["style_prompt"], "text": speech["text"],
            "selected_caption_attrs": speech["selected_caption_attrs"],
            "primary_attrs": speech["primary_attrs"],
            "all_source_attrs": speech["all_source_attrs"],
            "annotation_format": speech["annotation_format"],
            "caption_phrase": speech["speech_caption_phrase"],
        })
        next_id += 1

    if include_music:
        asset = assets.take("music")
        events.append(asset_event(f"E{next_id}", asset, 0.0, min(float(asset["duration"]), rng.uniform(7.5, 12.5)), rng.uniform(-11.5, -8.0), loop=False))
        next_id += 1

    current = sum(event["role"] != "ambience" for event in events)
    target_range = (
        foreground_no_background_range
        if background_mode == "none"
        else foreground_with_background_range
    )
    target = rng.randint(*target_range)
    # Two disjoint overlap spans require enough foreground groups to keep the
    # overlap pairs non-adjacent under the public timeline policy.
    if overlap_count >= 2:
        target = max(target, 5)
    max_gap = 2.5 if background_mode == "none" else 5.0
    for _ in range(max(2, target - current)):
        asset = assets.take(
            "sound_effect", blocked_slugs=blocked_slugs,
            blocked_families=blocked_families,
            max_slug_reuse=max_effect_slug_reuse,
            max_family_reuse=max_effect_family_reuse,
        )
        events.append(asset_event(f"E{next_id}", asset, 0.0, min(float(asset["duration"]), rng.uniform(4.2, 8.8)), rng.uniform(-6.0, -3.0), loop=False))
        blocked_slugs.add(asset["slug"])
        if family(asset["slug"]):
            blocked_families.add(family(asset["slug"]))
        next_id += 1
    reschedule_foreground(events, duration, rng, overlap_count, co_onset_count, max_gap)
    return {
        "id": f"{id_prefix}_{index + 1:04d}",
        "generation_mode": "route2_deterministic_composition",
        "sampling_seed": rng.randrange(2**31), "duration": duration,
        "background_mode": background_mode,
        "target_caption_language": spec["caption_language"],
        "mix_policy": {
            "source_selection": "globally_unique_no_fallback_reuse",
            "foreground_group_target": target,
            "max_foreground_layers": 2,
            "requested_foreground_overlap_count": overlap_count,
            "requested_co_onset_count": co_onset_count,
            "max_foreground_gap_seconds": max_gap,
            "max_detected_silence_seconds": 2.5,
            "continuous_background_gain_db_range": [-13.0, -9.5],
            "acoustic_confusion_family_limit": 1,
        },
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--speech-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usage-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of scenes to generate.",
    )
    parser.add_argument("--zh-speech-count", type=int, required=True)
    parser.add_argument("--en-speech-count", type=int, required=True)
    parser.add_argument("--zh-no-speech-count", type=int, required=True)
    parser.add_argument("--en-no-speech-count", type=int, required=True)
    parser.add_argument("--overlap-zero-count", type=int, required=True)
    parser.add_argument("--overlap-one-count", type=int, required=True)
    parser.add_argument("--overlap-two-count", type=int, required=True)
    parser.add_argument("--continuous-background-count", type=int, required=True)
    parser.add_argument("--music-count", type=int, required=True)
    parser.add_argument("--co-onset-count", type=int, required=True)
    parser.add_argument("--max-effect-slug-reuse", type=int)
    parser.add_argument("--max-effect-family-reuse", type=int)
    parser.add_argument("--foreground-no-background-min", type=int, default=6)
    parser.add_argument("--foreground-no-background-max", type=int, default=8)
    parser.add_argument("--foreground-with-background-min", type=int, default=5)
    parser.add_argument("--foreground-with-background-max", type=int, default=7)
    parser.add_argument("--duration-min", type=int, default=42)
    parser.add_argument("--duration-max", type=int, default=58)
    parser.add_argument("--id-prefix", required=True)
    args = parser.parse_args()
    for minimum, maximum, name in (
        (args.foreground_no_background_min, args.foreground_no_background_max, "no-background"),
        (args.foreground_with_background_min, args.foreground_with_background_max, "with-background"),
    ):
        if minimum < 2 or maximum < minimum:
            raise ValueError(f"invalid {name} foreground target range: {minimum}..{maximum}")
    if args.max_effect_slug_reuse is not None and args.max_effect_slug_reuse < 1:
        raise ValueError("--max-effect-slug-reuse must be at least 1")
    if args.max_effect_family_reuse is not None and args.max_effect_family_reuse < 1:
        raise ValueError("--max-effect-family-reuse must be at least 1")
    if args.duration_min < 30 or args.duration_max > 60 or args.duration_max < args.duration_min:
        raise ValueError("duration range must satisfy 30 <= min <= max <= 60")
    rng = random.Random(args.seed)
    asset_rows = read_jsonl(args.asset_manifest)
    speech_rows = read_jsonl(args.speech_manifest)
    specs = (
        [{"speech_language": "zh", "caption_language": "zh"}] * args.zh_speech_count
        + [{"speech_language": "en", "caption_language": "en"}] * args.en_speech_count
        + [{"speech_language": None, "caption_language": "zh"}] * args.zh_no_speech_count
        + [{"speech_language": None, "caption_language": "en"}] * args.en_no_speech_count
    )
    if len(specs) != args.count:
        raise ValueError("language/speech quotas must sum to --count")
    rng.shuffle(specs)
    overlaps = (
        [0] * args.overlap_zero_count
        + [1] * args.overlap_one_count
        + [2] * args.overlap_two_count
    )
    if len(overlaps) != args.count:
        raise ValueError("overlap quotas must sum to --count")
    rng.shuffle(overlaps)
    backgrounds = ["continuous"] * args.continuous_background_count + ["none"] * (
        args.count - args.continuous_background_count
    )
    music = [True] * args.music_count + [False] * (args.count - args.music_count)
    rng.shuffle(backgrounds)
    rng.shuffle(music)
    co_onsets = [0] * args.count
    eligible = [i for i, value in enumerate(overlaps) if value]
    if args.co_onset_count > len(eligible):
        raise ValueError("co-onset quota exceeds overlap-eligible scenes")
    for i in rng.sample(eligible, args.co_onset_count):
        co_onsets[i] = 1
    rejected = Counter()
    rows = []
    completed_restart = None
    for restart in range(64):
        generation_rng = random.Random(args.seed + restart * 104729)
        assets = StrictUniqueAssetPool(asset_rows, generation_rng)
        speeches = StrictSpeechPool(speech_rows, generation_rng)
        candidate_rows = []
        attempts_for_scene = 0
        restart_required = False
        while len(candidate_rows) < args.count:
            i = len(candidate_rows)
            assets.begin(); speeches.begin()
            try:
                row = make_scene(
                    i, generation_rng, assets, speeches, specs[i], overlaps[i], co_onsets[i],
                    backgrounds[i], music[i], args.id_prefix,
                    args.max_effect_slug_reuse,
                    args.max_effect_family_reuse,
                    (args.foreground_no_background_min, args.foreground_no_background_max),
                    (args.foreground_with_background_min, args.foreground_with_background_max),
                    (args.duration_min, args.duration_max),
                )
                candidate_rows.append(row); assets.commit(); speeches.commit()
                attempts_for_scene = 0
            except ValueError as exc:
                assets.rollback(); speeches.rollback()
                message = str(exc)
                rejected[message] += 1
                attempts_for_scene += 1
                if attempts_for_scene >= 200:
                    restart_required = True
                    break
        if not restart_required and len(candidate_rows) == args.count:
            rows = candidate_rows
            completed_restart = restart
            break
    if completed_restart is None:
        raise RuntimeError(
            "unable to satisfy scene scheduling constraints after 64 deterministic "
            f"restarts; top_rejections={rejected.most_common(8)}"
        )
    all_paths = [event["source_path"] for row in rows for event in row["events"]]
    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError("source path reuse detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    effect_slugs = [
        event["asset_slug"] for row in rows for event in row["events"]
        if event["role"] == "sound_effect"
    ]
    effect_slug_usage = Counter(effect_slugs)
    effect_family_usage = Counter(family(slug) or slug for slug in effect_slugs)
    max_slug_reuse = max(effect_slug_usage.values(), default=0)
    max_family_reuse = max(effect_family_usage.values(), default=0)
    slug_target_pass = (
        args.max_effect_slug_reuse is None
        or max_slug_reuse <= args.max_effect_slug_reuse
    )
    family_target_pass = (
        args.max_effect_family_reuse is None
        or max_family_reuse <= args.max_effect_family_reuse
    )
    report = {
        "samples": len(rows),
        "configured_plan_size": args.count,
        "speech_samples": sum(spec["speech_language"] is not None for spec in specs),
        "speech_distribution": Counter(spec["speech_language"] for spec in specs),
        "caption_language_distribution": Counter(spec["caption_language"] for spec in specs),
        "background_distribution": Counter(backgrounds),
        "music_samples": sum(music),
        "overlap_distribution": Counter(overlaps),
        "co_onset_samples": sum(co_onsets),
        "total_source_uses": len(all_paths), "unique_source_paths": len(set(all_paths)),
        "max_source_reuse": 1,
        "foreground_group_range": [
            min(args.foreground_no_background_min, args.foreground_with_background_min),
            max(args.foreground_no_background_max, args.foreground_with_background_max),
        ],
        "duration_range": [args.duration_min, args.duration_max],
        "sound_effect_uses": len(effect_slugs),
        "unique_sound_effect_slugs": len(effect_slug_usage),
        "max_sound_effect_slug_reuse": max_slug_reuse,
        "reused_sound_effect_slugs": sum(value > 1 for value in effect_slug_usage.values()),
        "unique_sound_effect_families": len(effect_family_usage),
        "max_sound_effect_family_reuse": max_family_reuse,
        "reused_sound_effect_families": sum(value > 1 for value in effect_family_usage.values()),
        "showcase_diversity_targets": {
            "max_effect_slug_reuse": args.max_effect_slug_reuse,
            "max_effect_family_reuse": args.max_effect_family_reuse,
            "slug_target_pass": slug_target_pass,
            "family_target_pass": family_target_pass,
            "all_targets_pass": slug_target_pass and family_target_pass,
        },
        "showcase_diversity_relaxations": dict(assets.diversity_relaxations),
        "deterministic_restarts": completed_restart,
        "rejected_attempts": rejected,
    }
    args.usage_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
