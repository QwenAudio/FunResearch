#!/usr/bin/env python3
"""Apply Route 2 hard gates and freeze the accepted caption pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from asset_pool import family


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value):
    return "".join(re.findall(r"[\w\u4e00-\u9fff]+", str(value).casefold(), re.UNICODE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-zh-count", type=int, required=True)
    parser.add_argument("--expected-en-count", type=int, required=True)
    parser.add_argument("--expected-zh-speech-count", type=int, required=True)
    parser.add_argument("--expected-en-speech-count", type=int, required=True)
    parser.add_argument("--expected-foreground-min", type=int, default=5)
    parser.add_argument("--expected-foreground-max", type=int, default=8)
    parser.add_argument("--max-effect-slug-reuse", type=int)
    parser.add_argument("--max-effect-family-reuse", type=int)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    errors = []
    if len(rows) != args.expected_count:
        errors.append(f"sample_count:{len(rows)}")
    paths, final, effect_slugs = [], [], []
    speech_counts, language_counts, source_counts = Counter(), Counter(), Counter()
    no_speech_languages = Counter()
    for row in rows:
        rid = row["id"]
        language = row["target_caption_language"]
        language_counts[language] += 1
        if row["caption_languages"] != [language]:
            errors.append(f"{rid}:caption_language")
        audit = row.get("caption_rewrite_audit_by_language", {}).get(language, {})
        if audit.get("validation_errors"):
            errors.append(f"{rid}:v2_validation")
        caption = row.get("caption_natural_by_language", {}).get(language, "")
        if not caption:
            errors.append(f"{rid}:empty_caption")
        foreground_groups = {event["event_group_id"] for event in row["events"] if event["role"] != "ambience"}
        if not args.expected_foreground_min <= len(foreground_groups) <= args.expected_foreground_max:
            errors.append(f"{rid}:foreground_groups:{len(foreground_groups)}")
        speech_events = [event for event in row["events"] if event["role"] == "speech"]
        if speech_events:
            event = speech_events[0]
            speech_counts[event["language"]] += 1
            source_counts[event["source_dataset"]] += 1
            if len(event.get("selected_caption_attrs") or []) not in {2, 3}:
                errors.append(f"{rid}:speech_attr_count")
            if norm(event["text"]) not in norm(caption):
                errors.append(f"{rid}:speech_text")
            if norm(event["speaker_description"]) not in norm(caption):
                errors.append(f"{rid}:speaker_description")
        else:
            no_speech_languages[language] += 1
        if row["background_mode"] == "continuous":
            ambience = [event for event in row["events"] if event["role"] == "ambience"]
            if len(ambience) != 1 or not -13.0 <= float(ambience[0]["gain_db"]) <= -9.5:
                errors.append(f"{rid}:ambience_gain")
        for event in row["events"]:
            paths.append(event["source_path"])
            if event["role"] == "sound_effect":
                effect_slugs.append(event.get("asset_slug"))
            if "instructttseval" in (event.get("source_dataset", "") + event["source_path"]).casefold():
                errors.append(f"{rid}:eval_leakage")
        serialized = json.dumps(row, ensure_ascii=False).casefold()
        if '"omni' in serialized:
            errors.append(f"{rid}:removed_stage_field")
        value = dict(row)
        value["caption_language"] = language
        value["caption"] = caption
        value["final_caption_method"] = "natural_text_rewrite"
        final.append(value)
    if len(paths) != len(set(paths)):
        errors.append(f"source_reuse:{len(paths) - len(set(paths))}")
    effect_slug_counts = Counter(effect_slugs)
    effect_family_counts = Counter(family(slug) or slug for slug in effect_slugs)
    max_effect_slug_reuse = max(effect_slug_counts.values(), default=0)
    max_effect_family_reuse = max(effect_family_counts.values(), default=0)
    slug_target_pass = (
        args.max_effect_slug_reuse is None
        or max_effect_slug_reuse <= args.max_effect_slug_reuse
    )
    family_target_pass = (
        args.max_effect_family_reuse is None
        or max_effect_family_reuse <= args.max_effect_family_reuse
    )
    expected = {
        "language_counts": {"zh": args.expected_zh_count, "en": args.expected_en_count},
        "speech_counts": {
            "zh": args.expected_zh_speech_count,
            "en": args.expected_en_speech_count,
        },
        "no_speech_languages": {
            "zh": args.expected_zh_count - args.expected_zh_speech_count,
            "en": args.expected_en_count - args.expected_en_speech_count,
        },
    }
    actual = {
        "language_counts": dict(language_counts), "speech_counts": dict(speech_counts),
        "no_speech_languages": dict(no_speech_languages), "source_counts": dict(source_counts),
    }
    for key, wanted in expected.items():
        if actual[key] != wanted:
            errors.append(f"distribution:{key}:{actual[key]}")
    report = {
        "valid": not errors, "samples": len(rows), "errors": errors[:500], "error_count": len(errors),
        **actual, "unique_source_paths": len(set(paths)), "total_source_uses": len(paths),
        "sound_effect_uses": len(effect_slugs),
        "unique_sound_effect_slugs": len(effect_slug_counts),
        "max_sound_effect_slug_reuse": max_effect_slug_reuse,
        "reused_sound_effect_slugs": sum(value > 1 for value in effect_slug_counts.values()),
        "unique_sound_effect_families": len(effect_family_counts),
        "max_sound_effect_family_reuse": max_effect_family_reuse,
        "reused_sound_effect_families": sum(value > 1 for value in effect_family_counts.values()),
        "showcase_diversity_targets": {
            "max_effect_slug_reuse": args.max_effect_slug_reuse,
            "max_effect_family_reuse": args.max_effect_family_reuse,
            "slug_target_pass": slug_target_pass,
            "family_target_pass": family_target_pass,
            "all_targets_pass": slug_target_pass and family_target_pass,
        },
        "expected_samples": args.expected_count,
        "expected_distributions": expected,
        "excluded_stages": ["audio_model_verification", "alternate_caption_generation"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        print(json.dumps(report, ensure_ascii=False)); raise SystemExit(1)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
