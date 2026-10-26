#!/usr/bin/env python3
"""Create one deterministic timestamped caption draft per composed pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scene-graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    target_languages = {
        row["id"]: row["target_caption_language"] for row in read_jsonl(args.scene_graphs)
    }
    output = []
    for source in rows:
        row = dict(source)
        language = target_languages[row["id"]]
        row["target_caption_language"] = language
        events, sentences = [], []
        for raw in sorted(row["events"], key=lambda x: (x["start"], x["end"], x["event_id"])):
            event = dict(raw)
            event["caption_label_zh"] = event["speaker_description"] if event["role"] == "speech" else event["label"]
            event["caption_label_en"] = event["speaker_description"] if event["role"] == "speech" else (event.get("source_event_name") or event["label"])
            label = event["caption_label_zh"] if language == "zh" else event["caption_label_en"]
            if event["role"] == "speech":
                if language == "zh":
                    sentences.append(f'{event["start"]:.2f}至{event["end"]:.2f}秒，{label}：“{event["text"]}”。')
                else:
                    sentences.append(f'From {event["start"]:.2f} to {event["end"]:.2f} seconds, {label}, saying, "{event["text"]}".')
            elif language == "zh":
                sentences.append(f'{event["start"]:.2f}至{event["end"]:.2f}秒，{label}出现。')
            else:
                sentences.append(f'From {event["start"]:.2f} to {event["end"]:.2f} seconds, {label} is heard.')
            events.append(event)
        row["events"] = events
        row["caption_languages"] = [language]
        row["caption_language_reason"] = "speech_language" if row["speech_present"] else "balanced_no_speech_assignment"
        row["caption_timestamped_by_language"] = {language: "".join(sentences) if language == "zh" else " ".join(sentences)}
        row["caption_timeline_policy"] = "deterministic_timeline_selected_speech_attributes_only"
        output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({
        "samples": len(output), "caption_jobs": sum(len(row["caption_languages"]) for row in output),
        "zh": sum(row["target_caption_language"] == "zh" for row in output),
        "en": sum(row["target_caption_language"] == "en" for row in output),
        "omni_jobs": 0, "v4_jobs": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
