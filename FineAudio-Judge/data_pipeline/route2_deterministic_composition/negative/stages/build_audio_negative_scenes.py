#!/usr/bin/env python3
"""Materialize Route 2 audio-side recipes as deterministic scene graphs."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


POSITIVE_STAGES = Path(__file__).resolve().parents[2] / "positive" / "stages"
sys.path.insert(0, str(POSITIVE_STAGES))
from generate_scene_graphs import family  # noqa: E402
MAX_SPEECH_SHORTFALL_SECONDS = 1.25


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def safe_id(negative_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", negative_id).strip("_")


def truth(subtype: str, left: tuple[float, float], right: tuple[float, float]) -> bool:
    if subtype == "overlap":
        return min(left[1], right[1]) - max(left[0], right[0]) > 0.08
    if subtype == "before":
        return left[1] <= right[0] + 0.08
    raise ValueError(subtype)


def choose_relation_layout(recipe: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    events = recipe["parent_scene"]["events"]
    original = {event["event_id"]: (float(event["start"]), float(event["end"])) for event in events}
    foreground = sorted(
        [event for event in events if event["role"] != "ambience"],
        key=lambda event: (event["start"], event["end"], event["event_id"]),
    )
    ids = [event["event_id"] for event in foreground]
    slots = [(float(event["start"]), float(event["end"])) for event in foreground]
    frozen = recipe.get("audio_relation_materialization")
    if frozen:
        slot_assignment = frozen["slot_assignment"]
        if set(slot_assignment) != set(ids) or sorted(slot_assignment.values()) != list(range(len(ids))):
            raise ValueError(f"invalid frozen slot assignment: {recipe['negative_id']}")
        intervals = dict(original)
        moved = []
        for index, event_id in enumerate(ids):
            slot_index = int(slot_assignment[event_id])
            intervals[event_id] = slots[slot_index]
            if slot_index != index:
                moved.append(event_id)
        relation_questions = [
            question for question in recipe["original_questions"]
            if question["subtype"] in {"before", "overlap"}
        ]
        changed = []
        for question in relation_questions:
            evidence = question["construction_evidence"]
            if not truth(
                question["subtype"],
                intervals[evidence["source_event_id"]],
                intervals[evidence["target_event_id"]],
            ):
                changed.append(question["rubric_id"])
        if set(changed) != set(frozen["changed_relation_rubric_ids"]):
            raise ValueError(f"frozen relation realization mismatch: {recipe['negative_id']}")
        return intervals, {
            "status": "frozen_exact",
            "method": frozen["method"],
            "moved_event_ids": moved,
            "changed_relation_rubric_ids": sorted(changed),
            "slot_assignment": slot_assignment,
        }
    target_ids = {
        primitive["rubric_id"] for primitive in recipe["primitives"]
        if primitive["type"] == "relation"
    }
    if not target_ids:
        return original, {"status": "identity", "moved_event_ids": [], "changed_relation_rubric_ids": []}
    retained = set(recipe["retained_rubric_ids"])
    relation_questions = [
        question for question in recipe["original_questions"]
        if question["subtype"] in {"before", "overlap"}
    ]
    by_event = {event["event_id"]: event for event in events}
    speech_replaced = {
        primitive.get("event_id") for primitive in recipe["primitives"]
        if primitive["type"] in {"event", "attribute"}
    }

    best = None
    for permutation in permutations(range(len(slots))):
        if all(index == slot for index, slot in enumerate(permutation)):
            continue
        intervals = dict(original)
        feasible = True
        for event_id, slot_index in zip(ids, permutation):
            start, end = slots[slot_index]
            duration = end - start
            event = by_event[event_id]
            if event["role"] == "speech" and event_id not in speech_replaced:
                if duration > float(event.get("source_duration", event["duration"])) + 1e-4:
                    feasible = False
                    break
            intervals[event_id] = (start, end)
        if not feasible:
            continue
        violations = []
        changed = []
        for question in relation_questions:
            evidence = question["construction_evidence"]
            rid = question["rubric_id"]
            observed = truth(
                question["subtype"],
                intervals[evidence["source_event_id"]],
                intervals[evidence["target_event_id"]],
            )
            if not observed:
                changed.append(rid)
            if rid in target_ids and observed:
                violations.append("target_still_true:" + rid)
            elif rid in retained and not observed:
                violations.append("retained_became_false:" + rid)
        moved = [event_id for index, event_id in enumerate(ids) if permutation[index] != index]
        displacement = sum(abs(slots[permutation[index]][0] - slots[index][0]) for index in range(len(ids)))
        score = (len(violations), len(set(changed) - target_ids), len(moved), displacement)
        if best is None or score < best[0]:
            best = (score, intervals, moved, sorted(changed), violations)
            if score[:2] == (0, 0):
                break
    if best is None or best[0][0]:
        detail = None if best is None else {"score": best[0], "violations": best[4][:12]}
        raise ValueError(f"no exact relation permutation for {recipe['negative_id']}: {detail}")
    return best[1], {
        "status": "exact",
        "score": best[0],
        "moved_event_ids": best[2],
        "changed_relation_rubric_ids": best[3],
    }


class UniquePool:
    def __init__(self, rows: list[dict[str, Any]], seed: int, used_positive_paths: set[str]):
        self.rows = [row for row in rows if self.path(row) not in used_positive_paths]
        self.rows.sort(key=lambda row: stable_rank(seed, self.path(row)))
        self.used = set()

    @staticmethod
    def path(row: dict[str, Any]) -> str:
        value = row.get("source_path") or row.get("audio_path")
        if not value:
            raise KeyError("pool row has neither source_path nor audio_path")
        return str(value)

    def take(self, predicate, description: str) -> dict[str, Any]:
        for row in self.rows:
            path = self.path(row)
            if path in self.used or not predicate(row):
                continue
            self.used.add(path)
            return row
        raise RuntimeError(f"unique replacement pool exhausted: {description}")


def speech_attrs(row: dict[str, Any]) -> dict[str, str]:
    return {str(item["name"]): str(item["value"]) for item in row.get("selected_caption_attrs") or []}


def speech_candidate_is_suitable(
    row: dict[str, Any],
    event: dict[str, Any],
    required_duration: float,
    targets: list[dict[str, Any]],
) -> bool:
    language = event["language"]
    old_attrs = {str(item["name"]): str(item["value"]) for item in event.get("selected_caption_attrs") or []}
    targeted_attrs = {
        primitive["attribute"]["name"]: str(primitive["attribute"]["value"])
        for primitive in targets if primitive.get("subtype") == "speech_attribute"
    }
    content_targeted = any(primitive.get("subtype") == "speech_content" for primitive in targets)
    if (
        row["language"] != language
        or float(row["duration"]) + MAX_SPEECH_SHORTFALL_SECONDS < required_duration
    ):
        return False
    if row["text"] == event.get("text"):
        return False
    attrs = speech_attrs(row)
    if targeted_attrs and any(attrs.get(name) == value for name, value in targeted_attrs.items()):
        return False
    if not targeted_attrs and not content_targeted and attrs == old_attrs:
        return False
    return True


def assign_speech_replacements(
    recipes: list[dict[str, Any]], pool: UniquePool
) -> dict[tuple[str, str], dict[str, Any]]:
    """Find a deterministic one-to-one assignment instead of using a greedy pool.

    Long speech slots and attribute constraints make greedy selection unsafe: an
    early easy item can consume the only valid source for a later constrained
    item.  A bipartite augmenting-path match proves that every requested speech
    replacement has a unique, compatible source before any scene is written.
    """
    requirements: dict[tuple[str, str], tuple[dict[str, Any], float, list[dict[str, Any]]]] = {}
    for recipe in recipes:
        intervals, _ = choose_relation_layout(recipe)
        by_event = {event["event_id"]: event for event in recipe["parent_scene"]["events"]}
        targets_by_event: dict[str, list[dict[str, Any]]] = {}
        for primitive in recipe["primitives"]:
            if primitive["type"] == "attribute":
                targets_by_event.setdefault(primitive["event_id"], []).append(primitive)
        for event_id, targets in targets_by_event.items():
            event = by_event[event_id]
            if event["role"] != "speech":
                raise ValueError(f"non-speech attribute primitive: {recipe['negative_id']}:{event_id}")
            start, end = intervals[event_id]
            requirements[(recipe["negative_id"], event_id)] = (event, end - start, targets)

    available = [row for row in pool.rows if pool.path(row) not in pool.used]
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(requirements) * 2 + 100))
    adjacency: dict[tuple[str, str], list[int]] = {}
    for key, (event, required_duration, targets) in requirements.items():
        adjacency[key] = [
            index for index, row in enumerate(available)
            if speech_candidate_is_suitable(row, event, required_duration, targets)
        ]
        if not adjacency[key]:
            raise RuntimeError(
                f"no speech replacement candidate for {key[0]}:{key[1]} "
                f"language={event['language']} duration>={required_duration:.3f}"
            )

    matched_row: dict[int, tuple[str, str]] = {}
    matched_requirement: dict[tuple[str, str], int] = {}

    def augment(key: tuple[str, str], seen: set[int]) -> bool:
        for index in adjacency[key]:
            if index in seen:
                continue
            seen.add(index)
            incumbent = matched_row.get(index)
            if incumbent is None or augment(incumbent, seen):
                matched_row[index] = key
                matched_requirement[key] = index
                return True
        return False

    order = sorted(
        requirements,
        key=lambda key: (
            len(adjacency[key]),
            -requirements[key][1],
            key,
        ),
    )
    for key in order:
        if not augment(key, set()):
            event, required_duration, _ = requirements[key]
            raise RuntimeError(
                f"unique speech assignment impossible for {key[0]}:{key[1]} "
                f"language={event['language']} duration>={required_duration:.3f}; "
                f"candidate_count={len(adjacency[key])}"
            )

    result = {key: available[index] for key, index in matched_requirement.items()}
    for row in result.values():
        pool.used.add(pool.path(row))
    return result


def replace_speech(
    event: dict[str, Any], required_duration: float, targets: list[dict[str, Any]], chosen: dict[str, Any]
) -> dict[str, Any]:
    if not speech_candidate_is_suitable(chosen, event, required_duration, targets):
        raise ValueError(f"preassigned incompatible speech source for {event['event_id']}")
    before = {key: event.get(key) for key in ("source_dataset", "source_id", "source_path", "text", "speaker_description", "selected_caption_attrs")}
    event.update({
        "source_dataset": chosen["source_dataset"],
        "source_id": chosen["uid"],
        "source_path": chosen["audio_path"],
        "language": chosen["language"],
        "speaker": chosen["speaker"],
        "speaker_description": chosen["speaker_description"],
        "style_prompt": chosen["style_prompt"],
        "text": chosen["text"],
        "selected_caption_attrs": chosen["selected_caption_attrs"],
        "primary_attrs": chosen["primary_attrs"],
        "all_source_attrs": chosen["all_source_attrs"],
        "annotation_format": chosen["annotation_format"],
        "caption_phrase": chosen["speech_caption_phrase"],
        "caption_label_zh": chosen["speaker_description"],
        "caption_label_en": chosen["speaker_description"],
        "source_offset": 0.0,
        "source_duration": chosen["duration"],
    })
    if float(chosen["duration"]) < required_duration:
        event["duration"] = round(float(chosen["duration"]), 4)
        event["end"] = round(float(event["start"]) + float(event["duration"]), 4)
    after = {key: event.get(key) for key in before}
    return {"event_id": event["event_id"], "role": "speech", "before": before, "after": after}


def nonspeech_candidate_is_suitable(
    row: dict[str, Any], event: dict[str, Any], scene_events: list[dict[str, Any]]
) -> bool:
    blocked_slugs = {
        other.get("asset_slug") for other in scene_events
        if other is not event and other.get("asset_slug")
    }
    blocked_families = {
        value for other in scene_events if other is not event
        for value in [other.get("acoustic_confusion_family") or (
            family(other.get("asset_slug")) if other.get("asset_slug") else None
        )]
        if value is not None
    }
    target_slug = event.get("asset_slug")
    target_family = event.get("acoustic_confusion_family") or (family(target_slug) if target_slug else None)
    if target_slug:
        blocked_slugs.add(target_slug)
    if target_family is not None:
        blocked_families.add(target_family)
    # The scene role is a mixing/layout role and is preserved below. The source
    # acoustic class may differ: music/ambience targets must use a non-music or
    # non-ambience source respectively, while a named sound effect may use any
    # unrelated non-speech source.
    allowed_roles = (
        {"sound_effect", "ambience"} if event["role"] == "music"
        else ({"sound_effect", "music"} if event["role"] == "ambience"
              else {"sound_effect", "ambience", "music"})
    )
    return (
        row["role"] in allowed_roles
        and row["slug"] not in blocked_slugs
        and (family(row["slug"]) is None or family(row["slug"]) not in blocked_families)
    )


def assign_nonspeech_replacements(
    recipes: list[dict[str, Any]], pool: UniquePool
) -> dict[tuple[str, str], dict[str, Any]]:
    requirements: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for recipe in recipes:
        by_event = {event["event_id"]: event for event in recipe["parent_scene"]["events"]}
        attribute_ids = {
            primitive["event_id"] for primitive in recipe["primitives"]
            if primitive["type"] == "attribute"
        }
        event_ids = {
            primitive["event_id"] for primitive in recipe["primitives"]
            if primitive["type"] == "event"
        }
        for event_id in sorted(event_ids - attribute_ids):
            requirements[(recipe["negative_id"], event_id)] = (
                by_event[event_id], recipe["parent_scene"]["events"]
            )

    available = [row for row in pool.rows if pool.path(row) not in pool.used]
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(requirements) * 2 + 100))
    adjacency: dict[tuple[str, str], list[int]] = {}
    for key, (event, scene_events) in requirements.items():
        adjacency[key] = [
            index for index, row in enumerate(available)
            if nonspeech_candidate_is_suitable(row, event, scene_events)
        ]
        if not adjacency[key]:
            raise RuntimeError(f"no nonspeech replacement candidate for {key[0]}:{key[1]}")

    matched_row: dict[int, tuple[str, str]] = {}
    matched_requirement: dict[tuple[str, str], int] = {}

    def augment(key: tuple[str, str], seen: set[int]) -> bool:
        for index in adjacency[key]:
            if index in seen:
                continue
            seen.add(index)
            incumbent = matched_row.get(index)
            if incumbent is None or augment(incumbent, seen):
                matched_row[index] = key
                matched_requirement[key] = index
                return True
        return False

    for key in sorted(requirements, key=lambda item: (len(adjacency[item]), item)):
        if not augment(key, set()):
            raise RuntimeError(
                f"unique nonspeech assignment impossible for {key[0]}:{key[1]}; "
                f"candidate_count={len(adjacency[key])}"
            )
    result = {key: available[index] for key, index in matched_requirement.items()}
    for row in result.values():
        pool.used.add(pool.path(row))
    return result


def replace_nonspeech(
    event: dict[str, Any], required_duration: float, scene_events: list[dict[str, Any]], pool: UniquePool
) -> dict[str, Any]:
    old = {key: event.get(key) for key in (
        "source_dataset", "source_id", "source_path", "asset_slug", "label", "role",
        "text", "speaker_description", "selected_caption_attrs",
    )}
    mixing_role = event["role"]
    chosen = pool.take(
        lambda row: nonspeech_candidate_is_suitable(row, event, scene_events),
        f"event={event['event_id']} role={event['role']} duration={required_duration:.3f}",
    )
    event.update({
        "source_dataset": chosen.get("source_dataset", "AudioSet-Strong"),
        "source_id": chosen["video_id"],
        "source_path": chosen["source_path"],
        "source_event_name": chosen["event_name"],
        "source_event_start": chosen["event_start"],
        "source_event_end": chosen["event_end"],
        "role": mixing_role,
        "replacement_source_role": chosen["role"],
        "label": chosen["label"],
        "asset_slug": chosen["slug"],
        "acoustic_confusion_family": family(chosen["slug"]),
        "caption_phrase": f"{chosen['label']}出现",
        "caption_label_zh": chosen["label"],
        "caption_label_en": chosen["event_name"],
        "loop": bool(event.get("loop") or float(chosen["duration"]) + 1e-4 < required_duration),
        "source_offset": 0.0,
        "source_duration": chosen["duration"],
    })
    # A speech event can be an event-level negative target and therefore be
    # replaced by a non-speech source.  In that case the event's semantic role
    # must follow the replacement source; otherwise compose_scene_graphs would
    # describe the sound as speech and re-emit the stale transcript.  Keep the
    # former role only as explicit provenance and remove speech-only metadata.
    if mixing_role == "speech":
        event["mixing_role"] = mixing_role
        event["role"] = chosen["role"]
        for key in (
            "language", "speaker", "speaker_description", "style_prompt", "text",
            "selected_caption_attrs", "primary_attrs", "all_source_attrs",
            "annotation_format",
        ):
            event.pop(key, None)
    after = {key: event.get(key) for key in old}
    return {
        "event_id": event["event_id"],
        "mixing_role": mixing_role,
        "replacement_source_role": chosen["role"],
        "before": old,
        "after": after,
    }


def build_scene(
    recipe: dict[str, Any],
    nonspeech: UniquePool,
    speech_assignments: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    parent = recipe["parent_scene"]
    scene = deepcopy(parent)
    scene["id"] = safe_id(recipe["negative_id"])
    scene["title"] = f"Path2 audio negative for {recipe['parent_positive_id']}"
    scene["generation_mode"] = "route2_audio_negative_controlled_edit"
    scene["sampling_seed"] = recipe["sampling_seed"]
    scene["negative_id"] = recipe["negative_id"]
    scene["parent_positive_id"] = recipe["parent_positive_id"]
    scene["original_caption"] = recipe["original_caption"]

    intervals, relation_audit = choose_relation_layout(recipe)
    by_event = {event["event_id"]: event for event in scene["events"]}
    for event_id, (start, end) in intervals.items():
        event = by_event[event_id]
        event["start"] = round(start, 4)
        event["end"] = round(end, 4)
        event["duration"] = round(end - start, 4)

    targets_by_event: dict[str, list[dict[str, Any]]] = {}
    event_replacements = set()
    for primitive in recipe["primitives"]:
        if primitive["type"] == "event":
            event_replacements.add(primitive["event_id"])
        elif primitive["type"] == "attribute":
            targets_by_event.setdefault(primitive["event_id"], []).append(primitive)

    replace_ids = sorted(event_replacements | set(targets_by_event))
    replacements = []
    for event_id in replace_ids:
        event = by_event[event_id]
        required_duration = float(event["duration"])
        if event["role"] == "speech" and targets_by_event.get(event_id):
            replacements.append(replace_speech(
                event,
                required_duration,
                targets_by_event.get(event_id, []),
                speech_assignments[(recipe["negative_id"], event_id)],
            ))
        elif event["role"] == "speech" and event_id in event_replacements:
            replacements.append(replace_nonspeech(
                event,
                required_duration,
                scene["events"],
                nonspeech,
            ))
        elif event_id in event_replacements:
            replacements.append(replace_nonspeech(
                event,
                required_duration,
                scene["events"],
                nonspeech,
            ))
        else:
            raise ValueError(f"non-speech attribute primitive: {recipe['negative_id']}:{event_id}")

    # A relation permutation can place an unchanged non-speech source into a
    # longer time slot. Looping is safe for these environmental/effect sources
    # and preserves the frozen interval instead of letting compose() truncate
    # it and introduce an undeclared gap or silence.
    for event in scene["events"]:
        if (
            event["role"] != "speech"
            and float(event.get("source_duration", event["duration"])) + 1e-4 < float(event["duration"])
        ):
            event["loop"] = True

    if any("instructttseval" in (str(event.get("source_dataset")) + str(event.get("source_path"))).casefold()
           for event in scene["events"]):
        raise ValueError(f"InstructTTSEval leakage: {recipe['negative_id']}")

    # Speech sources may be slightly shorter than a frozen slot.  Prove that
    # this shortening did not silently change any relation beyond the relation
    # rubrics explicitly declared by the deterministic slot permutation.
    realized_intervals = {
        event["event_id"]: (float(event["start"]), float(event["end"]))
        for event in scene["events"]
    }
    realized_false_relations = sorted(
        question["rubric_id"]
        for question in recipe["original_questions"]
        if question["subtype"] in {"before", "overlap"}
        and not truth(
            question["subtype"],
            realized_intervals[question["construction_evidence"]["source_event_id"]],
            realized_intervals[question["construction_evidence"]["target_event_id"]],
        )
    )
    expected_false_relations = sorted(relation_audit["changed_relation_rubric_ids"])
    if realized_false_relations != expected_false_relations:
        raise ValueError(
            f"replacement changed undeclared relations: {recipe['negative_id']}; "
            f"expected={expected_false_relations}; realized={realized_false_relations}"
        )
    relation_audit["post_replacement_changed_relation_rubric_ids"] = realized_false_relations
    scene["negative_construction"] = {
        "schema": "route2_audio_negative_construction",
        "negative_id": recipe["negative_id"],
        "parent_positive_id": recipe["parent_positive_id"],
        "perturbation_category": recipe["perturbation_category"],
        "intended_consistency_level": recipe["intended_consistency_level"],
        "direct_target_rubric_ids": recipe["direct_target_rubric_ids"],
        "hierarchy_affected_rubric_ids": recipe["hierarchy_affected_rubric_ids"],
        "relation_audit": relation_audit,
        "source_replacements": replacements,
        "original_audio_path": recipe["original_audio_path"],
        "original_audio_sha256": recipe["original_audio_sha256"],
        "caption_frozen_byte_identical": True,
    }
    # These are regenerated by compose(); keeping stale positive derived fields
    # would make the pre-render graph ambiguous.
    for key in (
        "audio_path", "audio_caption", "audio_caption_timestamped", "audio_caption_natural",
        "caption", "caption_natural_by_language", "caption_timestamped_by_language",
        "construction_controls", "overlap_qc", "signal_qc",
    ):
        scene.pop(key, None)
    return scene


def speech_replacement_duration(recipe: dict[str, Any]) -> float:
    """Sort constrained speech assignments longest-first to avoid greedy starvation."""
    attribute_ids = {
        primitive["event_id"] for primitive in recipe["primitives"]
        if primitive["type"] == "attribute"
    }
    if not attribute_ids:
        return 0.0
    intervals, _ = choose_relation_layout(recipe)
    by_event = {event["event_id"]: event for event in recipe["parent_scene"]["events"]}
    return max(
        intervals[event_id][1] - intervals[event_id][0]
        for event_id in attribute_ids if by_event[event_id]["role"] == "speech"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--nonspeech-pool", type=Path, required=True)
    parser.add_argument("--speech-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--failed-validation", type=Path)
    parser.add_argument(
        "--semantic-failures", type=Path,
        help="JSONL emitted by materialize_audio_negatives.py; rebuild only affected samples",
    )
    parser.add_argument("--exclude-valid-mix", type=Path)
    parser.add_argument("--audit-relations-only", action="store_true")
    args = parser.parse_args()

    recipes = read_jsonl(args.recipes)
    excluded_paths: set[str] = set()
    if args.failed_validation:
        failed_ids = {
            row["id"] for row in read_jsonl(args.failed_validation) if not row.get("valid")
        }
        recipes = [recipe for recipe in recipes if safe_id(recipe["negative_id"]) in failed_ids]
    if args.semantic_failures:
        failed_ids = {
            row["negative_id"] for row in read_jsonl(args.semantic_failures)
        }
        recipes = [recipe for recipe in recipes if recipe["negative_id"] in failed_ids]
    if args.exclude_valid_mix:
        validation_path = args.exclude_valid_mix / "validation_report.jsonl"
        valid_ids = {
            row["id"] for row in read_jsonl(validation_path) if row.get("valid")
        }
        for row in read_jsonl(args.exclude_valid_mix / "pairs.jsonl"):
            if row["id"] in valid_ids:
                excluded_paths.update(str(event["source_path"]) for event in row["events"])
    if args.limit:
        recipes = recipes[:args.limit]
    failures = []
    if args.audit_relations_only:
        exact = 0
        for recipe in recipes:
            try:
                choose_relation_layout(recipe)
                exact += 1
            except Exception as exc:
                failures.append({"negative_id": recipe["negative_id"], "error": repr(exc)})
        result = {"recipes": len(recipes), "exact": exact, "failed": len(failures), "examples": failures[:20]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if failures:
            raise SystemExit(2)
        return

    positive_paths = {
        event["source_path"] for recipe in recipes for event in recipe["parent_scene"]["events"]
    } | excluded_paths
    nonspeech = UniquePool(read_jsonl(args.nonspeech_pool), args.seed, positive_paths)
    speech = UniquePool(read_jsonl(args.speech_pool), args.seed, positive_paths)
    speech_assignments = assign_speech_replacements(recipes, speech)
    ordered_recipes = sorted(
        recipes, key=lambda recipe: (-speech_replacement_duration(recipe), recipe["negative_id"])
    )
    scenes = []
    for recipe in ordered_recipes:
        scenes.append(build_scene(recipe, nonspeech, speech_assignments))
    scenes.sort(key=lambda scene: scene["negative_id"])
    write_jsonl(args.output, scenes)
    summary = {
        "schema": "route2_audio_negative_scene_summary",
        "recipes": len(recipes),
        "scenes": len(scenes),
        "categories": dict(Counter(scene["negative_construction"]["perturbation_category"] for scene in scenes)),
        "nonspeech_replacements": len(nonspeech.used),
        "speech_replacements": len(speech.used),
        "unique_replacement_paths": len(nonspeech.used | speech.used),
        "positive_replacement_path_overlap": len((nonspeech.used | speech.used) & positive_paths),
        "instructttseval_rows": 0,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
