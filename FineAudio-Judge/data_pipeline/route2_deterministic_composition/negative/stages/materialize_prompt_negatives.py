#!/usr/bin/env python3
"""Freeze valid Route 2 P' patches while proving that A is byte-identical."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


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


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_all_casefold(text: str, old: str, new: str) -> tuple[str, int]:
    if not old:
        return text, 0
    return re.subn(re.escape(old), lambda _: new, text, flags=re.IGNORECASE)


def replace_flexible_whitespace(text: str, old: str, new: str) -> tuple[str, int]:
    """Case-insensitive replacement tolerant of caption whitespace normalization."""
    pattern = r"\s+".join(re.escape(part) for part in old.split())
    if not pattern:
        return text, 0
    return re.subn(pattern, lambda _: new, text, flags=re.IGNORECASE)


def replace_normalized_transcript(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace transcript despite renderer-only spacing/final-punctuation edits."""
    updated, count = replace_flexible_whitespace(text, old, new)
    if count:
        return updated, count
    stripped = old.rstrip().rstrip(".。!?！？,，;；")
    updated, count = replace_flexible_whitespace(text, stripped, new)
    if count:
        return updated, count
    # Natural caption rendering may normalize curly apostrophes and omit final
    # punctuation inside quotes.  Select only a strongly matching quoted span.
    candidates = []
    normalized_old = re.sub(r"[^\w]+", "", old, flags=re.UNICODE).casefold()
    for match in re.finditer(r'["“]([^"”]+)["”]', text, flags=re.DOTALL):
        normalized_candidate = re.sub(r"[^\w]+", "", match.group(1), flags=re.UNICODE).casefold()
        score = SequenceMatcher(None, normalized_old, normalized_candidate).ratio()
        candidates.append((score, match.start(1), match.end(1)))
    if candidates:
        score, start, end = max(candidates)
        if score >= 0.8:
            return text[:start] + new + text[end:], 1
    return text, 0


def replace_event_label(
    text: str, old: str, new: str, protected_longer_labels: Iterable[str]
) -> tuple[str, int]:
    """Replace an event label without modifying a longer retained label.

    AudioSet labels can be prefixes of each other (for example ``Aircraft`` and
    ``Aircraft engine``), so a plain global substitution is not safe.
    """
    protected_spans: list[tuple[int, int]] = []
    for label in protected_longer_labels:
        if label.casefold() == old.casefold() or len(label) <= len(old):
            continue
        protected_spans.extend(
            (match.start(), match.end())
            for match in re.finditer(re.escape(label), text, re.I)
        )
    matches = list(re.finditer(re.escape(old), text, re.I))
    replace_spans = [
        (match.start(), match.end()) for match in matches
        if not any(start <= match.start() and match.end() <= end for start, end in protected_spans)
    ]
    if not replace_spans:
        return text, 0
    pieces, cursor = [], 0
    for start, end in replace_spans:
        pieces.extend((text[cursor:start], new))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), len(replace_spans)


def event_label_present(text: str, label: str, all_labels: Iterable[str]) -> bool:
    """Return whether a standalone label occurrence remains in ``text``."""
    protected_spans: list[tuple[int, int]] = []
    for longer in all_labels:
        if longer.casefold() == label.casefold() or len(longer) <= len(label):
            continue
        protected_spans.extend(
            (match.start(), match.end()) for match in re.finditer(re.escape(longer), text, re.I)
        )
    return any(
        not any(start <= match.start() and match.end() <= end for start, end in protected_spans)
        for match in re.finditer(re.escape(label), text, re.I)
    )


def event_label(event: dict[str, Any], language: str) -> str:
    return str(event["caption_label_zh"] if language == "zh" else event["caption_label_en"])


def attribute_phrase(attribute: dict[str, Any], language: str) -> str:
    if attribute.get("caption_text"):
        return str(attribute["caption_text"])
    if language == "zh":
        mappings = {
            "speed": {"慢": "语速缓慢", "中": "语速适中", "快": "语速较快"},
            "volume": {"低": "音量较低", "中": "音量适中", "高": "音量较高"},
            "pitch": {"低": "音调较低", "中": "音调适中", "高": "音调较高"},
        }
        return mappings.get(str(attribute.get("name")), {}).get(
            str(attribute.get("value")), str(attribute.get("value") or "")
        )
    return str(attribute.get("value") or "")


def stable_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (key + ":" + str(row.get("uid") or row.get("audio_path"))).encode("utf-8")
        ).hexdigest(),
    )


def deterministic_missing_job_seed(job: dict[str, Any]) -> dict[str, Any]:
    """Seed a missing non-relation generation for deterministic materialization.

    Event, attribute, and all-three rows are rebuilt from the original positive
    caption by :func:`repair_direct_targets` (and, for all-three rows,
    :func:`flip_selected_relation`).  A model-produced rewrite is therefore not
    a correctness dependency.  Relation-only rows are intentionally excluded:
    ``build_relation_prompt_negatives.py`` supplies their audited deterministic
    reversal before this stage runs.
    """
    category = job["path2_perturbation_category"]
    if category == "relation":
        raise ValueError(f"relation job requires deterministic reversal: {job['sample_id']}")
    return {
        "schema": "route2_prompt_negative_generation",
        "sample_id": job["sample_id"],
        "parent_positive_id": job["parent_positive_id"],
        "status": "ok",
        "prompt": job["prompt"],
        "prompt_prime": job["prompt"],
        "patches": [],
        "attempts": 0,
        "endpoint": None,
        "model": "deterministic_missing_job_seed",
        "disturbance_category": job["disturbance_category"],
        "selected_rubric_ids": job["selected_rubric_ids"],
        "outside_patch_equality": True,
        "note": (
            "Model generation was missing or invalid; the fail-closed deterministic "
            "materializer will rebuild only the selected atomic facts."
        ),
    }


def repair_direct_targets(
    job: dict[str, Any], generated: dict[str, Any], pair: dict[str, Any],
    speech_pool: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministically enforce every non-relation direct target in P'.

    The original positive caption supplies the natural sentence skeleton.  This
    gate changes only selected atomic facts, preventing collateral edits to
    retained events.  Relation-only rows are supplied separately by
    deterministic_relation_reversal.
    """
    row = dict(generated)
    prompt_prime = str(row["prompt_prime"])
    events = {event["event_id"]: event for event in pair["events"]}
    selected_exists = {
        question["construction_evidence"]["event_id"]
        for question in job["selected_rubrics"] if question["subtype"] == "exists"
    }
    candidates = [
        item for item in job.get("replacement_candidates") or []
        if item.get("candidate_type") != "speech_content" and str(item.get("label") or "")
    ]
    speech_text_candidates = [
        item for item in job.get("replacement_candidates") or []
        if item.get("candidate_type") == "speech_content" and str(item.get("text") or "")
    ]
    used_candidate_labels: set[str] = set()
    selected_event_labels = {
        event_label(events[question["construction_evidence"]["event_id"]], job["language"]).casefold()
        for question in job["selected_rubrics"] if question["subtype"] == "exists"
    }
    all_original_event_labels = {
        event_label(event, job["language"]).casefold() for event in events.values()
    }
    selected_attribute_phrases = {
        attribute_phrase(question["construction_evidence"]["attribute"], job["language"]).casefold()
        for question in job["selected_rubrics"] if question["subtype"] == "speech_attribute"
    }
    used_attribute_phrases: set[str] = set()
    repairs = []

    def take_label(old: str) -> str:
        for item in candidates:
            label = str(item["label"])
            folded = label.casefold()
            if (
                folded != old.casefold()
                and folded not in used_candidate_labels
                and folded not in all_original_event_labels
                and not any(target in folded or folded in target for target in selected_event_labels)
            ):
                used_candidate_labels.add(label.casefold())
                return label
        raise ValueError(f"no event label candidate for {job['sample_id']}:{old}")

    def take_speech(event: dict[str, Any]) -> dict[str, Any]:
        original_attr_count = len(event.get("selected_caption_attrs") or [])
        original_attr_phrases = {
            attribute_phrase(attribute, job["language"]).casefold()
            for attribute in event.get("selected_caption_attrs") or []
            if attribute_phrase(attribute, job["language"])
        }
        choices = [
            item for item in speech_pool
            if item.get("language") == job["language"]
            and str(item.get("speaker_description") or "").casefold()
                != str(event.get("speaker_description") or "").casefold()
            and str(item.get("text") or "").casefold() != str(event.get("text") or "").casefold()
        ]
        if not choices:
            raise ValueError(f"no speech candidate for {job['sample_id']}:{event['event_id']}")
        # Matching the number of caption attributes keeps the replacement
        # sentence similarly detailed, but it is a preference rather than a
        # validity condition.  Small frozen pools can legitimately contain no
        # same-cardinality candidate.  In that case prefer the candidate with
        # the fewest surviving attribute phrases, then the closest cardinality.
        # ``stable_rows`` supplies a deterministic final tie-breaker.
        stable_choices = stable_rows(choices, job["sample_id"] + ":speech_event")
        return min(
            stable_choices,
            key=lambda item: (
                len(item.get("selected_caption_attrs") or []) != original_attr_count,
                len(original_attr_phrases & {
                    attribute_phrase(attribute, job["language"]).casefold()
                    for attribute in item.get("selected_caption_attrs") or []
                    if attribute_phrase(attribute, job["language"])
                }),
                abs(len(item.get("selected_caption_attrs") or []) - original_attr_count),
                stable_choices.index(item),
            ),
        )

    # Event targets: erase the selected original semantic identity.  A speech
    # target stays grammatically speech-like by using another training-pool
    # speaker and transcript; non-speech targets use the frozen dynamic pool.
    for question in job["selected_rubrics"]:
        if question["subtype"] != "exists":
            continue
        event = events[question["construction_evidence"]["event_id"]]
        old_label = event_label(event, job["language"])
        if old_label.casefold() not in prompt_prime.casefold():
            continue
        if event["role"] == "speech":
            chosen = take_speech(event)
            prompt_prime, count_label = replace_event_label(
                prompt_prime, old_label, str(chosen["speaker_description"]),
                (event_label(item, job["language"]) for item in events.values()),
            )
            prompt_prime, count_text = replace_all_casefold(
                prompt_prime, str(event.get("text") or ""), str(chosen["text"])
            )
            if count_text == 0:
                prompt_prime, count_text = replace_normalized_transcript(
                    prompt_prime, str(event.get("text") or ""), str(chosen["text"])
                )
            repairs.append({
                "rubric_id": question["rubric_id"], "kind": "speech_event_identity",
                "replacement_source_dataset": chosen["source_dataset"],
                "replacement_source_id": chosen["uid"],
                "original_attribute_count": len(event.get("selected_caption_attrs") or []),
                "replacement_attribute_count": len(chosen.get("selected_caption_attrs") or []),
                "replacements": count_label + count_text,
            })
        else:
            replacement = take_label(old_label)
            prompt_prime, count = replace_event_label(
                prompt_prime, old_label, replacement,
                (event_label(item, job["language"]) for item in events.values()),
            )
            repairs.append({
                "rubric_id": question["rubric_id"], "kind": "event_identity",
                "replacement_label": replacement, "replacements": count,
            })

    # Speech content and attributes are direct only when their speech parent was
    # not removed above.  Replace any original fact still left by the LLM.
    for question in job["selected_rubrics"]:
        subtype = question["subtype"]
        evidence = question["construction_evidence"]
        event_id = evidence.get("event_id")
        if event_id in selected_exists:
            continue
        if subtype == "speech_content":
            old = str(evidence["transcript"])
            choices = [item for item in speech_text_candidates if str(item["text"]).casefold() != old.casefold()]
            if not choices:
                choices = [
                    item for item in speech_pool
                    if item.get("language") == job["language"]
                    and str(item.get("text") or "").casefold() != old.casefold()
                ]
            chosen = stable_rows(choices, job["sample_id"] + ":speech_text")[0]
            prompt_prime, count = replace_all_casefold(prompt_prime, old, str(chosen["text"]))
            if count == 0:
                prompt_prime, count = replace_normalized_transcript(
                    prompt_prime, old, str(chosen["text"])
                )
            if count == 0:
                raise ValueError(f"speech transcript span not found for {job['sample_id']}")
            repairs.append({
                "rubric_id": question["rubric_id"], "kind": "speech_content",
                "replacement_source_dataset": chosen["source_dataset"],
                "replacement_source_id": chosen.get("uid") or chosen.get("source_id"),
                "replacements": count,
            })
        elif subtype == "speech_attribute":
            attribute = evidence["attribute"]
            old = attribute_phrase(attribute, job["language"])
            if old.casefold() in prompt_prime.casefold():
                choices = []
                for item in speech_pool:
                    if item.get("language") != job["language"]:
                        continue
                    for candidate_attr in item.get("selected_caption_attrs") or []:
                        if (
                            str(candidate_attr.get("value")) != str(attribute.get("value"))
                            and attribute_phrase(candidate_attr, job["language"])
                            and attribute_phrase(candidate_attr, job["language"]).casefold()
                                not in selected_attribute_phrases
                            and attribute_phrase(candidate_attr, job["language"]).casefold()
                                not in used_attribute_phrases
                        ):
                            choices.append({**candidate_attr, "source": item})
                if not choices:
                    raise ValueError(
                        f"no attribute replacement for {job['sample_id']}:{question['rubric_id']}"
                    )
                choices = sorted(
                    choices,
                    key=lambda item: (
                        str(item.get("name")) != str(attribute.get("name")),
                        hashlib.sha256(
                            (job["sample_id"] + ":" + question["rubric_id"] + ":"
                             + str(item["source"].get("uid"))).encode("utf-8")
                        ).hexdigest(),
                    ),
                )
                chosen = choices[0]
                used_attribute_phrases.add(attribute_phrase(chosen, job["language"]).casefold())
                prompt_prime, count = replace_all_casefold(
                    prompt_prime, old, attribute_phrase(chosen, job["language"])
                )
                repairs.append({
                    "rubric_id": question["rubric_id"], "kind": "speech_attribute",
                    "old_value": attribute["value"], "new_value": chosen["value"],
                    "new_attribute": {
                        "name": chosen.get("name"),
                        "value": chosen.get("value"),
                        "caption_text": chosen.get("caption_text"),
                    },
                    "replacement_source_dataset": chosen["source"]["source_dataset"],
                    "replacement_source_id": chosen["source"]["uid"],
                    "replacements": count,
                })

    # Fail closed if any directly selected textual fact survived.
    remaining = []
    for question in job["selected_rubrics"]:
        subtype = question["subtype"]
        evidence = question["construction_evidence"]
        event_id = evidence.get("event_id")
        if subtype == "exists":
            needle = event_label(events[event_id], job["language"])
        elif subtype == "speech_content" and event_id not in selected_exists:
            needle = str(evidence["transcript"])
        elif subtype == "speech_attribute" and event_id not in selected_exists:
            needle = attribute_phrase(evidence["attribute"], job["language"])
        else:
            continue
        if subtype == "exists":
            survived = event_label_present(
                prompt_prime, needle,
                (event_label(item, job["language"]) for item in events.values()),
            )
        else:
            survived = bool(needle and needle.casefold() in prompt_prime.casefold())
        if survived:
            remaining.append(question["rubric_id"])
    if remaining:
        raise ValueError(f"selected facts survived semantic repair for {job['sample_id']}: {remaining}")

    if repairs:
        row.update({
            "prompt_prime": prompt_prime,
            "patches": [{
                "original_span": job["prompt"], "replacement_span": prompt_prime,
                "changed_rubric_ids": job["selected_rubric_ids"],
                "start": 0, "end": len(job["prompt"]), "patch_order": 0,
                "new_start": 0, "new_end": len(prompt_prime),
            }],
            "model": "deterministic_atomic_prompt_edit",
            "outside_patch_equality": True,
            "note": "Original caption with fail-closed deterministic edits limited to selected atomic facts.",
        })
    audit = {
        "sample_id": job["sample_id"], "repair_count": len(repairs), "repairs": repairs,
        "remaining_selected_textual_facts": remaining, "pass": not remaining,
        "authoritative_model": row.get("model"),
    }
    return row, audit


def flip_selected_relation(
    job: dict[str, Any], row: dict[str, Any], pair: dict[str, Any], audit: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make the selected all-three relation false when neither endpoint was removed.

    The nearest textual occurrences of the two endpoint labels delimit the
    relation connector.  Replacing only that connector preserves both event
    identities and every byte outside the selected local relation span.
    """
    questions = [
        question for question in job["selected_rubrics"]
        if question["subtype"] in {"before", "overlap"}
    ]
    if not questions:
        return row, audit
    if len(questions) != 1:
        raise ValueError(f"expected one all-three relation for {job['sample_id']}")
    question = questions[0]
    evidence = question["construction_evidence"]
    selected_exists = {
        item["construction_evidence"]["event_id"]
        for item in job["selected_rubrics"] if item["subtype"] == "exists"
    }
    endpoints = {evidence["source_event_id"], evidence["target_event_id"]}
    if selected_exists & endpoints:
        audit.update({
            "selected_relation_realization": "false_via_selected_endpoint_identity_removal",
            "selected_relation_rubric_id": question["rubric_id"],
        })
        return row, audit

    events = {event["event_id"]: event for event in pair["events"]}
    source_label = event_label(events[evidence["source_event_id"]], job["language"])
    target_label = event_label(events[evidence["target_event_id"]], job["language"])
    text = str(row["prompt_prime"])

    def occurrences(label: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(re.escape(label), text, re.I)]

    source_occurrences = occurrences(source_label)
    target_occurrences = occurrences(target_label)
    if not source_occurrences or not target_occurrences:
        raise ValueError(
            f"relation endpoint missing for {job['sample_id']}: {source_label!r}, {target_label!r}"
        )
    _, source_span, target_span = min(
        (abs(source[0] - target[0]), source, target)
        for source in source_occurrences for target in target_occurrences
    )
    if source_span[0] < target_span[0]:
        left_span, right_span = source_span, target_span
        if question["subtype"] == "before":
            connector = "晚于" if job["language"] == "zh" else " occurs after "
        else:
            connector = "早于" if job["language"] == "zh" else " occurs before "
    else:
        left_span, right_span = target_span, source_span
        if question["subtype"] == "before":
            connector = "早于" if job["language"] == "zh" else " occurs before "
        else:
            connector = "早于" if job["language"] == "zh" else " occurs before "
    old_connector = text[left_span[1]:right_span[0]]
    if not old_connector:
        raise ValueError(f"empty relation connector for {job['sample_id']}")
    prompt_prime = text[:left_span[1]] + connector + text[right_span[0]:]
    updated = dict(row)
    updated.update({
        "prompt_prime": prompt_prime,
        "patches": [{
            "original_span": job["prompt"], "replacement_span": prompt_prime,
            "changed_rubric_ids": job["selected_rubric_ids"],
            "start": 0, "end": len(job["prompt"]), "patch_order": 0,
            "new_start": 0, "new_end": len(prompt_prime),
        }],
        "model": "deterministic_atomic_prompt_edit",
        "outside_patch_equality": True,
        "note": "Original caption with selected atomic facts changed and the selected relation replaced by a closed-set temporal relation.",
    })
    audit["repairs"].append({
        "rubric_id": question["rubric_id"], "kind": "relation_reversal",
        "subtype": question["subtype"], "old_connector": old_connector,
        "new_connector": connector, "source_label": source_label, "target_label": target_label,
    })
    audit["repair_count"] = len(audit["repairs"])
    audit.update({
        "selected_relation_realization": "explicit_local_relation_reversal",
        "selected_relation_rubric_id": question["rubric_id"],
        "authoritative_model": updated["model"],
    })
    return updated, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-pairs", type=Path, required=True)
    parser.add_argument("--speech-pool", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs)
    positive_pairs = {row["id"]: row for row in read_jsonl(args.positive_pairs)}
    speech_pool = read_jsonl(args.speech_pool)
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.generated) if args.generated.exists() else []:
        if row.get("status") == "ok":
            current = latest.get(row["sample_id"])
            # Relation-only P' has a deterministic semantic construction.  It
            # remains authoritative even if an already-running LLM retry appends
            # a later syntactically valid but semantically weaker row.
            deterministic_models = {
                "deterministic_relation_reversal", "deterministic_missing_job_seed",
            }
            if current and current.get("model") in deterministic_models \
                    and row.get("model") not in deterministic_models:
                continue
            latest[row["sample_id"]] = row
    deterministic_seeded = []
    for job in jobs:
        if job["sample_id"] in latest or job["path2_perturbation_category"] == "relation":
            continue
        latest[job["sample_id"]] = deterministic_missing_job_seed(job)
        deterministic_seeded.append(job["sample_id"])
    missing = [job["sample_id"] for job in jobs if job["sample_id"] not in latest]
    if missing and not args.allow_partial:
        raise SystemExit(f"missing {len(missing)} valid generated rows: {missing[:8]}")

    authoritative = {}
    semantic_audits = []
    for job in jobs:
        generated = latest.get(job["sample_id"])
        if generated is None:
            continue
        # Preserve relation-only deterministic reversal.  Every other branch is
        # rebuilt from the positive caption so an LLM cannot alter unselected
        # facts as a side effect of paraphrasing.
        if job["path2_perturbation_category"] != "relation":
            generated = {
                **generated,
                "prompt": job["prompt"],
                "prompt_prime": job["prompt"],
                "model": "deterministic_atomic_prompt_edit",
                "note": "Seeded from the original caption; only selected facts may change.",
            }
        repaired, audit = repair_direct_targets(
            job, generated, positive_pairs[job["parent_positive_id"]], speech_pool
        )
        if job["path2_perturbation_category"] == "all_three":
            repaired, audit = flip_selected_relation(
                job, repaired, positive_pairs[job["parent_positive_id"]], audit
            )
        authoritative[job["sample_id"]] = repaired
        semantic_audits.append(audit)
    write_jsonl(args.output_dir / "generated_authoritative.jsonl", (
        authoritative[job["sample_id"]] for job in jobs if job["sample_id"] in authoritative
    ))
    write_jsonl(args.output_dir / "semantic_repair_audit.jsonl", semantic_audits)

    pairs = []
    rubric_manifest = []
    for job in jobs:
        generated = authoritative.get(job["sample_id"])
        if generated is None:
            continue
        audio_path = Path(job["audio_path"])
        observed_audio_hash = file_sha256(audio_path)
        if observed_audio_hash != job["parent_audio_sha256"]:
            raise ValueError(f"parent audio changed: {job['parent_positive_id']}")
        prompt_prime = generated["prompt_prime"]
        if prompt_prime == job["prompt"]:
            raise ValueError(f"P' equals P: {job['negative_id']}")
        pair = {
            "schema": "route2_prompt_side_negative_pair",
            "sample_id": job["negative_id"],
            "negative_id": job["negative_id"],
            "parent_positive_id": job["parent_positive_id"],
            "route": "path2_prompt_side_negative",
            "positive_pair": False,
            "audio_path": str(audio_path),
            "audio_sha256": observed_audio_hash,
            "audio_byte_identical_to_parent": True,
            "caption": prompt_prime,
            "original_caption": job["prompt"],
            "caption_sha256": text_sha256(prompt_prime),
            "language": job["language"],
            "perturbation_category": job["path2_perturbation_category"],
            "intended_consistency_level": job["intended_consistency_level"],
            "construction_target_yes_range": job["construction_target_yes_range"],
            "selected_original_rubric_ids": job["selected_rubric_ids"],
            "hierarchy_affected_original_rubric_ids": job["hierarchy_affected_rubric_ids"],
            "patches": generated["patches"],
            "outside_patch_equality": generated.get("outside_patch_equality") is True,
            "generator_attempts": generated.get("attempts"),
            "generator_model": generated.get("model"),
            "generator_note": generated.get("note"),
            "sampling_seed": job["sampling_seed"],
        }
        pairs.append(pair)
        rubric_manifest.append({
            "sample_id": job["negative_id"],
            "prompt": prompt_prime,
            "rubric_source_prompt": prompt_prime,
            "prompt_kind": "P_prime",
            "route": "path2_prompt_side_negative",
            "variant": job["negative_id"],
            "audio_path": str(audio_path),
            "metadata": {
                "language": job["language"],
                "parent_positive_id": job["parent_positive_id"],
                "original_prompt": job["prompt"],
                "patches": generated["patches"],
                "outside_patch_equality": True,
                "selected_original_rubric_ids": job["selected_rubric_ids"],
            },
        })

    pairs.sort(key=lambda row: row["negative_id"])
    rubric_manifest.sort(key=lambda row: row["sample_id"])
    write_jsonl(args.output_dir / "pairs.jsonl", pairs)
    write_jsonl(args.output_dir / "rubric_manifest.jsonl", rubric_manifest)
    summary = {
        "schema": "route2_prompt_side_negative_summary",
        "jobs": len(jobs),
        "materialized": len(pairs),
        "missing": len(missing),
        "all_audio_byte_identical": all(row["audio_byte_identical_to_parent"] for row in pairs),
        "all_outside_patch_equality": all(row["outside_patch_equality"] for row in pairs),
        "semantic_direct_target_pass": all(row["pass"] for row in semantic_audits),
        "semantic_repaired_samples": sum(row["repair_count"] > 0 for row in semantic_audits),
        "semantic_repairs": sum(row["repair_count"] for row in semantic_audits),
        "deterministic_seeded_samples": len(deterministic_seeded),
        "authoritative_generation_sha256": file_sha256(args.output_dir / "generated_authoritative.jsonl"),
        "semantic_repair_audit_sha256": file_sha256(args.output_dir / "semantic_repair_audit.jsonl"),
        "languages": dict(Counter(row["language"] for row in pairs)),
        "categories": dict(Counter(row["perturbation_category"] for row in pairs)),
        "pairs_sha256": file_sha256(args.output_dir / "pairs.jsonl"),
        "rubric_manifest_sha256": file_sha256(args.output_dir / "rubric_manifest.jsonl"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
