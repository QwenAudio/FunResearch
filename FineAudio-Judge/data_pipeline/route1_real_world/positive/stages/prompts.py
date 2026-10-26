"""Prompt builders used by the real-world audio positive-data pipeline.

Each builder has one model task and an explicit output contract. Selection
thresholds come from the resolved run configuration; prompts must not silently
introduce additional fixed gates.
"""

from __future__ import annotations

import json
from typing import Any


def _prompt_json(value: Any, max_chars: int) -> str:
    """Serialize auxiliary evidence and expose any context truncation."""
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[auxiliary evidence truncated]"


AUDIO_EVIDENCE_RULES = """
Describe only content that is directly audible. Do not treat visible but silent
objects, actions, or locations as auditory facts.
If auxiliary subtitles conflict with the audio, follow the audio. Mark uncertain
content as uncertain and do not invent missing dialogue.
Preserve intelligible verbatim speech, speaker count and turn changes, vocal
delivery, ambience, music, sound effects, temporal order, and overlap.
Do not add color, appearance, camera motion, smell, silent actions, or other
details that cannot be verified from audio.
Keep Chinese or English speech in its original language. If wording is unclear,
mark it as unintelligible or describe it without quotation marks.
""".strip()


AUDIO_ONLY_RULES = f"""
{AUDIO_EVIDENCE_RULES}
Do not estimate exact timestamps. If reliable auxiliary evidence already contains
machine timecodes such as 00:06 or 01:02:03, they may be retained in the raw
verification caption; a later deterministic step converts narrative timecodes to
natural relative timing. Never drop event order or overlap merely to remove a
timecode.
Write a detailed, temporally coherent audio-only caption suitable for training a
long-form audio generation model.
""".strip()


def candidate_screening_prompt(
    source: dict[str, Any], selection_rules: dict[str, Any] | None = None
) -> str:
    """Build the first-pass audio screening request."""
    rules = selection_rules or {}
    policy = {
        "allowed_speech_languages": rules.get(
            "allowed_speech_languages", ["zh", "en", "mixed_zh_en"]
        ),
        "allow_no_speech": bool(rules.get("allow_no_speech", True)),
        "reject_profanity": bool(rules.get("reject_profanity", True)),
        "reject_corrupted_audio": bool(rules.get("reject_corrupted_audio", True)),
        "reject_corrupted_or_repetitive_asr": bool(
            rules.get("reject_corrupted_or_repetitive_asr", True)
        ),
        "speech_min_non_speech_event_types": int(
            rules.get("speech_min_non_speech_event_types", 1)
        ),
        "no_speech_min_non_speech_event_types": int(
            rules.get("no_speech_min_non_speech_event_types", 2)
        ),
    }
    auxiliary = {
        "source_asr": source.get("source_asr"),
        "source_metadata": source.get("source_meta")
        or source.get("finevideo_visual_metadata"),
    }
    return f"""
Listen to the complete attached audio. Record evidence about language, speech,
non-speech events, temporal development, clarity, and content safety for later
candidate selection. The audio is authoritative; auxiliary ASR and source
metadata are not auditory evidence.

Selection policy for this run:
{_prompt_json(policy, 4000)}

Evidence rules:
1. Record only directly audible events. Do not infer sounds from titles, visual
   context, ASR, or common sense.
2. Count a brief event only when it is clearly identifiable. Do not split
   indistinguishable continuous noise into several event types.
3. Set decision according to the policy above. A later deterministic selector
   will reapply the policy to the structured fields.
4. Use a 0--5 scale for all scores: 0 means absent or unusable, 3 acceptable,
   and 5 exceptionally clear or rich.

Untrusted auxiliary evidence, used only to cross-check language and possible
speech:
{_prompt_json(auxiliary, 16000)}

Return exactly one valid JSON object:
{{
  "language": {{
    "primary": "zh|en|mixed_zh_en|other|no_speech|uncertain",
    "confidence": 0.0,
    "evidence": "brief evidence"
  }},
  "speech": {{
    "ratio": 0.0,
    "continuous_monologue_max_seconds": 0.0,
    "style": "acted_dialogue|event_commentary|lecture|tutorial|podcast|news|talking_head|single_speaker_narration|interview|none|other",
    "speaker_count": 0
  }},
  "audible_event_types": ["distinct audible event type"],
  "non_speech_event_types": ["distinct audible non-speech event type"],
  "temporal_stages": [
    {{"order": 1, "description": "audible change in this stage"}}
  ],
  "max_simultaneous_layers": 0,
  "scene": {{
    "category": "acted_drama|action|sports|live_complex|event_driven|speech_program|other",
    "acted_or_event_driven": true,
    "speech_dominated": false
  }},
  "content_safety": {{
    "contains_profanity": false,
    "evidence_source": "audio|asr|both|none",
    "reason": "state whether profanity was detected without repeating it"
  }},
  "quality": {{
    "overall_clarity_score": 0.0,
    "dialogue_clarity_score": 0.0,
    "event_boundary_clarity_score": 0.0,
    "asr_reliability_score": 0.0,
    "narrative_event_coherence_score": 0.0,
    "audio_corrupted": false,
    "asr_obviously_corrupted_or_repetitive": false,
    "reasons": ["evidence about clarity and verifiability"]
  }},
  "complexity_score": 0.0,
  "rank_tier": "priority|standard|low_priority|reject",
  "decision": "pass|reject",
  "reasons": ["audible evidence supporting selection or rejection"]
}}

Use values from 0 to 1 for ratio and confidence. Do not return a caption or any
text outside the JSON object.
""".strip()


def non_speech_verification_prompt(
    source: dict[str, Any],
    screening_assessment: Any,
    selection_rules: dict[str, Any] | None = None,
) -> str:
    """Build an independent relistening request for non-speech evidence."""
    rules = selection_rules or {}
    minimum_confidence = float(rules.get("verified_event_min_confidence", 0.85))
    allowed_prominence = rules.get("verified_event_allowed_prominence") or [
        "clear_foreground",
        "clear_background",
    ]
    prominence_text = "/".join(str(value) for value in allowed_prominence)
    speech_min_events = int(rules.get("speech_min_non_speech_event_types", 1))
    no_speech_min_events = int(rules.get("no_speech_min_non_speech_event_types", 2))
    policy = {
        "allowed_speech_languages": rules.get(
            "allowed_speech_languages", ["zh", "en", "mixed_zh_en"]
        ),
        "allow_no_speech": bool(rules.get("allow_no_speech", True)),
        "reject_profanity": bool(rules.get("reject_profanity", True)),
        "reject_pure_or_near_pure_speech": bool(
            rules.get("reject_pure_or_near_pure_speech", True)
        ),
        "reject_indistinguishable_static_noise": bool(
            rules.get("reject_indistinguishable_static_noise", True)
        ),
        "speech_min_non_speech_event_types": speech_min_events,
        "no_speech_min_non_speech_event_types": no_speech_min_events,
        "verified_event_min_confidence": minimum_confidence,
        "verified_event_allowed_prominence": allowed_prominence,
    }
    return f"""
Listen to the complete audio again without trusting the first-pass decision.
Independently verify non-speech events and record evidence about pure speech,
static noise, language, and content safety. Do not write a caption.

Selection policy for this run:
{_prompt_json(policy, 5000)}

The first-pass assessment is an unverified claim:
{_prompt_json(screening_assessment, 16000)}

Auxiliary ASR may be used only to assess speech coverage, language, and profanity.
It cannot prove any non-speech sound:
{_prompt_json(source.get("source_asr"), 14000)}

Return exactly one valid JSON object:
{{
  "speech_language": "zh|en|mixed_zh_en|other|no_speech|uncertain",
  "speech_ratio": 0.0,
  "longest_continuous_speech_seconds": 0.0,
  "verified_non_speech_events": [
    {{
      "event_type": "specific sound type",
      "relative_stage": "beginning|early|middle|late|ending|throughout|uncertain",
      "occurrence_pattern": "single|repeated|continuous|uncertain",
      "audible_evidence": "directly audible acoustic characteristics and changes",
      "prominence": "clear_foreground|clear_background|faint|uncertain",
      "confidence": 0.0,
      "changes_over_time": true
    }}
  ],
  "rejected_sound_claims": ["first-pass claim not independently confirmed"],
  "pure_or_near_pure_speech": false,
  "background_too_noisy_or_masked": false,
  "has_distinguishable_temporal_development": true,
  "content_safety": {{
    "contains_profanity": false,
    "reason": "do not repeat profanity"
  }},
  "decision": "pass|reject",
  "reasons": ["evidence from this independent listening pass"]
}}

Rules:
1. Confidence and prominence must reflect this listening pass. Do not raise
   confidence or merge sound types merely to meet a configured count.
2. A later deterministic selector counts only events with confidence at least
   {minimum_confidence:.2f} and prominence in {prominence_text}. It then applies
   the configured minimum of {speech_min_events} non-speech types when speech is
   present and {no_speech_min_events} when no speech is present.
3. Base speech_language only on audible speech; use no_speech when none exists.
4. Report pure speech, noise masking, temporal development, and profanity
   truthfully; decision must follow the policy above.
5. Do not repeat profanity. Return JSON only, with no explanation or markdown.
""".strip()


def detective_chronology_prompt() -> str:
    return f"""
Conduct a first complete listening pass and record all audible events in temporal
order.
{AUDIO_EVIDENCE_RULES}
Do not estimate exact timestamps. Existing reliable machine timecodes may be
retained only to cross-check order and overlap.
The goal is complete coverage of audible facts and their temporal relationships
for later verification and synthesis. Use relative stages and event-order
numbers. Prioritize factual coverage over literary prose.
""".strip()


def detective_speech_prompt(
    source_asr: Any,
    trusted_time_annotations: Any = None,
) -> str:
    return f"""
Conduct a second complete listening pass focused only on speech.
For each segment, record relative position, speaker or turn change, intelligible
verbatim wording, language, delivery, pauses, interruptions, and overlap.
Keep intelligible wording in the original spoken language. Do not translate or
paraphrase it.
Candidate ASR is only a reference and must be checked against the audio. Correct
clear recognition errors from the audio and never invent unclear wording.
Do not estimate exact timestamps. Optional reference timing may be used only to
cross-check order, overlap, and turns.

Candidate ASR:
{json.dumps(source_asr, ensure_ascii=False)}

Optional reference timing:
{json.dumps(trusted_time_annotations or [], ensure_ascii=False)}
""".strip()


def detective_sound_prompt() -> str:
    return """
Conduct a third complete listening pass focused on non-speech audio evidence.
Record directly audible ambience, music, non-linguistic vocalizations, object or
action sounds, mechanical or traffic sounds, sound effects, and audible changes
in distance or intensity.
Describe their order, duration, and overlap with speech using relative stages
rather than exact timestamps.
Record only content that can be confirmed from audio. Mark unreliable
identifications as uncertain and do not infer sounds from context.
""".strip()


def detective_synthesis_prompt(
    observations: dict[str, str],
    trusted_time_annotations: Any = None,
    verified_non_speech: Any = None,
) -> str:
    return f"""
Act as the final verifier. Re-listen to the complete attached audio and merge the
three candidate investigations into one detailed Chinese audio-only caption.

{AUDIO_ONLY_RULES}

All candidate investigations may contain errors:
[Chronology]
{observations["chronology"]}
[Speech verification]
{observations["speech"]}
[Non-speech verification]
{observations["sound"]}

[Allowlist of independently verified non-speech events]
{json.dumps(verified_non_speech or [], ensure_ascii=False)[:12000]}

Return only the caption text. If a persistent sound bed exists, describe it
before advancing through the temporal sequence. Quote speech as accurately as
possible and preserve transitions, overlap, and the ending.
Do not write phrases such as "the image shows" or "the video displays."
Do not estimate machine timecodes. Reliable timecodes already present in
candidate evidence may be retained, but never at the cost of audible facts,
order, or overlap.

Non-speech constraint: only sound types in the independent allowlist may appear
in the caption. Discard non-speech claims from the third pass that were not
independently confirmed. You may refine the order, prominence, and overlap of an
allowlisted event, but may not introduce a new sound type.

Optional reference timing, used only to cross-check order and overlap:
{json.dumps(trusted_time_annotations or [], ensure_ascii=False)}
""".strip()


def caption_coverage_prompt(
    caption: str,
    investigation_rounds: Any,
    source_asr: Any,
    trusted_time_annotations: Any = None,
    verified_non_speech: Any = None,
    previous_gate: Any = None,
) -> str:
    previous = (
        ""
        if previous_gate is None
        else "\nPrevious gate result; repair its reported problems:\n"
        + json.dumps(previous_gate, ensure_ascii=False)[:12000]
    )
    return f"""
Independently verify the coverage and content safety of the current audio
caption. If audible evidence is missing, restore it using only the supplied
evidence. Never introduce a sound from visual context or common sense.

Current Chinese audio-only caption:
{caption}

Multi-pass audio investigation:
{json.dumps(investigation_rounds, ensure_ascii=False)[:24000]}

Source ASR candidate:
{json.dumps(source_asr, ensure_ascii=False)[:12000]}

Optional reference timing, used only to cross-check order and overlap:
{json.dumps(trusted_time_annotations or [], ensure_ascii=False)[:8000]}

Allowlist of independently verified non-speech events:
{json.dumps(verified_non_speech or [], ensure_ascii=False)[:12000]}
{previous}

Return valid JSON only:
{{
  "audio_caption": "complete revised Chinese audio-only caption",
  "coverage_checks": [
    {{
      "evidence": "specific audible fact being checked",
      "covered": true,
      "caption_evidence": "exact contiguous text from audio_caption"
    }}
  ],
  "omitted_audible_evidence": [],
  "contains_profanity": false
}}

Rules:
1. Check chronology, speech, and non-speech investigations separately. Do not
   omit clearly supported non-speech events or temporal relationships.
2. caption_evidence must be an exact contiguous span from audio_caption.
3. Do not force conflicting or explicitly uncertain candidates into the
   caption. Non-speech content must come from the allowlist; remove unsupported
   non-speech claims.
4. Set contains_profanity=true if the investigation, ASR, or caption contains
   profanity. Do not hide it through masking, deletion, or euphemism.
5. Keep the caption strictly audio-only. Return no markdown or explanation.
6. Reliable machine timecodes already present in the caption or evidence may be
   retained here; a later deterministic step converts them to relative timing.
""".strip()


PROMPT_BUILDERS = {
    "candidate_screening_prompt": candidate_screening_prompt,
    "non_speech_verification_prompt": non_speech_verification_prompt,
    "detective_chronology_prompt": detective_chronology_prompt,
    "detective_speech_prompt": detective_speech_prompt,
    "detective_sound_prompt": detective_sound_prompt,
    "detective_synthesis_prompt": detective_synthesis_prompt,
    "caption_coverage_prompt": caption_coverage_prompt,
}
