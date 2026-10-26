"""Tests for the public Route 1 controlled-negative contract."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline.route1_real_world.negative.stages import counterfactual_pipeline as pipeline


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class Route1NegativeTest(unittest.TestCase):
    def test_source_exact_bilingual_gloss_does_not_change_question_frame_language(self) -> None:
        prompt = 'He says "Good luck" (祝你好运), then music starts.'
        question = 'Does speech overlap music while he says "Good luck" (祝你好运)?'
        self.assertEqual(
            pipeline.question_frame_language_with_prompt(question, prompt), "en",
        )

    def test_question_basis_coverage_rejects_unrelated_short_evidence(self) -> None:
        question = "Does the piano piece play after the mechanical hum fades?"
        self.assertLess(
            pipeline.question_basis_coverage(question, "the space. After the"), 0.30,
        )
        self.assertGreaterEqual(
            pipeline.question_basis_coverage(
                question, "After the mechanical hum fades, the piano piece plays",
            ),
            0.30,
        )
        self.assertIn(
            "music",
            pipeline.detected_event_families("Does piano play after the hum fades?"),
        )
        self.assertNotIn(
            "music", pipeline.detected_event_families("After the hum fades"),
        )

    def test_exact_model_basis_determines_authoritative_offsets(self) -> None:
        prompt = "A bell rings. Then a dog barks."
        normalized = pipeline.normalize_rebuild_offsets(prompt, {
            "rubric_id": "R1", "basis": "Then a dog barks.",
            "basis_start": 0, "basis_end": 3,
        })
        self.assertEqual(prompt[normalized["basis_start"]:normalized["basis_end"]], "Then a dog barks.")
        self.assertEqual(normalized["basis_resolution"], "exact_unique_model_basis")

    def test_following_content_does_not_override_simultaneous_relation(self) -> None:
        text = "Simultaneously, a speaker delivers the following content in sequence."
        self.assertEqual(pipeline.relation_subtype_from_question(text), "overlap")

    def test_latest_failure_is_reused_when_generation_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            write_jsonl(path, [
                {"sample_id": "one", "status": "error", "error": "first"},
                {"sample_id": "other", "status": "error", "error": "ignored"},
                {"sample_id": "one", "status": "error", "error": "latest"},
            ])
            errors = pipeline.latest_failure_errors(path, {"one": {}})
            self.assertEqual(errors, {"one": "latest"})

    def test_public_manifests_join_and_prepare(self) -> None:
        caption = "A loud bell rings before a dog barks while soft rain continues."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.write_bytes(b"test-audio-bytes")
            positive = root / "positive.jsonl"
            rubrics = root / "rubrics.jsonl"
            write_jsonl(positive, [{
                "id": "one", "dataset": "demo", "audio_path": str(audio),
                "audio_duration": 12.0, "audio_caption": caption,
            }])
            write_jsonl(rubrics, [{
                "sample_id": "one", "dataset": "demo", "prompt": caption,
                "questions": [
                    {"rubric_id": "R1", "axis": "content_fidelity", "scope": "local",
                     "domain": ["sound_effect"], "subtype": "exists",
                     "question": "Is a bell audible?", "basis": "A loud bell rings",
                     "basis_start": 0, "basis_end": 17, "target_ids": ["EV1"],
                     "parent_ids": []},
                    {"rubric_id": "R2", "axis": "acoustic_presentation_fidelity",
                     "scope": "local", "domain": ["sound_effect"], "subtype": "loudness",
                     "question": "Is the bell loud?", "basis": "loud bell",
                     "basis_start": 2, "basis_end": 11, "target_ids": ["EV1"],
                     "parent_ids": ["R1"]},
                    {"rubric_id": "R3", "axis": "content_fidelity", "scope": "local",
                     "domain": ["animal"], "subtype": "exists",
                     "question": "Is a dog bark audible?", "basis": "a dog barks",
                     "basis_start": 25, "basis_end": 36, "target_ids": ["EV2"],
                     "parent_ids": []},
                    {"rubric_id": "R4", "axis": "temporal_structural_fidelity",
                     "scope": "relational", "domain": ["sound_effect", "animal"],
                     "subtype": "before", "question": "Does the bell ring before the dog barks?",
                     "basis": "A loud bell rings before a dog barks", "basis_start": 0,
                     "basis_end": 36, "target_ids": ["EV1", "EV2"],
                     "parent_ids": ["R1", "R3"]},
                    {"rubric_id": "R5", "axis": "content_fidelity", "scope": "local",
                     "domain": ["ambience"], "subtype": "exists",
                     "question": "Is rain audible?", "basis": "soft rain continues",
                     "basis_start": 43, "basis_end": 62, "target_ids": ["EV3"],
                     "parent_ids": []},
                ],
            }])
            adapted = pipeline.adapt_public_inputs(positive, rubrics)
            self.assertEqual(adapted[0]["sample_id"], "demo:one")
            self.assertEqual(adapted[0]["caption"], caption)
            args = argparse.Namespace(
                run=root / "run", input=None, positive_manifest=positive,
                rubric_manifest=rubrics, expected_sources=0, target_count=1,
                skip_audio_hash=False,
            )
            summary = pipeline.prepare(args)
            self.assertEqual(summary["source_rows"], 1)
            self.assertEqual(summary["planned_negatives"], 1)

    def _plan(self, rubric: dict, prompt: str) -> dict:
        return {
            "original_prompt": prompt,
            "direct_target_rubrics": [rubric],
            "direct_target_rubric_ids": [rubric["rubric_id"]],
            "retained_rubrics": [],
            "intended_perturbation_strength": "medium",
        }

    def test_attribute_cannot_change_event_family(self) -> None:
        prompt = "Background orchestral music plays continuously."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "placement",
            "domain": ["music"], "question": "Is the music in the background?",
            "basis": "Background orchestral music", "parent_ids": ["ROOT"],
        }
        patch = {
            "rubric_ids": ["R1"], "original_span": "orchestral music",
            "replacement_span": "industrial machinery", "edit_kind": "attribute",
            "original_event_family": "music", "replacement_event_family": "mechanical",
            "semantic_distance": "high", "contradiction_explanation": "R1 is contradicted.",
            "audible_test": "Listen for industrial machinery.",
        }
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        with self.assertRaisesRegex(ValueError, "attribute_changes_event_identity"):
            pipeline.validate_patch_result(self._plan(rubric, prompt), prime, normalized)

    def test_patch_must_remain_grammatical_at_article_boundary(self) -> None:
        prompt = "A music track with a steady beat plays."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "rhythm",
            "domain": ["music"], "question": "Is the beat steady?",
            "basis": "steady beat", "parent_ids": ["ROOT"],
        }
        patch = {
            "rubric_ids": ["R1"], "original_span": "steady beat",
            "replacement_span": "irregular rhythm", "edit_kind": "attribute",
            "original_event_family": "music", "replacement_event_family": "music",
            "semantic_distance": "high", "contradiction_explanation": "R1 is contradicted.",
            "audible_test": "Listen for an irregular rhythm.",
        }
        plan = self._plan(rubric, prompt)
        plan["language"] = "en"
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        with self.assertRaisesRegex(ValueError, "article_boundary_mismatch"):
            pipeline.validate_patch_result(plan, prime, normalized)

    def test_placement_flip_may_preserve_identical_music_identity(self) -> None:
        prompt = "Background orchestral and electronic music continues."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "placement",
            "domain": ["music"], "question": "Is the music in the background?",
            "basis": "Background orchestral and electronic music", "parent_ids": ["ROOT"],
        }
        patch = {
            "rubric_ids": ["R1"],
            "original_span": "Background",
            "replacement_span": "Foreground",
            "edit_kind": "attribute", "original_event_family": "music",
            "replacement_event_family": "music", "semantic_distance": "high",
            "contradiction_explanation": "R1 is contradicted by foreground placement.",
            "audible_test": "Listen for foreground music.",
        }
        plan = self._plan(rubric, prompt)
        plan["language"] = "en"
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        pipeline.validate_patch_result(plan, prime, normalized)

    def test_sound_effect_umbrella_accepts_impact_text(self) -> None:
        self.assertTrue(pipeline.declared_family_matches_text("sound_effect", {"impact"}))
        self.assertFalse(pipeline.declared_family_matches_text("sound_effect", {"speech"}))

    def test_metal_door_is_not_treated_as_music_genre_change(self) -> None:
        prompt = "A high-frequency electronic tone similar to an explosion is heard."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["sound_effect"], "question": "Is an electronic effect audible?",
            "basis": "A high-frequency electronic tone similar to an explosion",
            "parent_ids": [],
        }
        patch = {
            "rubric_ids": ["R1"],
            "original_span": "A high-frequency electronic tone similar to an explosion",
            "replacement_span": "A mechanical clatter from a heavy metal door",
            "edit_kind": "event", "original_event_family": "sound_effect",
            "replacement_event_family": "mechanical", "semantic_distance": "high",
            "contradiction_explanation": "R1 is contradicted by a mechanical clatter.",
            "audible_test": "Listen for a mechanical clatter from a metal door.",
        }
        plan = self._plan(rubric, prompt)
        plan.update({"language": "en", "intended_perturbation_strength": "strong"})
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        pipeline.validate_patch_result(plan, prime, normalized)

    def test_speech_event_prompt_requires_non_speech_replacement(self) -> None:
        prompt = "A male narrator explains a transformer experiment."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["speech"], "question": "Does a male narrator speak?",
            "basis": "A male narrator", "parent_ids": [],
        }
        plan = self._plan(rubric, prompt)
        plan.update({
            "language": "en", "disturbance_category": "event",
            "affected_rubrics": [rubric],
        })
        request = pipeline.patch_request(plan, None)
        self.assertIn("replacement must be non-speech", request)
        self.assertIn("not a different speaker gender", request)

    def test_all_three_prompt_names_immutable_relation_endpoint(self) -> None:
        prompt = "A bell rings. Subsequently, a dog barks."
        event = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["sound_effect"], "basis": "A bell rings",
            "target_ids": ["EV1"], "question": "Is a bell audible?", "parent_ids": [],
        }
        dog = {
            "rubric_id": "R2", "scope": "local", "subtype": "exists",
            "domain": ["animal"], "basis": "a dog barks", "target_ids": ["EV2"],
            "question": "Does a dog bark?", "parent_ids": [],
        }
        relation = {
            "rubric_id": "R3", "scope": "relational", "subtype": "before",
            "domain": ["sound_effect", "animal"], "basis": prompt,
            "target_ids": ["EV1", "EV2"], "question": "Is the bell before the dog?",
            "parent_ids": ["R1", "R2"],
        }
        plan = {
            "original_prompt": prompt, "direct_target_rubrics": [event, relation],
            "all_original_rubrics": [event, dog, relation],
        }
        instructions = pipeline.target_specific_instructions(plan)
        self.assertIn("UNCHANGED ENDPOINT ANCHORS", instructions)
        self.assertIn("a dog barks", instructions)

    def test_event_change_rejects_stale_temporal_coreference(self) -> None:
        prompt = "A narrator explains the experiment. After the speech ends, piano music starts."
        event = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["speech"], "question": "Does a narrator speak?",
            "basis": "A narrator explains the experiment", "target_ids": ["EV1"],
            "parent_ids": [],
        }
        relation = {
            "rubric_id": "R2", "scope": "relational", "subtype": "before",
            "domain": ["speech", "music"],
            "question": "Does the speech end before the piano starts?",
            "basis": prompt, "target_ids": ["EV1", "EV2"], "parent_ids": ["R1"],
        }
        patch = {
            "rubric_ids": ["R1"], "original_span": "A narrator explains the experiment",
            "replacement_span": "Mechanical grinding fills the room", "edit_kind": "event",
            "original_event_family": "speech", "replacement_event_family": "mechanical",
            "semantic_distance": "high", "contradiction_explanation": "R1 is contradicted.",
            "audible_test": "Listen for mechanical grinding.",
        }
        plan = self._plan(event, prompt)
        plan.update({"language": "en", "affected_rubrics": [event, relation]})
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        with self.assertRaisesRegex(ValueError, "stale_event_coreference"):
            pipeline.validate_patch_result(plan, prime, normalized)

    def test_event_coreference_update_must_preserve_untargeted_relation(self) -> None:
        prompt = "A narrator explains the experiment. After the speech ends, piano music starts."
        event = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["speech"], "question": "Does a narrator speak?",
            "basis": "A narrator explains the experiment", "target_ids": ["EV1"],
            "parent_ids": [],
        }
        relation = {
            "rubric_id": "R2", "scope": "relational", "subtype": "after",
            "domain": ["speech", "music"], "question": "Does music start after speech?",
            "basis": prompt, "target_ids": ["EV1", "EV2"], "parent_ids": ["R1"],
        }
        base_patch = {
            "rubric_ids": ["R1"],
            "original_span": "A narrator explains the experiment. After the speech ends,",
            "replacement_span": "Mechanical grinding fills the room. After the grinding ends,",
            "edit_kind": "event", "original_event_family": "speech",
            "replacement_event_family": "mechanical", "semantic_distance": "high",
            "contradiction_explanation": "R1 is contradicted by mechanical grinding.",
            "audible_test": "Listen for mechanical grinding.",
        }
        plan = self._plan(event, prompt)
        plan.update({
            "language": "en", "affected_rubrics": [event, relation],
            "intended_perturbation_strength": "strong",
        })
        prime, normalized = pipeline.apply_exact_patches(prompt, [base_patch], {"R1"})
        pipeline.validate_patch_result(plan, prime, normalized)

        changed = dict(base_patch)
        changed["replacement_span"] = "Mechanical grinding fills the room. Before the grinding ends,"
        prime, normalized = pipeline.apply_exact_patches(prompt, [changed], {"R1"})
        with self.assertRaisesRegex(ValueError, "incidental_relation_change"):
            pipeline.validate_patch_result(plan, prime, normalized)

    def test_chinese_event_change_rejects_stale_temporal_coreference(self) -> None:
        prompt = "一名男子说话。说话结束后，音乐开始。"
        event = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["speech"], "question": "是否有人说话？",
            "basis": "一名男子说话", "target_ids": ["EV1"], "parent_ids": [],
        }
        relation = {
            "rubric_id": "R2", "scope": "relational", "subtype": "before",
            "domain": ["speech", "music"], "question": "说话是否先于音乐？",
            "basis": prompt, "target_ids": ["EV1", "EV2"], "parent_ids": ["R1"],
        }
        patch = {
            "rubric_ids": ["R1"], "original_span": "一名男子说话",
            "replacement_span": "机器持续轰鸣", "edit_kind": "event",
            "original_event_family": "speech", "replacement_event_family": "mechanical",
            "semantic_distance": "high", "contradiction_explanation": "R1 被机械声反驳。",
            "audible_test": "聆听持续的机器轰鸣。",
        }
        plan = self._plan(event, prompt)
        plan.update({"language": "zh", "affected_rubrics": [event, relation]})
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        with self.assertRaisesRegex(ValueError, "stale_event_coreference"):
            pipeline.validate_patch_result(plan, prime, normalized)

    def test_all_three_relation_uses_replacement_event_anchor(self) -> None:
        prompt = "A loud impact occurs. Then a dog barks."
        event = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["sound_effect"], "question": "Is an impact audible?",
            "basis": "A loud impact occurs", "target_ids": ["EV1"],
            "parent_ids": [],
        }
        relation = {
            "rubric_id": "R2", "scope": "relational", "subtype": "before",
            "domain": ["sound_effect", "animal"],
            "question": "Does the impact occur before the bark?", "basis": prompt,
            "target_ids": ["EV1", "EV2"], "parent_ids": ["R1"],
        }
        patches = [
            {
                "rubric_ids": ["R1"], "original_span": "A loud impact occurs",
                "replacement_span": "A mechanical engine idles", "edit_kind": "event",
                "original_event_family": "sound_effect",
                "replacement_event_family": "mechanical", "semantic_distance": "high",
                "contradiction_explanation": "R1 is contradicted by a mechanical engine.",
                "audible_test": "Listen for a mechanical engine.",
            },
            {
                "rubric_ids": ["R2"], "original_span": "Then",
                "replacement_span": "Meanwhile", "edit_kind": "relation",
                "relation_endpoints": ["A mechanical engine idles", "a dog barks"],
                "semantic_distance": "high",
                "contradiction_explanation": "R2 is contradicted by overlap.",
                "audible_test": "Listen for the engine and bark to overlap.",
            },
        ]
        plan = {
            "original_prompt": prompt, "direct_target_rubrics": [event, relation],
            "direct_target_rubric_ids": ["R1", "R2"], "affected_rubrics": [event, relation],
            "retained_rubrics": [], "intended_perturbation_strength": "strong",
            "language": "en",
        }
        prime, normalized = pipeline.apply_exact_patches(prompt, patches, {"R1", "R2"})
        pipeline.validate_patch_result(plan, prime, normalized)

        patches[1]["replacement_span"] = "Prior to this"
        prime, normalized = pipeline.apply_exact_patches(prompt, patches, {"R1", "R2"})
        with self.assertRaisesRegex(ValueError, "relation_patch_subtype_not_flipped"):
            pipeline.validate_patch_result(plan, prime, normalized)

        mixed = [{
            "rubric_ids": ["R1", "R2"],
            "original_span": "A loud impact occurs. Then",
            "replacement_span": "A mechanical engine idles. After",
            "edit_kind": "mixed", "original_event_family": "sound_effect",
            "replacement_event_family": "mechanical",
            "relation_endpoints": ["A mechanical engine idles", "a dog barks"],
            "semantic_distance": "high",
            "contradiction_explanation": "R1 changes family and R2 changes relation.",
            "audible_test": "Listen for an engine and the temporal relation to the bark.",
        }]
        prime, normalized = pipeline.apply_exact_patches(prompt, mixed, {"R1", "R2"})
        with self.assertRaisesRegex(ValueError, "relation_patch_subtype_not_flipped"):
            pipeline.validate_patch_result(plan, prime, normalized)

        patches[1]["replacement_span"] = "After"
        prime, normalized = pipeline.apply_exact_patches(prompt, patches, {"R1", "R2"})
        with self.assertRaisesRegex(ValueError, "relation_patch_subtype_not_flipped"):
            pipeline.validate_patch_result(plan, prime, normalized)

    def test_event_edit_rejects_invented_person(self) -> None:
        prompt = "A male narrator explains a transformer experiment."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["speech"], "question": "Does a male narrator speak?",
            "basis": "A male narrator", "parent_ids": [],
        }
        patch = {
            "rubric_ids": ["R1"], "original_span": "A male narrator",
            "replacement_span": "Sarah Jenkins narrates", "edit_kind": "event",
            "original_event_family": "speech", "replacement_event_family": "music",
            "semantic_distance": "high", "contradiction_explanation": "R1 is contradicted.",
            "audible_test": "Listen for music.",
        }
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        with self.assertRaisesRegex(ValueError, "invented_proper_names"):
            pipeline.validate_patch_result(self._plan(rubric, prompt), prime, normalized)

    def test_event_edit_rejects_visual_sparks(self) -> None:
        prompt = "A narrator explains the experiment while piano music plays."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["speech"], "question": "Does a narrator speak?",
            "basis": "A narrator explains the experiment", "target_ids": ["EV1"],
            "parent_ids": [],
        }
        patch = {
            "rubric_ids": ["R1"], "original_span": "A narrator explains the experiment",
            "replacement_span": "Metal clashes while sparks fly", "edit_kind": "event",
            "original_event_family": "speech", "replacement_event_family": "impact",
            "semantic_distance": "high", "contradiction_explanation": "R1 is contradicted.",
            "audible_test": "Listen for metal clashes.",
        }
        plan = self._plan(rubric, prompt)
        plan.update({"language": "en", "affected_rubrics": [rubric]})
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        with self.assertRaisesRegex(ValueError, "visual_leakage"):
            pipeline.validate_patch_result(plan, prime, normalized)

    def test_speech_event_replacement_rejects_leftover_explanation(self) -> None:
        prompt = "A narrator explains how the transformer operates."
        rubric = {
            "rubric_id": "R1", "scope": "local", "subtype": "exists",
            "domain": ["speech"], "question": "Does a narrator speak?",
            "basis": prompt, "target_ids": ["EV1"], "parent_ids": [],
        }
        patch = {
            "rubric_ids": ["R1"], "original_span": prompt,
            "replacement_span": "Mechanical grinding occurs, covering the operation of the transformer.",
            "edit_kind": "event", "original_event_family": "speech",
            "replacement_event_family": "mechanical", "semantic_distance": "high",
            "contradiction_explanation": "R1 is contradicted by grinding.",
            "audible_test": "Listen for mechanical grinding.",
        }
        plan = self._plan(rubric, prompt)
        plan.update({
            "language": "en", "affected_rubrics": [rubric],
            "intended_perturbation_strength": "strong",
        })
        prime, normalized = pipeline.apply_exact_patches(prompt, [patch], {"R1"})
        with self.assertRaisesRegex(ValueError, "stale_speech_content"):
            pipeline.validate_patch_result(plan, prime, normalized)

    def test_relation_rebuild_flips_subtype_and_uses_word_boundaries(self) -> None:
        prompt = "Music overlaps speech."
        original = {
            "rubric_id": "R1", "axis": "temporal_structural_fidelity",
            "scope": "relational", "domain": ["music", "speech"],
            "subtype": "before", "question": "Does music play before speech?",
            "basis": "Music before speech", "target_ids": ["EV1", "EV2"],
            "parent_ids": [],
        }
        manifest = {
            "prompt_prime": prompt, "language": "en",
            "direct_target_rubric_ids": ["R1"], "affected_rubrics": [original],
        }
        good = {"rubrics": [{
            "rubric_id": "R1", "subtype": "overlap",
            "question": "Does music play at the same time as speech?",
            "basis_start": 0, "basis_end": len(prompt) - 1,
        }]}
        pipeline.validate_rebuild_result(manifest, good)
        bad = {"rubrics": [{**good["rubrics"][0], "basis_start": 1}]}
        with self.assertRaisesRegex(ValueError, "basis_cuts_word"):
            pipeline.validate_rebuild_result(manifest, bad)
        normalized = {"rubrics": [pipeline.normalize_rebuild_offsets(
            prompt, bad["rubrics"][0],
        )]}
        pipeline.validate_rebuild_result(manifest, normalized)


if __name__ == "__main__":
    unittest.main()
