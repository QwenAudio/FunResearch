"""Tests for the configurable Route 2 demo/custom recipe contract."""

from __future__ import annotations

import json
import importlib.util
from copy import deepcopy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_pipeline.route2_deterministic_composition.positive.pipeline import (
    load_recipe,
    validate_recipe,
)


DEMO_RECIPE = Path(__file__).parents[1] / "data_pipeline/route2_deterministic_composition/positive/config.json"
BACKEND = Path(__file__).parents[1] / "data_pipeline/route2_deterministic_composition/positive/stages/natural_caption_backend.py"
PROMPT_MATERIALIZER = Path(__file__).parents[1] / "data_pipeline/route2_deterministic_composition/negative/stages/materialize_prompt_negatives.py"
AUDIO_SCENE_BUILDER = Path(__file__).parents[1] / "data_pipeline/route2_deterministic_composition/negative/stages/build_audio_negative_scenes.py"
SCENE_GENERATOR = Path(__file__).parents[1] / "data_pipeline/route2_deterministic_composition/positive/stages/generate_scene_graphs.py"
NEGATIVE_PIPELINE = Path(__file__).parents[1] / "data_pipeline/route2_deterministic_composition/negative/pipeline.py"


def load_backend():
    spec = importlib.util.spec_from_file_location("route2_natural_caption_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_prompt_materializer():
    spec = importlib.util.spec_from_file_location("route2_prompt_materializer", PROMPT_MATERIALIZER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_audio_scene_builder():
    spec = importlib.util.spec_from_file_location("route2_audio_scene_builder", AUDIO_SCENE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_scene_generator():
    stage_dir = str(SCENE_GENERATOR.parent)
    sys.path.insert(0, stage_dir)
    try:
        spec = importlib.util.spec_from_file_location("route2_scene_generator", SCENE_GENERATOR)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(stage_dir)


class Route2RecipeTest(unittest.TestCase):
    def test_negative_prompt_stage_has_no_llm_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable, str(NEGATIVE_PIPELINE),
                    "--positive-dir", str(Path(directory) / "positive"),
                    "--output-dir", str(Path(directory) / "negative"),
                    "--start-at", "prompt_generation",
                    "--stop-after", "prompt_generation",
                    "--dry-run",
                ],
                cwd=Path(__file__).parents[1], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("build_relation_prompt_negatives", completed.stdout)
            self.assertNotIn("generate_prompt_negatives.py", completed.stdout)
            summary = json.loads(
                (Path(directory) / "negative/run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["prompt_llm_calls"], 0)

    def test_construction_labels_use_canonical_atomic_dependency_graph(self) -> None:
        from data_pipeline.route2_deterministic_composition.negative.stages import (
            build_construction_labels,
        )

        row = {
            "id": "demo", "target_caption_language": "en", "audio_path": "/demo.wav",
            "caption_natural_by_language": {
                "en": 'A woman speaks with a quiet delivery saying "hello", before Bell.'
            },
            "caption_rewrite_audit_by_language": {"en": {"relation_evidence": [{
                "relation_id": "R1", "quote": "A woman speaks with a quiet delivery saying \"hello\", before Bell",
            }]}},
            "events": [
                {
                    "event_id": "E1", "role": "speech", "start": 0.0, "end": 2.0,
                    "caption_label_en": "A woman speaks with a quiet delivery",
                    "caption_label_zh": "一名女性轻声说话", "speaker": "woman", "text": "hello",
                    "selected_caption_attrs": [{
                        "name": "situational_style", "value": "quiet",
                        "caption_text": "a quiet delivery",
                    }],
                },
                {
                    "event_id": "E2", "role": "sound_effect", "start": 2.2, "end": 4.0,
                    "caption_label_en": "Bell", "caption_label_zh": "铃声",
                },
            ],
        }
        built = build_construction_labels.build_row(row)
        self.assertEqual(
            built["construction_label_method"],
            "canonical_atomic_rubric_graph_from_deterministic_route2_truth",
        )
        self.assertEqual(len(built["atomic_tuples"]), len(built["dependencies"]))
        self.assertEqual(len(built["questions"]), len(built["atomic_tuples"]))
        speech_presence = next(
            item for item in built["questions"]
            if item["subtype"] == "exists"
            and item["construction_evidence"]["event_id"] == "E1"
        )
        self.assertEqual(speech_presence["target_ids"], ["EV1"])
        self.assertNotIn("quiet", speech_presence["question"].casefold())
        speech_attr = next(item for item in built["questions"] if item["subtype"] == "speech_attribute")
        self.assertEqual(speech_attr["parent_ids"], [speech_presence["rubric_id"]])
        relation = next(item for item in built["questions"] if item["subtype"] == "before")
        self.assertEqual(len(relation["parent_ids"]), 2)

        from data_pipeline.route2_deterministic_composition.negative.stages.audit_outputs import (
            canonical_rubric_errors,
        )
        self.assertEqual(canonical_rubric_errors(built), [])
        corrupted = deepcopy(built)
        corrupted["scene_graph"]["events"].reverse()
        errors = canonical_rubric_errors(corrupted)
        self.assertTrue(any(item.startswith("event_prompt_order_mismatch") for item in errors))
        self.assertIn("tuple_graph_mismatch", errors)

    def test_small_negative_speech_pool_can_collect_partial_candidates(self) -> None:
        from data_pipeline.route2_deterministic_composition.positive.stages import build_speech_pool

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "ZH_labels.csv"
            labels.write_text(
                "Key,Gender,Transcript,Speed,Pitch,Energy,Age,Emotion\n"
                "aishell_SSB00010001,女,测试语音,中,高,低,青年,平静\n",
                encoding="utf-8-sig",
            )
            audio_root = root / "audio"
            audio = audio_root / "SSB0001/SSB00010001.wav"
            audio.parent.mkdir(parents=True)
            audio.touch()
            build_speech_pool.ZH_LABELS = labels
            build_speech_pool.ZH_AUDIO = audio_root
            build_speech_pool.EXCLUDED_IDS = set()
            build_speech_pool.EXCLUDED_PATHS = set()

            with patch.object(build_speech_pool.sf, "info") as info:
                info.return_value.duration = 3.0
                rows = build_speech_pool.select_zh(7, 8, allow_partial=True)
                self.assertEqual([row["uid"] for row in rows], ["aishell_SSB00010001"])
                with self.assertRaisesRegex(RuntimeError, "Only selected 1/8"):
                    build_speech_pool.select_zh(7, 8)

    def test_speech_to_nonspeech_replacement_clears_stale_speech_metadata(self) -> None:
        builder = load_audio_scene_builder()
        event = {
            "event_id": "E1", "role": "speech", "source_dataset": "speech",
            "source_id": "old", "source_path": "/old.wav", "asset_slug": None,
            "label": "speech", "text": "stale transcript", "language": "en",
            "speaker": "spk", "speaker_description": "A woman speaks slowly",
            "style_prompt": "slow", "selected_caption_attrs": [{"name": "speed", "value": "slow"}],
            "primary_attrs": {}, "all_source_attrs": [], "annotation_format": "test",
        }
        chosen = {
            "source_dataset": "AudioSet-Strong", "video_id": "new",
            "source_path": "/new.wav", "event_name": "Engine", "event_start": 0.0,
            "event_end": 3.0, "role": "sound_effect", "label": "发动机声",
            "slug": "engine", "duration": 3.0,
        }
        pool = builder.UniquePool([chosen], seed=0, used_positive_paths=set())
        change = builder.replace_nonspeech(event, 2.0, [event], pool)

        self.assertEqual(event["role"], "sound_effect")
        self.assertEqual(event["mixing_role"], "speech")
        self.assertEqual(event["replacement_source_role"], "sound_effect")
        for key in ("text", "language", "speaker", "speaker_description", "selected_caption_attrs"):
            self.assertNotIn(key, event)
        self.assertEqual(change["before"]["text"], "stale transcript")
        self.assertIsNone(change["after"]["text"])

    def test_demo_recipe_is_balanced_and_valid(self) -> None:
        recipe = load_recipe(DEMO_RECIPE)
        validate_recipe(recipe)
        scenes = recipe["scenes"]
        self.assertEqual(scenes["count"], 10)
        self.assertEqual(
            scenes["zh_speech_count"] + scenes["zh_no_speech_count"], 5
        )
        self.assertEqual(
            scenes["en_speech_count"] + scenes["en_no_speech_count"], 5
        )
        self.assertEqual(recipe["asset_pool"], {
            "effect_count": 96, "ambience_count": 16, "music_count": 16,
        })
        self.assertEqual(recipe["speech_pool"], {
            "zh_count": 8, "ears_count": 6, "expresso_count": 2,
        })
        self.assertEqual(recipe["negative_pool"], {
            "effect_count": 32, "ambience_count": 8, "music_count": 6,
            "zh_speech_count": 8, "en_speech_count": 8,
        })
        self.assertEqual(scenes["max_effect_slug_reuse"], 1)
        self.assertEqual(scenes["music_count"], 6)
        self.assertEqual(scenes["max_effect_family_reuse"], 4)
        self.assertEqual(
            (scenes["foreground_no_background_min"], scenes["foreground_no_background_max"]),
            (5, 5),
        )
        self.assertEqual(
            (scenes["foreground_with_background_min"], scenes["foreground_with_background_max"]),
            (4, 4),
        )
        self.assertEqual((scenes["duration_min"], scenes["duration_max"]), (36, 48))

    def test_showcase_diversity_limit_relaxes_without_source_reuse(self) -> None:
        generator = load_scene_generator()
        rows = [
            {
                "role": "sound_effect", "slug": "bell", "video_id": f"source-{index}",
                "source_path": f"/source-{index}.flac",
            }
            for index in range(2)
        ]
        pool = generator.StrictUniqueAssetPool(rows, __import__("random").Random(0))

        first = pool.take("sound_effect", max_slug_reuse=1)
        second = pool.take("sound_effect", max_slug_reuse=1)

        self.assertNotEqual(first["video_id"], second["video_id"])
        self.assertEqual(pool.diversity_relaxations["sound_effect"], 1)

    def test_acoustic_families_use_tokens_and_specific_transport_priority(self) -> None:
        stage_dir = str(BACKEND.parent)
        sys.path.insert(0, stage_dir)
        try:
            from asset_pool import family
        finally:
            sys.path.remove(stage_dir)

        self.assertEqual(family("railroad_car_train_wagon"), "rail")
        self.assertEqual(family("train_wheels_squealing"), "rail")
        self.assertEqual(family("jet_engine"), "aircraft")
        self.assertEqual(family("aircraft_engine"), "aircraft")
        self.assertEqual(family("toilet_flush"), "water")

    def test_negative_plan_scales_without_changing_2k_policy(self) -> None:
        from data_pipeline.route2_deterministic_composition.negative.stages.plan_negatives import (
            LEVEL_BY_CATEGORY, balanced_category_counts,
        )

        self.assertEqual(
            balanced_category_counts(10),
            {"event": 3, "attribute": 2, "relation": 3, "all_three": 2},
        )
        self.assertEqual(
            balanced_category_counts(2000),
            {"event": 500, "attribute": 500, "relation": 500, "all_three": 500},
        )
        self.assertEqual(LEVEL_BY_CATEGORY, {
            "event": "low", "attribute": "high", "relation": "high", "all_three": "low",
        })

    def test_invalid_scene_sum_fails_closed(self) -> None:
        recipe = load_recipe(DEMO_RECIPE)
        recipe["scenes"]["en_no_speech_count"] = 2
        with self.assertRaisesRegex(ValueError, "language/speech quotas"):
            validate_recipe(recipe)

    def test_schema_version_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version=1"):
                load_recipe(path)

    def test_relation_evidence_quote_repair_keeps_caption_unchanged(self) -> None:
        backend = load_backend()
        row = {
            "background_mode": "none",
            "events": [
                {"event_id": "E1", "role": "sound_effect", "start": 0.0, "end": 2.0,
                 "caption_label_zh": "蒸汽喷出声", "caption_label_en": "Steam"},
                {"event_id": "E2", "role": "sound_effect", "start": 2.2, "end": 4.0,
                 "caption_label_zh": "闹钟声", "caption_label_en": "Alarm"},
            ],
        }
        value = {
            "caption": "蒸汽喷出声重叠响起，随后闹钟声响起。",
            "covered_event_ids": ["E1", "E2"],
            "relation_evidence": [{"relation_id": "R1", "quote": "蒸汽喷出声，随后闹钟声"}],
        }
        repaired = backend.repair_relation_evidence_quotes(row, "zh", value)
        self.assertEqual(repaired["caption"], value["caption"])
        self.assertEqual(repaired["relation_evidence"][0]["quote"], "蒸汽喷出声重叠响起，随后闹钟声")
        self.assertEqual(backend.validate(row, "zh", repaired), [])

    def test_missing_speech_text_repair_is_anchored_and_revalidated(self) -> None:
        backend = load_backend()
        row = {
            "background_mode": "none",
            "events": [
                {
                    "event_id": "E1", "role": "speech", "start": 0.0, "end": 2.0,
                    "caption_label_en": "A woman speaks slowly", "caption_label_zh": "一名女性缓慢说话",
                    "speaker_description": "A woman speaks slowly", "text": "Locked transcript.",
                },
                {
                    "event_id": "E2", "role": "sound_effect", "start": 2.2, "end": 4.0,
                    "caption_label_en": "Bell", "caption_label_zh": "铃声",
                },
            ],
        }
        value = {
            "caption": "A woman speaks slowly before Bell.",
            "covered_event_ids": ["E1", "E2"],
            "relation_evidence": [{"relation_id": "R1", "quote": "A woman speaks slowly before Bell"}],
        }
        self.assertEqual(backend.validate(row, "en", value), ["missing_speech:E1"])
        repaired, repairs = backend.repair_missing_speech_text(row, "en", value)
        self.assertEqual(repairs[0]["event_id"], "E1")
        self.assertIn('saying "Locked transcript."', repaired["caption"])
        self.assertEqual(backend.validate(row, "en", repaired), [])

    def test_adjacent_chinese_speaker_and_text_get_quotation_structure(self) -> None:
        backend = load_backend()
        row = {
            "background_mode": "none",
            "events": [{
                "event_id": "E1", "role": "speech", "start": 0.0, "end": 2.0,
                "caption_label_en": "A woman speaks", "caption_label_zh": "一名女性缓慢说话",
                "speaker_description": "一名女性缓慢说话", "text": "锁定台词",
            }],
        }
        value = {
            "caption": "一名女性缓慢说话锁定台词。",
            "covered_event_ids": ["E1"], "relation_evidence": [],
        }
        repaired, repairs = backend.repair_adjacent_speech_text(row, "zh", value)
        self.assertEqual(repaired["caption"], "一名女性缓慢说话，说出“锁定台词”。")
        self.assertEqual(repairs[0]["type"], "separate_locked_speaker_and_text")
        self.assertEqual(backend.validate(row, "zh", repaired), [])

    def test_unquoted_locked_transcripts_get_quotation_structure(self) -> None:
        backend = load_backend()
        zh_row = {
            "background_mode": "none",
            "events": [{
                "event_id": "E1", "role": "speech", "start": 0.0, "end": 2.0,
                "caption_label_en": "A woman speaks", "caption_label_zh": "一名女性说话",
                "speaker_description": "一名女性说话", "text": "锁定台词",
            }],
        }
        zh_value = {
            "caption": "一名女性说话，内容为锁定台词。",
            "covered_event_ids": ["E1"], "relation_evidence": [],
        }
        repaired, repairs = backend.repair_adjacent_speech_text(zh_row, "zh", zh_value)
        self.assertEqual(repaired["caption"], "一名女性说话，内容为“锁定台词”。")
        self.assertEqual(repairs[0]["type"], "quote_locked_speech_text")
        self.assertEqual(backend.validate(zh_row, "zh", repaired), [])

        en_row = {
            "background_mode": "none",
            "events": [{
                "event_id": "E1", "role": "speech", "start": 0.0, "end": 2.0,
                "caption_label_en": "A woman speaks", "caption_label_zh": "一名女性说话",
                "speaker_description": "A woman speaks", "text": "Locked transcript.",
            }],
        }
        en_value = {
            "caption": "A woman speaks saying Locked transcript.",
            "covered_event_ids": ["E1"], "relation_evidence": [],
        }
        repaired, repairs = backend.repair_adjacent_speech_text(en_row, "en", en_value)
        self.assertEqual(repaired["caption"], 'A woman speaks saying "Locked transcript."')
        self.assertEqual(repairs[0]["type"], "quote_locked_speech_text")
        self.assertEqual(backend.validate(en_row, "en", repaired), [])

        en_value = {
            "caption": "Bell rings, followed by a woman speaks saying Locked transcript.",
            "covered_event_ids": ["E1"], "relation_evidence": [],
        }
        repaired, repairs = backend.repair_adjacent_speech_text(en_row, "en", en_value)
        self.assertEqual(
            repaired["caption"],
            'Bell rings, and then a woman speaks saying "Locked transcript."',
        )
        self.assertEqual(
            [repair["type"] for repair in repairs],
            ["quote_locked_speech_text", "normalize_speech_clause_connector"],
        )

    def test_missing_non_relation_generation_gets_deterministic_seed(self) -> None:
        materializer = load_prompt_materializer()
        job = {
            "sample_id": "demo::neg::attribute",
            "parent_positive_id": "demo",
            "prompt": "A woman speaks slowly.",
            "disturbance_category": "attribute",
            "path2_perturbation_category": "attribute",
            "selected_rubric_ids": ["demo::speech_attr::speed"],
        }
        row = materializer.deterministic_missing_job_seed(job)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["prompt_prime"], job["prompt"])
        self.assertEqual(row["patches"], [])
        self.assertEqual(row["model"], "deterministic_missing_job_seed")

    def test_missing_relation_generation_must_use_reversal_stage(self) -> None:
        materializer = load_prompt_materializer()
        job = {
            "sample_id": "demo::neg::relation",
            "parent_positive_id": "demo",
            "prompt": "A happens before B.",
            "disturbance_category": "relation",
            "path2_perturbation_category": "relation",
            "selected_rubric_ids": ["demo::before"],
        }
        with self.assertRaisesRegex(ValueError, "requires deterministic reversal"):
            materializer.deterministic_missing_job_seed(job)

    def test_multiple_speech_attributes_can_use_cross_dimension_fallback(self) -> None:
        materializer = load_prompt_materializer()
        prompt = "A woman speaks with an American accent and a monotonous voice."
        selected = [
            {
                "rubric_id": "demo::american",
                "subtype": "speech_attribute",
                "construction_evidence": {
                    "event_id": "E1",
                    "attribute": {
                        "name": "intrinsic_style", "value": "american",
                        "caption_text": "an American accent",
                    },
                },
            },
            {
                "rubric_id": "demo::monotonous",
                "subtype": "speech_attribute",
                "construction_evidence": {
                    "event_id": "E1",
                    "attribute": {
                        "name": "intrinsic_style", "value": "monotonous",
                        "caption_text": "a monotonous voice",
                    },
                },
            },
        ]
        job = {
            "sample_id": "demo::neg::attribute",
            "prompt": prompt,
            "language": "en",
            "selected_rubrics": selected,
            "selected_rubric_ids": [row["rubric_id"] for row in selected],
            "replacement_candidates": [],
        }
        generated = {
            "prompt": prompt, "prompt_prime": prompt, "patches": [],
            "model": "deterministic_missing_job_seed",
        }
        pair = {
            "events": [{
                "event_id": "E1", "role": "speech",
                "caption_label_en": "A woman speaks with an American accent and a monotonous voice",
                "caption_label_zh": "一名女性说话",
            }],
        }
        speech_pool = [
            {
                "uid": "same-dimension", "language": "en", "source_dataset": "test",
                "selected_caption_attrs": [
                    {"name": "intrinsic_style", "value": "husky", "caption_text": "a husky voice"},
                ],
            },
            {
                "uid": "cross-dimension", "language": "en", "source_dataset": "test",
                "selected_caption_attrs": [
                    {"name": "situational_style", "value": "loud", "caption_text": "a loud delivery"},
                ],
            },
        ]
        repaired, audit = materializer.repair_direct_targets(job, generated, pair, speech_pool)
        self.assertTrue(audit["pass"])
        self.assertNotIn("American accent", repaired["prompt_prime"])
        self.assertNotIn("monotonous voice", repaired["prompt_prime"])
        self.assertIn("husky voice", repaired["prompt_prime"])
        self.assertIn("loud delivery", repaired["prompt_prime"])
        repairs = {row["rubric_id"]: row for row in audit["repairs"]}
        self.assertEqual(
            repairs["demo::monotonous"]["new_attribute"]["name"],
            "situational_style",
        )

    def test_speech_event_replacement_allows_attribute_count_fallback(self) -> None:
        materializer = load_prompt_materializer()
        old_text = "The original locked transcript."
        old_description = (
            "A woman speaks with an American accent, a measured pace, "
            "and a monotonous voice"
        )
        prompt = f'{old_description} saying "{old_text}"'
        selected = [{
            "rubric_id": "demo::exists::E1",
            "subtype": "exists",
            "construction_evidence": {"event_id": "E1"},
        }]
        job = {
            "sample_id": "demo::neg::event",
            "prompt": prompt,
            "language": "en",
            "selected_rubrics": selected,
            "selected_rubric_ids": ["demo::exists::E1"],
            "replacement_candidates": [],
        }
        generated = {"prompt": prompt, "prompt_prime": prompt, "patches": []}
        pair = {"events": [{
            "event_id": "E1", "role": "speech",
            "caption_label_en": old_description,
            "caption_label_zh": "一名女性说话",
            "speaker_description": old_description,
            "text": old_text,
            "selected_caption_attrs": [
                {"caption_text": "an American accent"},
                {"caption_text": "a measured pace"},
                {"caption_text": "a monotonous voice"},
            ],
        }]}
        speech_pool = [{
            "uid": "two-attributes", "language": "en", "source_dataset": "test",
            "speaker_description": "A man speaks loudly with a husky voice",
            "text": "A different training transcript.",
            "selected_caption_attrs": [
                {"caption_text": "a loud delivery"},
                {"caption_text": "a husky voice"},
            ],
        }]

        repaired, audit = materializer.repair_direct_targets(
            job, generated, pair, speech_pool
        )

        self.assertTrue(audit["pass"])
        self.assertNotIn(old_description, repaired["prompt_prime"])
        self.assertNotIn(old_text, repaired["prompt_prime"])
        self.assertIn("A man speaks loudly with a husky voice", repaired["prompt_prime"])
        self.assertIn("A different training transcript.", repaired["prompt_prime"])
        repair = audit["repairs"][0]
        self.assertEqual(repair["original_attribute_count"], 3)
        self.assertEqual(repair["replacement_attribute_count"], 2)


if __name__ == "__main__":
    unittest.main()
