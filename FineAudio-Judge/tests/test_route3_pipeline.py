"""Regression tests for Route 3's shared construction contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
NEGATIVE_STAGES = (
    REPO_ROOT / "data_pipeline/route3_prompt_generation/negative/stages"
)
if str(NEGATIVE_STAGES) not in sys.path:
    sys.path.insert(0, str(NEGATIVE_STAGES))

import export_prompt_rubric_input as rubric_export  # noqa: E402
import audit_outputs as route3_audit  # noqa: E402
import generate_prompt_negatives as grouped  # noqa: E402
import plan_negatives as planner  # noqa: E402

from rubric_decomposition.models import (  # noqa: E402
    AudioSceneGraph, Event, GeneratedQuestion,
)
from rubric_decomposition.tuples import build_tuples  # noqa: E402
from rubric_decomposition.validation import (  # noqa: E402
    build_dependencies, canonicalize_questions, canonicalize_scene_graph,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class Route3PipelineTest(unittest.TestCase):
    def _event_rubric(self) -> dict:
        return {
            "rubric_id": "R1", "tuple_id": "T001",
            "axis": "content_fidelity", "scope": "local",
            "domain": ["sound_effect"], "subtype": "exists",
            "question": "Is a bell audible?", "basis": "A bell rings",
            "basis_start": 0, "basis_end": 12,
            "target_ids": ["EV1"], "parent_ids": [],
        }

    def test_split_input_strips_only_canonical_positive_constant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "one.wav"
            audio.write_bytes(b"audio")
            positive = root / "positive.jsonl"
            rubrics = root / "rubrics.jsonl"
            prompt = "A bell rings clearly."
            write_jsonl(positive, [{
                "sample_id": "one", "route": "route3_prompt_generation",
                "prompt": prompt, "audio_path": str(audio), "language": "en",
            }])
            question = {**self._event_rubric(), "expected_answer": "yes"}
            write_jsonl(rubrics, [{
                "sample_id": "one", "prompt": prompt, "questions": [question],
            }])
            joined = planner.adapt_public_inputs(positive, rubrics)
            clean = planner._sanitize_source(joined[0], location="test")
            self.assertIsNotNone(clean)
            self.assertNotIn("expected_answer", clean["questions"][0])

            joined[0]["questions"][0]["label"] = "yes"
            with self.assertRaisesRegex(ValueError, "forbidden label keys"):
                planner._sanitize_source(joined[0], location="test")

    def test_grouped_event_patch_satisfies_shared_route1_contract(self) -> None:
        rubric = self._event_rubric()
        plan = {
            "sample_id": "route3neg_one", "parent_positive_id": "one",
            "execution_rank": 0, "language": "en", "target_band": "low",
            "original_prompt": "A bell rings clearly.",
            "direct_target_rubric_ids": ["R1"],
            "direct_target_rubrics": [rubric], "affected_rubrics": [rubric],
            "retained_rubrics": [], "all_original_rubrics": [rubric],
            "intended_perturbation_strength": "strong",
        }
        group = grouped.make_groups(plan)[0]
        value = grouped.validate_group(group, {
            "replacement_span": "An engine revs",
            "original_event_family": "sound_effect",
            "replacement_event_family": "mechanical",
            "relation_endpoints": [],
            "contradiction_explanation": "R1 changes the bell into an engine.",
            "audible_test": "Listen for an engine revving.",
        })
        prime, patches = grouped.build_sample(
            plan, [group], {group["group_id"]: value},
        )
        self.assertEqual(prime, "An engine revs clearly.")
        self.assertEqual(patches[0]["replacement_event_family"], "mechanical")

    def test_relation_group_requires_endpoint_contract(self) -> None:
        group = {
            "original_span": "A bell rings before a dog barks",
            "expected_kinds": ["relation"],
        }
        with self.assertRaisesRegex(ValueError, "relation_endpoints"):
            grouped.validate_group(group, {
                "replacement_span": "A dog barks before a bell rings",
                "relation_endpoints": [],
                "contradiction_explanation": "R1 reverses the order.",
                "audible_test": "Listen for the dog before the bell.",
            })

    def test_prompt_rubric_export_uses_pprime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.jsonl"
            output = root / "rubric_input.jsonl"
            write_jsonl(pairs, [{
                "sample_id": "negative-one", "language": "en",
                "prompt_prime": "A dog barks before a bell rings.",
            }])
            rows = rubric_export.export(pairs, output)
            self.assertEqual(rows[0]["prompt"], "A dog barks before a bell rings.")
            self.assertEqual(rows[0]["route"], "route3_prompt_side_negative")

    def test_final_audit_attaches_only_canonical_pprime_rubrics(self) -> None:
        prompt = "A bell rings."
        graph = canonicalize_scene_graph(prompt, AudioSceneGraph(events=[
            Event(id="EV1", type="sound", label="bell rings", basis="A bell rings")
        ]))
        tuples = build_tuples(graph)
        dependencies = build_dependencies(tuples)
        questions = canonicalize_questions(
            tuples, dependencies,
            [GeneratedQuestion(tuple_id=tuples[0].tuple_id, question="Is a bell audible?")],
            prompt, "en",
        )
        rubric = {
            "sample_id": "negative-one", "prompt": prompt, "language": "en",
            "scene_graph": graph.model_dump(),
            "atomic_tuples": [item.model_dump() for item in tuples],
            "dependencies": [item.model_dump() for item in dependencies],
            "questions": [item.model_dump() for item in questions],
            "validation": {"valid": True},
        }
        pair = {"sample_id": "negative-one", "prompt_prime": prompt}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rubrics.jsonl"
            write_jsonl(path, [rubric])
            enriched, errors = route3_audit.attach_prompt_rubrics([pair], path)
        self.assertEqual(errors, [])
        self.assertEqual(
            enriched[0]["evaluation_rubrics_status"],
            "canonical_regenerated_from_P_prime",
        )


if __name__ == "__main__":
    unittest.main()
