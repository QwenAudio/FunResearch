"""CPU-only checks for the frozen paper decomposition pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rubric_decomposition.decompose import decompose_prompt
from rubric_decomposition.llm_client import (
    SEED_ENV,
    TEMPERATURE_ENV,
    VllmJsonClient,
)
from rubric_decomposition.models import (
    Attribute,
    AudioSceneGraph,
    Event,
    Relation,
)
from rubric_decomposition.pipeline import (
    PIPELINE_NAME,
    _validate_manifest,
    _validate_runtime_args,
    _write_run_config,
)
from rubric_decomposition.tuples import build_tuples
from rubric_decomposition.validation import (
    canonicalize_scene_graph,
    validate_scene_graph,
)


class DecompositionTest(unittest.TestCase):
    def test_pipeline_name(self) -> None:
        self.assertEqual(
            PIPELINE_NAME,
            "rubric_decomposition",
        )

    def test_scene_graph_to_atomic_tuples(self) -> None:
        prompt = "A dog barks three times, then a bell rings."
        graph = AudioSceneGraph(
            events=[
                Event(
                    id="EV1",
                    type="sound",
                    label="dog bark",
                    basis="A dog barks three times",
                ),
                Event(
                    id="EV2",
                    type="sound",
                    label="bell",
                    basis="a bell rings",
                ),
            ],
            attributes=[
                Attribute(
                    id="AT1",
                    owner_id="EV1",
                    family="quantity",
                    type="count",
                    value=3,
                    basis="three times",
                )
            ],
            relations=[
                Relation(
                    id="REL1",
                    type="before",
                    source_id="EV1",
                    target_id="EV2",
                    basis="A dog barks three times, then a bell rings",
                )
            ],
        )
        graph = canonicalize_scene_graph(prompt, graph)
        self.assertEqual(validate_scene_graph(prompt, graph), [])
        self.assertEqual(
            [row.predicate for row in build_tuples(graph)],
            ["exists", "exists", "count", "before"],
        )

    def test_empty_prompt_is_rejected_before_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "prompt must be a non-empty string"):
                asyncio.run(decompose_prompt(
                    "   ",
                    base_url="http://127.0.0.1:8001/v1",
                    model="model",
                    tokenizer="tokenizer",
                    output_dir=directory,
                ))

    def test_manifest_requires_unique_non_empty_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            _validate_manifest([
                {"sample_id": "same", "route": "test", "prompt": "one"},
                {"sample_id": "same", "route": "test", "prompt": "two"},
            ])
        with self.assertRaisesRegex(ValueError, "'prompt'"):
            _validate_manifest([{"sample_id": "one", "route": "test", "prompt": ""}])

    def test_runtime_budget_validation(self) -> None:
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8001/v1", model="model", tokenizer="tokenizer",
            concurrency=1, timeout=600.0, scene_graph_tokens=8192,
            context_window=4096, question_tokens=4096,
            question_output_tokens_per_question=160, question_safety_margin=2048,
            question_max_tuples_per_chunk=64,
        )
        with self.assertRaisesRegex(ValueError, "smaller than context_window"):
            _validate_runtime_args(args)

    def test_run_config_records_resolved_values_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"sample_id":"one","route":"test","prompt":"sound"}\n')
            args = argparse.Namespace(
                manifest=manifest, base_url="http://127.0.0.1:8001/v1",
                model="model", tokenizer="tokenizer", concurrency=1, timeout=600.0,
                scene_graph_tokens=8192, context_window=32768, question_tokens=4096,
                question_output_tokens_per_question=160, question_safety_margin=2048,
                question_max_tuples_per_chunk=64, stop_after_relation_audit=False,
            )
            client = type("Client", (), {"temperature": 0.0, "seed": 42})()
            destination = root / "run_config.json"
            _write_run_config(destination, args, client)
            config = json.loads(destination.read_text())
            self.assertEqual(config["temperature"], 0.0)
            self.assertEqual(config["seed"], 42)
            self.assertNotIn("api_key", config)

    def test_sampling_environment_and_explicit_precedence(self) -> None:
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test", TEMPERATURE_ENV: "0.25", SEED_ENV: "7"},
            clear=True,
        ):
            environment_client = VllmJsonClient(
                base_url="http://127.0.0.1:8001/v1", model="model"
            )
            explicit_client = VllmJsonClient(
                base_url="http://127.0.0.1:8001/v1", model="model",
                temperature=0.5, seed=11,
            )
            self.assertEqual(environment_client.temperature, 0.25)
            self.assertEqual(environment_client.seed, 7)
            self.assertEqual(explicit_client.temperature, 0.5)
            self.assertEqual(explicit_client.seed, 11)
            asyncio.run(environment_client.client.close())
            asyncio.run(explicit_client.client.close())


if __name__ == "__main__":
    unittest.main()
