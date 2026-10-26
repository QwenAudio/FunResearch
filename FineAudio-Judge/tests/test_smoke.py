"""CPU-only smoke tests for public deterministic components."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock
from judge.judge import parse_yes_no
from judge.protocols import (
    build_batch_rubric_prompt,
    build_holistic_messages,
    build_single_rubric_messages,
    order_and_chunk_rubrics,
    parse_batch_labels,
    parse_holistic_score,
)
from rubric_decomposition.models import Attribute, AudioSceneGraph, Event, Relation
from rubric_decomposition.tuples import build_tuples
from rubric_decomposition.validation import canonicalize_scene_graph, validate_scene_graph
from judge.scoring import aggregate_rubric_scores, normalized_yes_probability


class SmokeTest(unittest.TestCase):
    def test_llm_client_records_every_physical_attempt(self):
        from rubric_decomposition.llm_client import VllmJsonClient
        from rubric_decomposition.models import GeneratedQuestion

        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "llm_call_audit.jsonl"
            client = VllmJsonClient(
                base_url="http://localhost:1/v1", model="demo", api_key="test",
                concurrency=1, timeout=1, temperature=0, seed=10,
                audit_path=audit_path,
            )
            client.client.chat.completions.create = AsyncMock(side_effect=[
                SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="not json")
                )]),
                SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"tuple_id":"T001","question":"Q?"}')
                )]),
            ])
            parsed, raw = asyncio.run(client.generate(
                system="system", prompt="user", schema=GeneratedQuestion, max_tokens=20,
                audit_context={"sample_id": "demo", "stage": "questions"},
            ))
            self.assertEqual(parsed.tuple_id, "T001")
            self.assertIn("T001", raw)
            rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["error", "ok"])
            self.assertEqual([row["seed"] for row in rows], [10, 11])
            self.assertEqual({row["user"] for row in rows}, {"user"})
            self.assertEqual(rows[1]["context"], {"sample_id": "demo", "stage": "questions"})
            self.assertEqual(rows[0]["raw_model_output"], "not json")

    def test_scene_graph_to_tuples(self):
        prompt = "A dog barks three times, then a bell rings."
        graph = AudioSceneGraph(events=[
            Event(id="EV1", type="sound", label="dog bark", basis="A dog barks three times"),
            Event(id="EV2", type="sound", label="bell", basis="a bell rings"),
        ], attributes=[
            Attribute(id="AT1", owner_id="EV1", family="quantity", type="count", value=3, basis="three times"),
        ], relations=[
            Relation(id="REL1", type="before", source_id="EV1", target_id="EV2",
                     basis="A dog barks three times, then a bell rings"),
        ])
        graph = canonicalize_scene_graph(prompt, graph)
        self.assertEqual(validate_scene_graph(prompt, graph), [])
        self.assertEqual([row.predicate for row in build_tuples(graph)], ["exists", "exists", "count", "before"])

    def test_pipeline_name(self):
        from rubric_decomposition.pipeline import PIPELINE_NAME
        self.assertEqual(PIPELINE_NAME, "rubric_decomposition")

    def test_multilingual_quote_and_cycle_repairs(self):
        from rubric_decomposition.validation import (
            _contains_value, _natural_language_mismatch, _verbatim_foreign_content,
        )
        self.assertTrue(_contains_value("说话的音调是否适中？", "音调适中"))
        self.assertTrue(_contains_value("说话的音量是不是较低？", "音量较低"))
        self.assertFalse(_contains_value("说话的音调是否较高？", "音调适中"))
        prompt = 'A woman says "你好世界" and then a bell rings.'
        question = 'Does the woman contain the spoken content "你好世界"?'
        self.assertTrue(_natural_language_mismatch(question, "en"))
        self.assertTrue(_verbatim_foreign_content(question, prompt, "en"))

        invented_prompt = "A woman speaks and then a bell rings."
        invented_question = 'Does the woman say "你好世界"?'
        self.assertTrue(_natural_language_mismatch(invented_question, "en"))
        self.assertFalse(_verbatim_foreign_content(invented_question, invented_prompt, "en"))

        chinese_prompt = '一名女子说“hello world”，随后铃声响起。'
        self.assertTrue(_natural_language_mismatch("hello world", "zh"))
        self.assertTrue(_verbatim_foreign_content("hello world", chinese_prompt, "zh"))

        label_prompt = "男主持人的声音在音乐中慢慢淡出。"
        label_graph = AudioSceneGraph(events=[
            Event(id="EV1", type="speech", label="男主持人 fade out", basis=label_prompt),
        ])
        repaired_label = canonicalize_scene_graph(label_prompt, label_graph)
        self.assertEqual(repaired_label.events[0].label, "男主持人")

        cycle_prompt = "先有说话，然后有钟声，之后又有说话。"
        graph = AudioSceneGraph(events=[
            Event(id="EV1", type="speech", label="说话", basis="说话"),
            Event(id="EV2", type="sound", label="钟声", basis="钟声"),
        ], relations=[
            Relation(id="REL1", type="before", source_id="EV1", target_id="EV2", basis="然后有钟声"),
            Relation(id="REL2", type="before", source_id="EV2", target_id="EV1", basis="之后又有说话"),
        ])
        repaired = canonicalize_scene_graph(cycle_prompt, graph)
        self.assertEqual([(r.source_id, r.target_id) for r in repaired.relations], [("EV1", "EV2")])

    def test_judge_parsing_and_scoring(self):
        self.assertEqual(parse_yes_no("Yes."), ("yes", "punctuation_normalized"))
        self.assertAlmostEqual(normalized_yes_probability(0.0, 0.0), 0.5)
        report = aggregate_rubric_scores([{"answer": "yes", "axis": "content"},
                                          {"answer": "no", "axis": "content"}])
        self.assertEqual(report["score"], 0.5)

    def test_three_paper_inference_protocols(self):
        rubrics = [
            {"rubric_id": "R001", "subtype": "exists", "question": "Can a bell be heard?"},
            {"rubric_id": "R002", "subtype": "before", "question": "Is speech before the bell?"},
        ]
        single = build_single_rubric_messages("Can a bell be heard?", "exists")
        self.assertEqual(single[1]["content"], "<audio>\n\n判断题：\n【exists】Can a bell be heard?")

        batch_prompt = build_batch_rubric_prompt(rubrics)
        self.assertIn("## 判断题（共 2 道）", batch_prompt)
        self.assertIn('"criterion_type":"exists"', batch_prompt)
        labels = parse_batch_labels(
            '[{"id":"R001","answer":"yes"},{"id":"R002","answer":"no"}]',
            rubrics,
        )
        self.assertEqual(labels, [
            {"rubric_id": "R001", "answer": "yes"},
            {"rubric_id": "R002", "answer": "no"},
        ])
        first = order_and_chunk_rubrics("sample-1", rubrics, chunk_size=1)
        second = order_and_chunk_rubrics("sample-1", rubrics, chunk_size=1)
        self.assertEqual(first, second)

        holistic = build_holistic_messages("A bell rings.")
        self.assertIn("Reference prompt:\nA bell rings.", holistic[1]["content"])
        self.assertEqual(parse_holistic_score('{"score": 87}'), (87, "strict_json"))
        self.assertEqual(parse_holistic_score("101"), (None, "unparsed"))

    def test_batch_parser_rejects_incomplete_coverage(self):
        rubrics = [
            {"rubric_id": "R001", "question": "One?"},
            {"rubric_id": "R002", "question": "Two?"},
        ]
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            parse_batch_labels('[{"id":"R001","answer":"yes"}]', rubrics)

if __name__ == "__main__":
    unittest.main()
