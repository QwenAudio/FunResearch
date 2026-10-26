"""Tests for the released benchmark adapters and paper metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from judge.benchmark import (
    load_preference_candidates,
    load_rubric_qa,
    preference_report,
    rubric_qa_report,
)
from judge.protocols import order_and_chunk_rubrics


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BenchmarkTest(unittest.TestCase):
    def test_paper_ordering_seed(self):
        rubrics = [{"rubric_id": f"R{i:03d}", "question": str(i)} for i in range(1, 8)]
        ordered = order_and_chunk_rubrics("sample", rubrics, chunk_size=80)
        self.assertEqual(
            [row["rubric_id"] for row in ordered[0]],
            ["R001", "R002", "R005", "R004", "R003", "R007", "R006"],
        )

    def test_public_loaders_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rubric_qa/audio/a1").mkdir(parents=True)
            (root / "rubric_qa/audio/a1/a.wav").touch()
            write_jsonl(root / "rubric_qa/rubric_qa.jsonl", [{
                "sample_id": "a1:x", "subset": "a1", "rubric_set_id": "a1:x",
                "language": "en", "audio": "rubric_qa/audio/a1/a.wav",
                "rubric_id": "R001", "question": "Bell?", "subtype": "exists",
                "axis": "content", "scope": "local", "domain": ["sound"],
                "parent_ids": [], "label": "yes", "vote_pattern": "5-0",
            }])
            qa = load_rubric_qa(root)
            self.assertEqual(qa[0]["questions"][0]["reference_label"], "yes")
            qa_metrics = rubric_qa_report([{
                "sample_id": "a1:x", "rubric_id": "R001", "subset": "a1",
                "reference_label": "yes", "pred_answer": "yes", "status": "success",
            }])
            self.assertEqual(qa_metrics["overall"]["accuracy"], 1.0)

            audio_root = root / "human_preference/audio/g1"
            audio_root.mkdir(parents=True)
            candidates = []
            for candidate_id in ("a", "b", "c"):
                (audio_root / f"{candidate_id}.wav").touch()
                candidates.append({
                    "candidate_id": candidate_id,
                    "audio": f"human_preference/audio/g1/{candidate_id}.wav",
                })
            write_jsonl(root / "human_preference/candidate_groups.jsonl", [{
                "group_id": "g1", "language": "en", "category": "test",
                "prompt": "prompt", "candidates": candidates,
            }])
            rubric_groups = [{
                "group_id": f"g{i}", "language": "en", "category": "test",
                "prompt": "prompt", "rubric_count": 1,
                "rubrics": [{"rubric_id": "R001", "question": "Bell?"}],
            } for i in range(1, 101)]
            write_jsonl(root / "human_preference/prompt_rubrics.jsonl", rubric_groups)
            jobs = load_preference_candidates(root)
            self.assertEqual(len(jobs), 3)
            write_jsonl(root / "human_preference/human_preference_pairwise.jsonl", [{
                "pair_id": "p1", "group_id": "g1",
                "candidate_a": {"candidate_id": "a"},
                "candidate_b": {"candidate_id": "b"},
                "label": "a", "consensus_level": "high", "vote_pattern": "7:0",
            }])
            write_jsonl(root / "human_preference/human_preference_ranking.jsonl", [{
                "group_id": "g1", "label": ["a", "b", "c"],
                "consensus_level": "strict_majority", "modal_votes": 4,
            }])
            scores = [
                {"group_id": "g1", "candidate_id": "a", "yes_count": 3, "rubric_count": 3, "status": "success"},
                {"group_id": "g1", "candidate_id": "b", "yes_count": 2, "rubric_count": 3, "status": "success"},
                {"group_id": "g1", "candidate_id": "c", "yes_count": 1, "rubric_count": 3, "status": "success"},
            ]
            report = preference_report(root, scores)
            self.assertEqual(report["pairwise"]["overall"]["accuracy"], 1.0)
            self.assertEqual(report["ranking"]["overall"]["exact_ranking_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
