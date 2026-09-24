"""Offline integration regression: failed partial outputs must not be scored."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReportingTests(unittest.TestCase):
    def test_failed_recording_falls_back_in_both_languages(self) -> None:
        for language, script, reference, hypothesis, entity in (
            ("CH", "agentic_ger.evaluation.evaluate_zh", "人工智能", "人工智慧", "人工智能"),
            ("EN", "agentic_ger.evaluation.evaluate_en", "neural network", "neural work", "network"),
        ):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, data = root / "run", root / "data"
                run_id = f"AGR-{language}#synthetic"
                model = data / f"AGR-{language}" / "FunASR-Realtime"
                output = run / run_id / "workspace/output"
                output.mkdir(parents=True)
                (model / "ref").mkdir(parents=True)
                (model / "hyp").mkdir()
                ref = {"segments": [{"id": 0, "start": 0.0, "end": 1.0, "text_final": reference, "entities": [entity]}]}
                hyp = {"segments": [{"id": 0, "start": 0.0, "end": 1.0, "text_original": hypothesis}]}
                baseline_path = model / "hyp" / f"{run_id}.json"
                baseline_path.write_text(json.dumps(hyp))
                (model / "ref" / f"{run_id}.json").write_text(json.dumps(ref))
                (run / "manifest.jsonl").write_text(json.dumps({"run_id": run_id, "baseline": str(baseline_path)}) + "\n")
                (run / "batch_summary.json").write_text(json.dumps({"status": "failed", "tasks": 1, "complete": 0, "failed": 1, "batch_wall_seconds": 1}))
                (run / run_id / "runner_summary.json").write_text(json.dumps({"runner_status": "failed", "audio_seconds": 1, "total_seconds": 1}))
                # A perfect but partial final must be ignored for the failed recording.
                (output / "final_transcript.json").write_text(json.dumps(ref))
                (output / "trace.jsonl").write_text("")
                command = [sys.executable, "-m", script, "--run-root", str(run), "--data-root", str(data)]
                strict = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(strict.returncode, 0)
                fallback = subprocess.run(command + ["--allow-incomplete"], capture_output=True, text=True)
                self.assertEqual(fallback.returncode, 0, fallback.stderr)
                summary = json.loads((run / "evaluation/summary.json").read_text())
                self.assertEqual(summary["aggregate"]["baseline"], summary["aggregate"]["final"])
                self.assertEqual(summary["failures"]["count"], 1)
                self.assertEqual(summary["failures"]["scoring"], "baseline_unchanged")
                self.assertGreater(summary["aggregate"]["final"]["errors"], 0)


if __name__ == "__main__":
    unittest.main()
