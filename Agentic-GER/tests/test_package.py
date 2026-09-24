"""Package/CLI integration using synthetic data and a loopback mock, not a GPU."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import wave

from agentic_ger.evaluation.verify_zh import EXPECTED as ZH_DOMAINS
from agentic_ger.evaluation.verify_en import EXPECTED as EN_DOMAINS
from agentic_ger.utils import REPO_ROOT


class PackageTests(unittest.TestCase):
    def run_command(self, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_module_entry_points_outside_checkout(self) -> None:
        modules = (
            "agent", "runner", "experiment", "evaluation.evaluate_zh",
            "evaluation.evaluate_en", "evaluation.verify_zh", "evaluation.verify_en",
            "reporting.dashboard", "reporting.server",
        )
        with tempfile.TemporaryDirectory() as directory:
            for module in modules:
                with self.subTest(module=module):
                    self.run_command("-m", f"agentic_ger.{module}", "--help", cwd=Path(directory))

    def test_both_experiments_with_mock_services(self) -> None:
        for language, suffix, domains, count, before, after in (
            ("zh", "CH", ZH_DOMAINS, 524, "人工智慧", "人工智能"),
            ("en", "EN", EN_DOMAINS, 387, "neural work", "neural network"),
        ):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, run = root / "data", root / "run"
                run_id = f"AGR-{suffix}#synthetic0000"
                requests_seen: list[dict] = []

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args: object) -> None:
                        pass

                    def respond(self, body: dict) -> None:
                        payload = json.dumps(body).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)

                    def do_GET(self) -> None:
                        self.respond({"data": [
                            {"id": "qwen3.8-27b", "max_model_len": 98304},
                            {"id": "Qwen3-ASR-1.7B"},
                        ]})

                    def do_POST(self) -> None:
                        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        requests_seen.append(payload)
                        if self.path == "/tokenize":
                            self.respond({"count": 100, "max_model_len": 98304})
                            return
                        if payload["model"] == "Qwen3-ASR-1.7B":
                            content = after
                        else:
                            label = payload["response_format"]["json_schema"]["name"]
                            if label == "initial_summary":
                                value = {"summary": "Synthetic example", "domain": "test", "entities": []}
                            elif label == "scan_loop_1":
                                value = {"suspects": [{"segment_id": 0, "focus": before, "reason": "test"}]}
                            elif label.startswith("scan_"):
                                value = {"suspects": []}
                            else:
                                value = {"decision": "edit", "edited_segment": after, "reason": "test"}
                            content = json.dumps(value, ensure_ascii=False)
                        self.respond({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

                # Preserve the real discovery contract (12 domains, full cohort),
                # but run only one synthetic recording using the public entry point.
                hyp = {"recording_id": run_id, "segments": [
                    {"id": 0, "start": 0.0, "end": 1.0, "text_original": before},
                ]}
                for index, domain in enumerate(sorted(domains)):
                    model = data / domain / "FunASR-Realtime"
                    (model / "hyp").mkdir(parents=True)
                    domain_count = count - 11 if index == 0 else 1
                    for number in range(domain_count):
                        (model / "hyp" / f"{domain}#synthetic{number:04d}.json").write_text(json.dumps(hyp))
                model = data / f"AGR-{suffix}" / "FunASR-Realtime"
                (model / "ref").mkdir()
                (model / "audio").mkdir()
                reference = {"segments": [{"id": 0, "start": 0.0, "end": 1.0,
                                           "text_final": after, "entities": ["REFERENCE_ONLY_SENTINEL"]}]}
                (model / "ref" / f"{run_id}.json").write_text(json.dumps(reference))
                with wave.open(str(model / "audio" / f"{run_id}.wav"), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(b"\x00\x00" * 16000)

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
                config = str(REPO_ROOT / "configs" / f"{language}.json")
                command = (
                    "-m", "agentic_ger.experiment", config, "--data-root", str(data),
                    "--run-root", str(run), "--limit", "1", "--no-thinking",
                    "--llm-url", url, "--asr-url", url,
                )
                try:
                    dry = self.run_command(*command, "--dry-run", cwd=root)
                    self.assertIn("-m agentic_ger.runner", dry.stdout)
                    self.assertFalse(run.exists())
                    self.run_command(str(REPO_ROOT / "tools/check_services.py"), config,
                                     "--llm-url", url, "--asr-url", url, cwd=root)
                    self.run_command(*command, cwd=root)
                    batch = json.loads((run / "batch_summary.json").read_text())
                    self.assertEqual((batch["complete"], batch["failed"]), (1, 0))
                    output = run / run_id / "workspace/output"
                    result = json.loads((output / "result.json").read_text())
                    self.assertEqual(result["counters"]["accepted"], 1)
                    self.assertEqual(result["counters"]["asr_requests"], 1)
                    self.assertEqual(result["stop_reason"], "no_new_suspects")
                    self.assertNotIn("REFERENCE_ONLY_SENTINEL", json.dumps(requests_seen))
                    source = run / "repro/source"
                    for name in ("src/agentic_ger/runner.py", "src/agentic_ger/evaluation/evaluate_zh.py",
                                 "vendor/gsb_chinese_normalizer.py", "pyproject.toml", "prompts/zh/pack.json"):
                        self.assertTrue((source / name).is_file(), name)
                    self.assertTrue((run / "cases/index.html").is_file())
                    self.assertTrue((run / "evaluation/report.html").is_file())
                    calls_before = len(requests_seen)
                    self.run_command(*command, "--report-only", cwd=root)
                    self.assertEqual(len(requests_seen), calls_before)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
