"""Small synthetic tests: no data downloads or model requests."""

import copy
import hashlib
from pathlib import Path
import unittest

from agentic_ger.agent import SchemaViolation, load_schema_json, validate_schema
from agentic_ger.config import load_prompt_pack
from tools.prepare_data import aligned_segments

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_frozen_vendor(self) -> None:
        expected = {"gsb_chinese_normalizer.py": "926ea04cc2417b96c14345efb4a7a342794e162cad0e43901b0c00bf3b3ff88d",
                    "gsb_english_normalizer.py": "08dfb3f04120e19615b947d6dee822023921a52b8510c477a0ea9cd5cff19056"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / "vendor" / name).read_bytes()).hexdigest(), digest)

    def test_packs(self) -> None:
        for name in ("zh", "en"):
            pack = load_prompt_pack(ROOT / "prompts" / name)
            self.assertTrue(pack.sha256)
            self.assertEqual(pack.scan_schema(4)["properties"]["suspects"]["maxItems"], 4)

    def test_schema(self) -> None:
        schema = {"type": "object", "properties": {"text": {"type": "string", "maxLength": 4}}, "required": ["text"], "additionalProperties": False}
        value, _ = load_schema_json('{"text":"ok"}', schema)
        validate_schema(value, schema)
        for value in ({}, {"text": 1}, {"text": "too long"}, {"text": "ok", "extra": 0}):
            with self.assertRaises(SchemaViolation):
                validate_schema(value, schema)

    def test_alignment_and_reference_separation(self) -> None:
        ref = {"aid": "AGR-CH#synthetic", "segments": [
            {"sid": "AGR-CH#synthetic#0.000#1.000", "begin_time": "0.000", "end_time": "1.000", "text": "REFERENCE_ONLY", "entities": ["PRIVATE_LABEL"]},
            {"sid": "AGR-CH#synthetic#1.000#2.000", "begin_time": "1.000", "end_time": "2.000", "text": "reference two", "entities": []}]}
        hyp = copy.deepcopy(ref)
        for item in hyp["segments"]:
            item["text"] = "baseline"
        hyp["segments"].reverse()
        refs, hyps = aligned_segments(ref, hyp)
        self.assertEqual(hyps[0]["sid"], ref["segments"][0]["sid"])
        self.assertNotIn("REFERENCE_ONLY", str(hyps))
        self.assertNotIn("PRIVATE_LABEL", str(hyps))
        self.assertNotIn("entities", hyps[0])
        self.assertEqual(refs[0]["entities"], ["PRIVATE_LABEL"])
        hyp["segments"][0]["end_time"] = "3.000"
        with self.assertRaises(ValueError):
            aligned_segments(ref, hyp)


if __name__ == "__main__":
    unittest.main()
