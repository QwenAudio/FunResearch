#!/usr/bin/env python3
"""Verify English metrics against official GigaSpeechBench baseline counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_ger.evaluation.metrics_en import add_result, empty_result, evaluate_record, result_json


DEFAULT_DATA_ROOT = Path(
    "data/prepared/Vertical-Domain"
)

# Counts reproduced by the official GigaSpeechBench evaluator at commit
# ca782bff09a424233cd3aa1aff11c346cd0f2ed7.
EXPECTED = {
    "AGR-EN": (1718, 3092, 3010, 116349, 122451, 6432, 520, 6667),
    "AIT-EN": (1768, 3918, 4304, 101385, 109607, 5280, 2377, 10559),
    "ART-EN": (1121, 1897, 2681, 97487, 102065, 5620, 820, 9192),
    "BIO-EN": (753, 2242, 2788, 97667, 102697, 5280, 3074, 20608),
    "ECM-EN": (3297, 3343, 5603, 129300, 138246, 5595, 1928, 14455),
    "ENG-EN": (901, 1790, 3025, 112404, 117219, 6636, 1259, 13672),
    "ENT-EN": (1875, 5256, 4402, 116250, 125908, 8052, 774, 3780),
    "FIN-EN": (2672, 3650, 4001, 132656, 140307, 5831, 1298, 12634),
    "HUM-EN": (1492, 1853, 2932, 93782, 98567, 4914, 534, 10578),
    "LAW-EN": (1512, 4178, 5005, 101398, 110581, 7126, 2104, 14071),
    "MED-EN": (993, 1998, 2662, 104344, 109004, 5158, 2092, 16934),
    "MIL-EN": (1072, 2368, 3377, 123762, 129507, 5211, 2965, 19161),
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    failures = []
    total_errors = total_reference = 0
    total_biased_errors = total_biased_reference = 0
    domain_wer: list[float] = []
    domain_bwer: list[float] = []
    recordings = 0
    for domain, expected in EXPECTED.items():
        total = empty_result()
        model_root = data_root / domain / "FunASR-Realtime"
        for hypothesis_path in sorted((model_root / "hyp").glob(f"{domain}#*.json")):
            add_result(
                total,
                evaluate_record(
                    load(model_root / "ref" / hypothesis_path.name),
                    load(hypothesis_path),
                    hypothesis_is_final=False,
                ),
            )
            recordings += 1
        actual_json = result_json(total)
        actual = (
            actual_json["I"],
            actual_json["D"],
            actual_json["S"],
            actual_json["C"],
            actual_json["N"],
            actual_json["segments"],
            actual_json["hotword"]["errors"],
            actual_json["hotword"]["reference"],
        )
        if actual != expected:
            failures.append({"domain": domain, "expected": expected, "actual": actual})
        print(
            f"{domain}: WER {actual_json['rate']:.6f}%  "
            f"B-WER {actual_json['hotword']['b_wer']:.6f}%  "
            f"{'OK' if actual == expected else 'MISMATCH'}",
            flush=True,
        )
        total_errors += actual_json["errors"]
        total_reference += actual_json["N"]
        total_biased_errors += actual_json["hotword"]["errors"]
        total_biased_reference += actual_json["hotword"]["reference"]
        domain_wer.append(float(actual_json["rate"]))
        domain_bwer.append(float(actual_json["hotword"]["b_wer"]))

    if recordings != 387:
        failures.append({"recordings": recordings, "expected_recordings": 387})
    if failures:
        raise SystemExit(json.dumps(failures, ensure_ascii=False, indent=2))
    print(
        "PASS: 387/387; "
        f"macro WER={sum(domain_wer) / len(domain_wer):.12f}%; "
        f"macro B-WER={sum(domain_bwer) / len(domain_bwer):.12f}%; "
        f"micro WER={100 * total_errors / total_reference:.12f}%; "
        f"micro B-WER={100 * total_biased_errors / total_biased_reference:.12f}%"
    )


if __name__ == "__main__":
    main()
