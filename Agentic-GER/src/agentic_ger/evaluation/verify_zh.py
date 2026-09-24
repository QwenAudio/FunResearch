#!/usr/bin/env python3
"""Verify the local evaluator against the frozen 524-item FunASR baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_ger.evaluation.metrics_zh import add_result, empty_result, evaluate_record, result_json


DEFAULT_DATA_ROOT = Path(
    "data/prepared/Vertical-Domain"
)

# Counts produced earlier by the official GigaSpeechBench scripts. Exact integer
# comparison is stronger than comparing rounded percentages.
EXPECTED = {
    "AGR-CH": (444, 689, 2229, 103754, 106672, 3342, 445, 7280),
    "AIT-CH": (1420, 1969, 3464, 200173, 205606, 5262, 886, 5626),
    "ART-CH": (620, 610, 3297, 170523, 174430, 3682, 886, 9811),
    "BIO-CH": (702, 583, 2133, 162271, 164987, 4274, 532, 3781),
    "ECM-CH": (2269, 4102, 10840, 174198, 189140, 5830, 1428, 6335),
    "ENG-CH": (1264, 843, 2841, 183205, 186889, 4337, 839, 8499),
    "ENT-CH": (772, 1034, 3774, 158687, 163495, 4535, 972, 4809),
    "FIN-CH": (623, 645, 1678, 202897, 205220, 3814, 235, 9573),
    "HUM-CH": (261, 317, 2469, 178519, 181305, 3536, 500, 4733),
    "LAW-CH": (1011, 2222, 5630, 170555, 178407, 4411, 1414, 15365),
    "MED-CH": (534, 540, 1655, 172589, 174784, 3888, 596, 6816),
    "MIL-CH": (489, 481, 1754, 180749, 182984, 3673, 258, 10687),
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    failures = []
    total_errors = total_reference = total_biased_errors = total_biased_reference = 0
    domain_cer: list[float] = []
    domain_bwer: list[float] = []
    recordings = 0
    for domain, expected in EXPECTED.items():
        total = empty_result()
        model_root = data_root / domain / "FunASR-Realtime"
        for hypothesis_path in sorted((model_root / "hyp").glob(f"{domain}#*.json")):
            reference_path = model_root / "ref" / hypothesis_path.name
            add_result(
                total,
                evaluate_record(
                    load(reference_path),
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
            f"{domain}: CER {actual_json['rate']:.6f}%  "
            f"B-WER {actual_json['hotword']['b_wer']:.6f}%  "
            f"{'OK' if actual == expected else 'MISMATCH'}"
        )
        total_errors += actual_json["errors"]
        total_reference += actual_json["N"]
        total_biased_errors += actual_json["hotword"]["errors"]
        total_biased_reference += actual_json["hotword"]["reference"]
        domain_cer.append(float(actual_json["rate"]))
        domain_bwer.append(float(actual_json["hotword"]["b_wer"]))

    if recordings != 524:
        failures.append({"recordings": recordings, "expected_recordings": 524})
    if failures:
        raise SystemExit(json.dumps(failures, ensure_ascii=False, indent=2))
    print(
        "PASS: 524/524; "
        f"macro CER={sum(domain_cer) / len(domain_cer):.12f}%; "
        f"macro B-WER={sum(domain_bwer) / len(domain_bwer):.12f}%; "
        f"micro CER={100 * total_errors / total_reference:.12f}%; "
        f"micro B-WER={100 * total_biased_errors / total_biased_reference:.12f}%"
    )


if __name__ == "__main__":
    main()
