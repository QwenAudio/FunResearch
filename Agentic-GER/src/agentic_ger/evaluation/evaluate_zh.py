#!/usr/bin/env python3
"""Evaluate one completed Chinese run and build its report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_ger.evaluation.metrics_zh import (
    add_result,
    empty_result,
    evaluate_record,
    normalize_text,
    result_json,
)
from agentic_ger.reporting.visualize import write_report


from agentic_ger.utils import REPO_ROOT


ROOT = REPO_ROOT
DEFAULT_RUN_ROOT = ROOT / "runs/zh"
DEFAULT_DATA_ROOT = Path(
    "data/gigaspeechbench/prepared/Vertical-Domain"
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_item in enumerate(right, 1):
        current = [row_index]
        for column, left_item in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def accepted_patches(trace: Path) -> list[dict[str, Any]]:
    result = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "patch.decision" and event.get("decision") == "accept":
            result.append(event["patch"])
    return result


def patch_outcome(
    run_id: str,
    patch: dict[str, Any],
    reference_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    segment_id = int(patch["segment_id"])
    reference_segment = reference_by_id[segment_id]
    reference_text = str(
        reference_segment.get("text_final")
        or reference_segment.get("text_working")
        or reference_segment.get("text_original")
        or reference_segment.get("text")
        or ""
    )

    def units(value: str) -> list[str]:
        return list(normalize_text(value.replace("&", " and ")).replace(" ", ""))

    before_distance = edit_distance(units(patch["before_segment"]), units(reference_text))
    after_distance = edit_distance(units(patch["after_segment"]), units(reference_text))
    delta = after_distance - before_distance
    return {
        "run_id": run_id,
        "domain": run_id.split("#", 1)[0],
        "segment_id": segment_id,
        "old_text": patch["old_text"],
        "new_text": patch["new_text"],
        "reference": reference_text,
        "before_distance": before_distance,
        "after_distance": after_distance,
        "delta": delta,
        "outcome": "improved" if delta < 0 else "worsened" if delta > 0 else "neutral",
        "reason": patch.get("reason"),
        "asr_text": patch.get("asr_text"),
    }


def aggregate(domains: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("baseline", "final"):
        errors = sum(int(row[side]["errors"]) for row in domains)
        reference = sum(int(row[side]["N"]) for row in domains)
        biased_errors = sum(int(row[side]["hotword"]["errors"]) for row in domains)
        biased_reference = sum(
            int(row[side]["hotword"]["reference"]) for row in domains
        )
        result[side] = {
            "macro_cer": (
                sum(float(row[side]["rate"]) for row in domains) / len(domains)
                if domains
                else 0.0
            ),
            "micro_cer": 100.0 * errors / reference if reference else 0.0,
            "errors": errors,
            "reference_characters": reference,
            "macro_bwer": (
                sum(float(row[side]["hotword"]["b_wer"]) for row in domains)
                / len(domains)
                if domains
                else 0.0
            ),
            "micro_bwer": (
                100.0 * biased_errors / biased_reference if biased_reference else 0.0
            ),
            "biased_errors": biased_errors,
            "biased_reference_tokens": biased_reference,
        }
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    before, after = summary["aggregate"]["baseline"], summary["aggregate"]["final"]
    patches = summary["patches"]
    lines = [
        f"# 中文 {summary['recordings']} 条评测结果",
        "",
        f"CER（领域 macro）：{before['macro_cer']:.4f}% → "
        f"{after['macro_cer']:.4f}% "
        f"({after['macro_cer'] - before['macro_cer']:+.4f} 个百分点)",
        "",
        f"B-WER（领域 macro）：{before['macro_bwer']:.4f}% → "
        f"{after['macro_bwer']:.4f}% "
        f"({after['macro_bwer'] - before['macro_bwer']:+.4f} 个百分点)",
        "",
        f"接受修改 {patches['accepted']} 个：改善 {patches['improved']}，"
        f"变差 {patches['worsened']}，不变 {patches['neutral']}。",
        "",
    ]
    failures = summary["failures"]
    if failures["count"]:
        lines.extend(
            [
                f"失败 {failures['count']} 条，按原始转写未修改计分；"
                "失败仍单独计数，不视为成功。",
                "",
            ]
        )
    lines.extend(["| 领域 | CER | B-WER |", "|---|---:|---:|"])
    for row in summary["domains"]:
        lines.append(
            f"| {row['domain']} | {row['baseline']['rate']:.4f} → "
            f"{row['final']['rate']:.4f} | "
            f"{row['baseline']['hotword']['b_wer']:.4f} → "
            f"{row['final']['hotword']['b_wer']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline-system", default="FunASR-Realtime")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score failed records as unchanged baselines and report them explicitly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    batch = load(run_root / "batch_summary.json")
    tasks = int(batch.get("tasks") or 0)
    complete = int(batch.get("complete") or 0)
    failed = int(batch.get("failed") or 0)
    strict_complete = (
        batch.get("status") == "complete"
        and tasks > 0
        and complete == tasks
        and failed == 0
    )
    allowed_incomplete = (
        args.allow_incomplete
        and tasks > 0
        and complete + failed == tasks
        and failed > 0
    )
    if not (strict_complete or allowed_incomplete):
        raise SystemExit("the requested batch is not complete; evaluation refused")

    rows = [
        json.loads(line)
        for line in (run_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(batch["tasks"]):
        raise SystemExit(
            f"batch reports {batch['tasks']} tasks, but manifest contains {len(rows)} rows"
        )
    domains = sorted({row["run_id"].split("#", 1)[0] for row in rows})

    domain_totals = {
        domain: {"baseline": empty_result(), "final": empty_result()}
        for domain in domains
    }
    patch_records: list[dict[str, Any]] = []
    runtime_rows = []
    failure_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        run_id = row["run_id"]
        domain = run_id.split("#", 1)[0]
        output = run_root / run_id / "workspace/output"
        reference = load(
            data_root / domain / args.baseline_system / "ref" / f"{run_id}.json"
        )
        baseline_path = output / "baseline_transcript.json"
        baseline = load(baseline_path if baseline_path.exists() else Path(row["baseline"]))
        final_path = output / "final_transcript.json"
        trace_path = output / "trace.jsonl"
        runner_path = run_root / run_id / "runner_summary.json"
        runner = load(runner_path) if runner_path.exists() else {}
        record_complete = (
            runner.get("runner_status") == "complete"
            and final_path.exists()
            and trace_path.exists()
        )
        add_result(
            domain_totals[domain]["baseline"],
            evaluate_record(reference, baseline, hypothesis_is_final=False),
        )
        if record_complete:
            final = load(final_path)
            add_result(
                domain_totals[domain]["final"],
                evaluate_record(reference, final, hypothesis_is_final=True),
            )
            reference_by_id = {int(item["id"]): item for item in reference["segments"]}
            for patch in accepted_patches(trace_path):
                patch_records.append(patch_outcome(run_id, patch, reference_by_id))
        elif args.allow_incomplete:
            add_result(
                domain_totals[domain]["final"],
                evaluate_record(reference, baseline, hypothesis_is_final=False),
            )
            failure_records.append(
                {
                    "run_id": run_id,
                    "runner_status": runner.get("runner_status", "missing"),
                    "error": runner.get("error") or runner.get("console_tail") or "",
                }
            )
        else:
            raise SystemExit(f"required outputs are missing for {run_id}")
        if runner:
            runtime_rows.append(runner)
        if index % 50 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)}", flush=True)

    domain_metrics = [
        {
            "domain": domain,
            "baseline": result_json(domain_totals[domain]["baseline"]),
            "final": result_json(domain_totals[domain]["final"]),
        }
        for domain in domains
    ]
    patch_stats = {
        "accepted": len(patch_records),
        "improved": sum(row["outcome"] == "improved" for row in patch_records),
        "worsened": sum(row["outcome"] == "worsened" for row in patch_records),
        "neutral": sum(row["outcome"] == "neutral" for row in patch_records),
        "net_error_delta": sum(int(row["delta"]) for row in patch_records),
        "records": patch_records,
    }
    summary = {
        "status": "complete",
        "metric_label": "CER",
        "language_label": "中文",
        "recordings": len(rows),
        "domains": domain_metrics,
        "aggregate": aggregate(domain_metrics),
        "patches": {
            key: patch_stats[key]
            for key in ("accepted", "improved", "worsened", "neutral", "net_error_delta")
        },
        "failures": {
            "count": len(failure_records),
            "scoring": "baseline_unchanged" if failure_records else "none",
            "records": failure_records,
        },
        "runtime": {
            "audio_hours": sum(float(row["audio_seconds"]) for row in runtime_rows) / 3600,
            "cumulative_run_hours": sum(float(row["total_seconds"]) for row in runtime_rows) / 3600,
            "batch_wall_hours": float(batch["batch_wall_seconds"]) / 3600,
            "llm_requests": sum(
                int((row.get("counters") or {}).get("llm_requests") or 0)
                for row in runtime_rows
            ),
            "asr_requests": sum(
                int((row.get("counters") or {}).get("asr_requests") or 0)
                for row in runtime_rows
            ),
        },
    }

    evaluation_root = run_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    write_json(evaluation_root / "summary.json", summary)
    write_json(evaluation_root / "patch_outcomes.json", patch_stats)
    markdown = render_markdown(summary)
    (evaluation_root / "summary.md").write_text(markdown, encoding="utf-8")
    write_report(summary, patch_stats, evaluation_root / "report.html")
    print(markdown)
    print(f"HTML report: {evaluation_root / 'report.html'}")


if __name__ == "__main__":
    main()
