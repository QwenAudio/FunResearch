#!/usr/bin/env python3
"""Route 1 controlled prompt-negative construction entry point.

This exposes the production stages used for the training construction:
prepare, generate, materialize, rebuild, audit, status, and run.
"""

from __future__ import annotations

import sys

try:
    from .stages.counterfactual_pipeline import main as counterfactual_main
    from .stages.vllm_offline_runner import main as offline_main
except ImportError:  # Direct script execution.
    from stages.counterfactual_pipeline import main as counterfactual_main
    from stages.vllm_offline_runner import main as offline_main


OFFLINE_COMMANDS = {
    "generate-shard": "run-shard",
    "merge-shards": "merge",
    "shard-status": "status",
}


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: pipeline.py COMMAND [OPTIONS]\n\n"
            "Core commands:\n"
            "  prepare          freeze inputs and build the balanced edit plan\n"
            "  generate         endpoint-backed exact-span generation\n"
            "  generate-shard   direct local vLLM generation for one deterministic shard\n"
            "  merge-shards     merge and revalidate offline shard checkpoints\n"
            "  shard-status     inspect offline shard completion\n"
            "  materialize      apply validated patches and verify frozen audio\n"
            "  rebuild          rebuild affected question text when requested\n"
            "  audit            audit and assemble outputs\n"
            "  status           inspect pipeline checkpoints\n"
            "  run              endpoint-backed all-stage compatibility path\n\n"
            "Run `pipeline.py COMMAND --help` for command arguments."
        )
        return
    command = sys.argv[1]
    if command in OFFLINE_COMMANDS:
        sys.argv[1] = OFFLINE_COMMANDS[command]
        offline_main()
        return
    counterfactual_main()


if __name__ == "__main__":
    main()
