#!/usr/bin/env python3
"""Initialize Fun-ASR-Nano projections from a Qwen3 transfer package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hotbias.transfer import build_target_initialization_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--audio-dim", type=int, default=1024)
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--force", action="store_true", help="Replace an existing output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.audio_dim < 1 or args.text_dim < 1:
        raise ValueError("target dimensions must be positive")
    if not args.transfer_checkpoint.is_file():
        raise FileNotFoundError(args.transfer_checkpoint)
    if args.output_checkpoint.exists() and not args.force:
        raise FileExistsError(
            f"{args.output_checkpoint} exists; pass --force to replace it"
        )

    torch.manual_seed(args.seed)
    checkpoint, report = build_target_initialization_checkpoint(
        args.transfer_checkpoint,
        audio_dim=args.audio_dim,
        text_dim=args.text_dim,
        target_name="Fun-ASR-Nano-2512",
    )
    output = args.output_checkpoint
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    print(json.dumps({"output_checkpoint": str(output), "transfer_report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
