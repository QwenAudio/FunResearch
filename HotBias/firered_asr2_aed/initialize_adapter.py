"""Create the FireRedASR2-AED adapter initialization from a Qwen3 transfer package."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hotbias.transfer import build_target_initialization_checkpoint


MODEL_NAME = "FireRedTeam/FireRedASR2-AED"
MODEL_REVISION = "2304afed56eacfee6256dee5937ed22ffa0b64ec"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True, help="Qwen3 transferable package")
    parser.add_argument("--output", type=Path, required=True, help="New FireRed initialization checkpoint")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    source = args.transfer.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    checkpoint, report = build_target_initialization_checkpoint(
        source, audio_dim=1280, text_dim=1280, target_name=MODEL_NAME,
        reuse_shape_compatible_source_adapter=True,
    )
    if report["total_tensors"] != 148 or report["target_audio_dim"] != 1280 or report["target_text_dim"] != 1280:
        raise ValueError("Transfer package does not match the audited FireRedASR2-AED architecture")
    checkpoint.update(
        source_package_sha256=sha256(source), target_model_revision=MODEL_REVISION,
        trainable_scope="backbone_adapters", initialization_seed=args.seed,
        alignment_completed=False,
    )
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved initialization-only checkpoint: {output}")
    print(f"Source package SHA256: {checkpoint['source_package_sha256']}")
    print(f"Inherited tensors: {report['loaded_tensors']} / {report['total_tensors']}")


if __name__ == "__main__":
    main()
