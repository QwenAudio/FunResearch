"""Decompose one prompt through the reference Qwen/vLLM JSON endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


from rubric_decomposition.decompose import decompose_prompt


async def run(args: argparse.Namespace) -> None:
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt
    audit_dir = args.checkpoint_dir or args.output.with_name(f"{args.output.stem}_audit")
    result = await decompose_prompt(
        prompt,
        base_url=args.base_url,
        model=args.model,
        tokenizer=args.tokenizer,
        output_dir=audit_dir,
        timeout=args.timeout,
        temperature=args.temperature,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True,
                        help="Qwen-compatible vLLM OpenAI API base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="audit directory; defaults to OUTPUT_STEM_audit beside --output")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--temperature", type=float,
        help="sampling temperature; defaults to FINEAUDIO_DECOMPOSITION_TEMPERATURE or 0",
    )
    parser.add_argument(
        "--seed", type=int,
        help="sampling seed; defaults to FINEAUDIO_DECOMPOSITION_SEED when set",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
