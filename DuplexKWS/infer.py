# SPDX-License-Identifier: Apache-2.0
"""Run single-file or batched JSONL inference."""

import argparse
import contextlib
import json
from pathlib import Path
import sys

import torch

from duplex_kws.inference import Predictor, batches, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--mic', help='Microphone WAV path')
    source.add_argument('--manifest',
                        help='JSONL with mic_wav and optional ref_wav')
    parser.add_argument('--ref', help='Aligned playback reference WAV')
    parser.add_argument('--output', help='Output JSONL (default: stdout)')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:N')
    parser.add_argument('--model-dir',
                        help='Bundle root (defaults to script location)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.ref and not args.mic:
        parser.error('--ref requires --mic; batch references belong in JSONL')
    if args.output:
        inputs = [args.manifest, args.mic, args.ref]
        if any(path and Path(path).resolve() == Path(args.output).resolve()
               for path in inputs):
            parser.error('--output must not overwrite an input file')
    if min(args.batch_size, args.threads) < 1 or not 1 <= args.top_k <= 35:
        parser.error(
            'batch-size/threads must be positive; top-k must be 1..35')
    torch.set_num_threads(args.threads)
    predictor = Predictor(args.model_dir, args.device)
    records = (read_manifest(args.manifest)
               if args.manifest else [{
                   'mic_wav': args.mic,
                   'ref_wav': args.ref
               }])
    output = (open(args.output, 'w', encoding='utf-8')
              if args.output else contextlib.nullcontext(sys.stdout))
    with output as stream:
        for batch in batches(records, args.batch_size):
            for result in predictor.predict_batch(batch, args.top_k):
                stream.write(json.dumps(result, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
