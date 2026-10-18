# SPDX-License-Identifier: Apache-2.0
"""Evaluate historical 35-class argmax accuracy and mean one-hot BCE."""

import argparse
import json

import torch
from torch.nn import functional as F

from duplex_kws.features import collate, prepare_audio
from duplex_kws.inference import Predictor, batches, read_manifest


def evaluate(predictor, records, batch_size):
    total = 0
    correct = 0
    loss_sum = 0.0
    labels = {label: i for i, label in enumerate(predictor.labels)}
    for batch in batches(records, batch_size):
        targets = []
        for row in batch:
            label = row.get('label', row.get('txt'))
            if label is None:
                raise ValueError('Evaluation records require label or txt')
            label = label.strip('<>').lower()
            if label not in labels:
                raise ValueError('Unknown target label: ' + label)
            targets.append(labels[label])
        items = [
            prepare_audio(r['mic_wav'], r.get('ref_wav'),
                          predictor.audio_config) for r in batch
        ]
        logits = predictor.predict_features(collate(items))
        target = torch.tensor(targets)
        expected = F.one_hot(target, len(labels)).to(logits.dtype)
        loss_sum += float(
            F.binary_cross_entropy_with_logits(
                logits, expected, reduction='sum')) / len(labels)
        correct += int((logits.argmax(1) == target).sum())
        total += len(batch)
    if total == 0:
        raise ValueError('Evaluation manifest is empty')
    return {
        'num_examples': total,
        'correct': correct,
        'accuracy_percent': 100 * correct / total,
        'mean_bce': loss_sum / total,
        'decision_rule': 'argmax_logits'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-dir')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1:
        parser.error('threads and batch-size must be positive')
    torch.set_num_threads(args.threads)
    result = evaluate(Predictor(args.model_dir, args.device),
                      read_manifest(args.manifest), args.batch_size)
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:
            stream.write(content)
    print(content, end='')


if __name__ == '__main__':
    main()
