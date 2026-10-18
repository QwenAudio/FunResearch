# SPDX-License-Identifier: Apache-2.0
"""Self-contained model loading, label decoding and file inference."""

import json
from pathlib import Path

import torch
import yaml

from .features import collate, prepare_audio
from .model import DuplexKWS

ROOT = Path(__file__).resolve().parent.parent


class Predictor:

    def __init__(self, model_dir=None, device='cpu'):
        root = Path(model_dir) if model_dir is not None else ROOT
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        config = yaml.safe_load((root / 'configs/inference.yaml').read_text())
        self.audio_config = config['audio']
        self.labels = json.loads((root / 'resources/labels.json').read_text())
        self.model = DuplexKWS(**config['model'])
        if len(self.labels) != self.model.output_dim:
            raise ValueError('Label count does not match model output size')
        weights = torch.load(root / 'weights/avg_10.pt',
                             map_location='cpu',
                             weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_features(self, batch):
        """Return CPU logits from a batch produced by features.collate."""
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        return self.model(**tensors).cpu()

    def decode(self, logits, top_k=5):
        if not 1 <= top_k <= len(self.labels):
            raise ValueError('top_k must be between 1 and 35')
        scores = logits.sigmoid()
        # Historical Speech Commands accuracy uses argmax on raw logits.
        indices = logits.argsort(dim=1, descending=True, stable=True)
        results = []
        for row, order in enumerate(indices):
            best = int(order[0])
            results.append({
                'class_id':
                best,
                'label':
                self.labels[best],
                'score':
                float(scores[row, best]),
                'top_k': [{
                    'class_id': int(i),
                    'label': self.labels[int(i)],
                    'score': float(scores[row, i]),
                } for i in order[:top_k]],
                'scores':
                scores[row].tolist(),
                'logits':
                logits[row].tolist(),
            })
        return results

    def predict_batch(self, records, top_k=5):
        if not records:
            return []
        items = [
            prepare_audio(r['mic_wav'], r.get('ref_wav'), self.audio_config)
            for r in records
        ]
        results = self.decode(self.predict_features(collate(items)), top_k)
        for record, item, result in zip(records, items, results):
            result['key'] = record.get('key', str(record['mic_wav']))
            result['has_reference'] = item[2]
            result['frames'] = item[3]
        return results

    def predict(self, mic_wav, ref_wav=None, top_k=5):
        return self.predict_batch([{
            'mic_wav': mic_wav,
            'ref_wav': ref_wav
        }], top_k)[0]


def read_manifest(path):
    """Resolve paths relative to the JSONL file; accept WeKws wav/mic_wav."""
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            mic = row.get('mic_wav') or row.get('wav')
            if not mic:
                raise ValueError('Missing mic_wav at line {}'.format(line_no))
            row['mic_wav'] = str((path.parent / mic).resolve())
            if row.get('ref_wav'):
                row['ref_wav'] = str((path.parent / row['ref_wav']).resolve())
            else:
                row.pop('ref_wav', None)
            yield row


def batches(records, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    pending = []
    for row in records:
        pending.append(row)
        if len(pending) == batch_size:
            yield pending
            pending = []
    if pending:
        yield pending
