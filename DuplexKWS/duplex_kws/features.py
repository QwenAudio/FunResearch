# Copyright (c) 2026 Wenet Community.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# Modified for DuplexKWS
"""Audio preprocessing"""

import torch
import torchaudio
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance import kaldi


def load_audio(path, sample_rate):
    waveform, original_rate = torchaudio.load(str(path), backend='soundfile')
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError('Audio must be nonempty: {}'.format(path))
    waveform = waveform[0].float()
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError('Audio contains nonfinite samples: {}'.format(path))
    if original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate,
                                                  sample_rate)
    return waveform


def fit_reference(reference, num_samples):
    if reference.numel() < num_samples:
        return F.pad(reference, (0, num_samples - reference.numel()))
    return reference[:num_samples]


def fbank(waveform, config):
    sample_rate = config['sample_rate']
    options = config['fbank']
    minimum = max(1, int(sample_rate * options['frame_length'] / 1000))
    if waveform.numel() < minimum:
        waveform = F.pad(waveform, (0, minimum - waveform.numel()))
    return kaldi.fbank((waveform * (1 << 15)).unsqueeze(0),
                       sample_frequency=sample_rate,
                       **options)


def prepare_audio(mic_path, ref_path, config):
    microphone = load_audio(mic_path, config['sample_rate'])
    mic_features = fbank(microphone, config)
    actual_frames = mic_features.shape[0]
    padded_frames = max(actual_frames, config['min_frames'])
    mic_features = F.pad(mic_features,
                         (0, 0, 0, padded_frames - actual_frames))
    active = ref_path is not None
    if active:
        reference = load_audio(ref_path, config['sample_rate'])
        reference = fit_reference(reference, microphone.numel())
        ref_features = fbank(reference, config)
        ref_features = F.pad(ref_features,
                             (0, 0, 0, padded_frames - actual_frames))
    else:
        ref_features = torch.zeros_like(mic_features)
    return mic_features, ref_features, active, actual_frames


def collate(items):
    if not items:
        raise ValueError('Cannot collate an empty batch')
    return {
        'mixture': pad_sequence([x[0] for x in items], batch_first=True),
        'reference': pad_sequence([x[1] for x in items], batch_first=True),
        'has_ref': torch.tensor([x[2] for x in items], dtype=torch.bool),
        'lengths': torch.tensor([x[3] for x in items], dtype=torch.long),
    }
