# DuplexKWS-speechcommands-v2

DuplexKWS provides standalone PyTorch inference for a 35-class English
Speech Commands v2 classifier. It accepts a microphone recording and an
optional device playback reference.

The package includes a pretrained checkpoint, feature extraction, a Python
API, and scripts for single-file inference, batch inference, and evaluation.
The code and bundled weights are licensed under Apache-2.0.

## Installation

Use Python 3.10. Run the following commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a CPU-only setup, install the matching PyTorch and torchaudio versions
before installing the remaining dependencies:

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

This release was validated on CPU with PyTorch and torchaudio 2.5.1.
For GPU inference, use `--device cuda:0` with a compatible CUDA build;
GPU execution has not been verified for this release.

The included scripts run directly from this directory. You do not need to
install this project as a package or configure an external `PYTHONPATH`.

## Quick start

Run inference or evaluation on the bundled examples:

```bash
python infer.py --mic examples/yes.wav
python infer.py --mic examples/no.wav
python infer.py --mic examples/echo/sir_0/yes_mic.wav --ref examples/echo/sir_0/yes_ref.wav
python infer.py --manifest examples/batch.jsonl --output predictions.jsonl
python evaluate.py --manifest examples/batch.jsonl
```

For your own recordings, provide the microphone audio and its corresponding
device playback reference:

```bash
python infer.py --mic your_microphone.wav --ref your_reference.wav --top-k 3
```

### Bundled examples

The bundle contains two clean recordings and 24 paired echo test examples:

- `examples/yes.wav` and `examples/no.wav`: unchanged recordings from the
  official Speech Commands v2 test split, for microphone-only inference.
- `examples/echo/sir_<dB>/`: one "yes" and one "no" example at each SIR level:
  -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, and 20 dB. Each example has
  a `*_mic.wav` mixture and its corresponding `*_ref.wav` playback reference.
- `examples/batch.jsonl`: all 26 examples, with ground-truth labels and
  `sir_db` for echo conditions. Lower SIR means stronger interfering echo.

Manifest labels identify the near-end spoken word. Strong echo can cause
incorrect predictions; all examples retain their original labels.

### Inference output

Inference produces one JSON object per input. Each result includes `class_id`,
`label`, `score`, `top_k`, all 35 `scores`, raw `logits`, `frames`, and
`has_reference`. The order of `scores` and `logits` follows
`resources/labels.json`.

Scores are independent **sigmoid** outputs and do not sum to one. The class
with the largest logit is selected, matching the original 35-class evaluation.

## Python API

Run the following from this directory, or add this directory to your
application's import path:

```python
from duplex_kws import Predictor

predictor = Predictor(device="cpu")
result = predictor.predict("your_microphone.wav", "your_reference.wav")
print(result["label"], result["score"])

results = predictor.predict_batch([
    {"key": "single", "mic_wav": "your_microphone.wav"},
    {"key": "paired", "mic_wav": "your_microphone.wav",
     "ref_wav": "your_reference.wav"},
])
```

Weights and resources are located relative to the bundle, regardless of the
working directory. Use `Predictor(model_dir="/path/to/bundle")` to select a
different copy of the bundle.

## Audio input and inference behavior

- Audio is resampled to 16 kHz. Multichannel inputs retain the **first channel**.
- Provide microphone and reference files separately; a second channel in the
  microphone file is not automatically treated as the playback reference.
- Microphone and reference recordings must have corresponding starting
  timestamps. The reference is cropped or right-padded to the microphone
  recording's length. The model does not perform automatic waveform time
  alignment.
- Omit the reference when playback is inactive. An explicitly supplied silent
  reference activates the echo branch and is not equivalent to omitting it.
- Audio shorter than 25 ms is zero-padded before feature extraction. Features
  are padded to at least 98 frames; max pooling excludes padded frames.

Each call resets the GRU states and classifies a complete window. The intended
input is a one-second command recording. A longer recording still produces a
single utterance-level prediction. Keyword timestamps and enhanced audio are
not provided.

## Batch inference and evaluation

Each JSONL manifest record accepts `mic_wav` (or `wav`) and an optional
`ref_wav`. Relative audio paths are resolved from the manifest file's
directory, independently of the process working directory.

```json
{"key":"sample_1","mic_wav":"audio/mic.wav","ref_wav":"audio/ref.wav","label":"yes"}
```

For evaluation, provide a word in `label` or the original token in `txt`,
such as `<YES>`. Use the original test split when comparing with historical
results:

```bash
python evaluate.py --manifest your_test.jsonl --output metrics.json
```

Accuracy is calculated using the class with the largest of the 35 logits.

## License and attribution

### Code and weights

The code and bundled `weights/avg_10.pt` are released under Apache-2.0.
The complete Apache License, Version 2.0 is included in [LICENSE](LICENSE).

### Runtime dependencies

Runtime dependencies are installed separately, not vendored. Their own
distribution licenses and notices apply: PyTorch, torchaudio, PyYAML, and
SoundFile, including their transitive/native libraries. This project's
Apache-2.0 license does not replace those licenses. See each installed
distribution's notices. Dataset and upstream software licenses are separate
from the code and weight license.

### Example audio

The clean example recordings and the near-end speech used in the echo mixtures
come from Speech Commands v0.02, released by Pete Warden and the TensorFlow
Speech Commands project and its contributors:

| Label | Original dataset path | Local example |
| --- | --- | --- |
| yes | `yes/022cd682_nohash_0.wav` | `examples/yes.wav` |
| no | `no/03401e93_nohash_0.wav` | `examples/no.wav` |

Both original paths appear in the official `testing_list.txt`. The clean WAV
files are byte-identical to the originals; only their filenames have changed.
Echo mixtures retain the test-set mixing and cropping.

The Speech Commands audio used in the clean recordings and echo mixtures
retains the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/), including its
disclaimer of warranties; see the
[full license text](https://creativecommons.org/licenses/by/4.0/legalcode.en).

- Source: [Speech Commands v0.02](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
- Citation: Pete Warden, *Speech Commands: A Dataset for Limited-Vocabulary
  Speech Recognition*, 2018, [arXiv:1804.03209](https://arxiv.org/abs/1804.03209).
