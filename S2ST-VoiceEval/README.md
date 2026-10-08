# S2ST-VoiceEval

[🤗 Model weights](https://huggingface.co/liangwenrui/S2ST-VoiceEval)

**Score how well translated speech preserves the source speaker's voice.**
Give the evaluator a source recording and translated candidates; it returns
cosine similarity scores and a best-first ranking. No target-language reference,
transcript, teacher model or language-classification head is needed.

Companion implementation for **Cross-Lingual Voice-Preservation Evaluation in
Speech-to-Speech Translation**, part of [Fun Research](https://github.com/QwenAudio/FunResearch).

中文快速使用说明见 [QUICKSTART.zh-CN.md](QUICKSTART.zh-CN.md)。

## Quick start

Use Python **3.10–3.13** (tested with 3.12):

```bash
git clone https://github.com/QwenAudio/FunResearch.git
cd FunResearch/S2ST-VoiceEval
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

s2st-voiceeval score \
  --model liangwenrui/S2ST-VoiceEval \
  --reference source.wav \
  --candidate translated.wav
```

The first run downloads the complete model (about 1.30 GB) from
[Hugging Face](https://huggingface.co/liangwenrui/S2ST-VoiceEval) and caches it
locally. No separate Microsoft checkpoint is needed. You can also pass a local
directory containing the downloaded `config.json` and `model.safetensors`
to `--model`.

CUDA is selected when available; otherwise inference runs on CPU.
Use `--device cpu` or `--device cuda:0` to choose explicitly.
Install matching PyTorch and torchaudio builds if your platform needs a
specific CPU/CUDA build. Tested: PyTorch/torchaudio 2.7.0 and Transformers 4.53.1.

## Python API

```python
from s2st_voiceeval import VoiceEvaluator

evaluator = VoiceEvaluator("liangwenrui/S2ST-VoiceEval")

score = evaluator.score("source.wav", "translated.wav")
scores = evaluator.score_many("source.wav", ["system_a.wav", "system_b.wav"])
ranking = evaluator.rank("source.wav", ["system_a.wav", "system_b.wav"])

print(score)     # cosine similarity in [-1, 1]
print(scores)    # scores in the same order as the input paths
print(ranking)   # best-first: rank, index, candidate, score
```

Load the evaluator once and reuse it. The reference is encoded once per
`score_many`/`rank` call. `embed(path)` returns a 256-dimensional, unnormalized
float32 NumPy embedding. Exact ties retain input order in the displayed ranking.

`VoiceEvaluator` also accepts a Hugging Face model repository ID via `model`,
with optional `revision` and `local_files_only=True`. It downloads only the
exported configuration and weights and does not execute remote code.

## Rank several candidates

```bash
s2st-voiceeval rank \
  --model liangwenrui/S2ST-VoiceEval \
  --reference source.wav \
  --candidate system_a.wav system_b.wav system_c.wav
```

The command prints JSON. Larger scores indicate greater model-estimated speaker
similarity. A score is not a probability or a translation-quality score; there
is no universal pass/fail threshold.

## Batch evaluation

Create a JSONL manifest with one reference and candidate list per line:

```json
{"id": "utterance-1", "reference": "audio/source.wav", "candidates": ["audio/a.wav", "audio/b.wav"]}
```

Relative audio paths are resolved against the **manifest's directory**.

```bash
s2st-voiceeval batch \
  --model liangwenrui/S2ST-VoiceEval \
  --input manifest.jsonl \
  --output scores.jsonl
```

Each output record contains the input ID, reference, ranking and precision.
An invalid input aborts the run without publishing a partial output file.
Existing outputs are not overwritten.

## PairAcc

PairAcc needs preference labels in addition to similarity scores.
For each comparison, supply `score_a`, `score_b` and `preferred` (`"a"` or `"b"`):

```bash
s2st-voiceeval pairacc --input examples/pairs.jsonl
```

The example contains illustrative scores and returns 66.67% accuracy.
The command pools comparisons directly, as in the paper. Predicted score ties
(difference at most 1e-8) receive half credit, following the original evaluation
implementation. Exclude tied human/proxy labels before calling this command.
The training teacher's 0.01 tie threshold is not an evaluation threshold.

## Audio and numerical behavior

- WAV and FLAC are supported through SoundFile; other formats depend on the
  installed libsndfile build.
- Channels are averaged and audio is resampled to 16 kHz automatically.
- Full utterances are encoded individually, without padding, cropping, VAD,
  loudness normalization or waveform layer normalization.
- Inputs must be 0.25–300 seconds. Short utterances are recommended for speed
  and memory use; the duration limit does not guarantee that a long file fits
  in every GPU.
- Silent references and non-finite audio are rejected. Silent candidates remain
  scoreable, matching the original evaluation protocol.
- `--precision auto` uses CUDA BF16 when supported, matching paper extraction.
  CPU uses FP32. `--precision fp32` selects FP32 explicitly.
- Precision and hardware can slightly change scores. Reproduce paper extraction
  using `--precision bf16` and the tested dependencies. Model checksums are
  verified on load.

## Model and results

The released evaluator is **BIA+ALA+Rank**, the paper's full two-stage model.
Bilingual identity adaptation uses contrastive and similarity-preservation
losses; target-reference ranking distillation adds supervision from generated
speech. Both stages retain alternating language adaptation.

| Mandarin–English EMIME | Proxy PairAcc | Human PairAcc | Cross-language EER |
| --- | ---: | ---: | ---: |
| WavLM baseline, source reference | 71.44% | 71.75% | 7.78% |
| S2ST-VoiceEval, source reference | **75.26%** | **78.53%** | **1.49%** |

Proxy evaluation uses 7,056 comparisons. The human evaluation uses 177
comparisons screened for agreement with ECAPA preferences.
See [MODEL_CARD.md](MODEL_CARD.md) for the model overview.

## Development and export

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Maintainers can merge the paper checkpoint with its exact original base:

```bash
python scripts/export_checkpoint.py \
  --checkpoint /path/to/checkpoint-1000.pt \
  --base-checkpoint /path/to/wavlm_large_finetune.pth \
  --wavlm-config /path/to/wavlm-large \
  --output ./checkpoints/s2st-voiceeval
```

The exporter verifies source/base configuration hashes and every expected delta
key, retains all normalization buffers, and writes a full safetensors model.
It excludes training metadata, local paths, optimizer and classifier states.
It refuses to overwrite an existing export.

## License and attribution

Project code is covered by [LICENSE](LICENSE), with the third-party model
implementation and pretrained components covered by their retained notices.
See [NOTICE.md](NOTICE.md) and
[UPSTREAM_LICENSE](src/s2st_voiceeval/UPSTREAM_LICENSE). Internal training audio
is not part of this release.

Paper title: **Cross-Lingual Voice-Preservation Evaluation in Speech-to-Speech
Translation**. A bibliographic entry will be added when publication details are
available.
