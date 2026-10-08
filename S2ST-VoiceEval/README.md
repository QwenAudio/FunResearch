# S2ST-VoiceEval

[🤗 Model weights](https://huggingface.co/liangwenrui/S2ST-VoiceEval)

Evaluate how well translated speech preserves the source speaker's voice,
without a target-language reference. Higher cosine scores indicate greater
speaker similarity.

The WavLM-based BIA+ALA+Rank model from
**Cross-Lingual Voice-Preservation Evaluation in Speech-to-Speech Translation**.

## Installation

Use Python 3.10–3.13:

```bash
git clone https://github.com/QwenAudio/FunResearch.git
cd FunResearch/S2ST-VoiceEval
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
s2st-voiceeval score \
  --model liangwenrui/S2ST-VoiceEval \
  --reference source.wav \
  --candidate translated.wav
```

The first run downloads and caches the complete model (about 1.30 GB).
A local model directory is also accepted. CUDA is used when available;
add `--device cpu` to use CPU.

WAV and FLAC inputs are converted to mono and resampled to 16 kHz automatically.
Use utterances between 0.25 and 300 seconds. Scores measure speaker similarity,
not translation accuracy.

## Python API

```python
from s2st_voiceeval import VoiceEvaluator

model = VoiceEvaluator("liangwenrui/S2ST-VoiceEval")
print(model.score("source.wav", "translated.wav"))
print(model.rank("source.wav", ["system_a.wav", "system_b.wav"]))
```

Reuse the model for multiple calls. Run `s2st-voiceeval --help` for CLI commands.

## License

Original project code: [Apache-2.0](LICENSE).
Model weights: [CC BY-SA 3.0](https://huggingface.co/liangwenrui/S2ST-VoiceEval).
The adapted Microsoft implementation retains its
[upstream license](src/s2st_voiceeval/UPSTREAM_LICENSE); see [NOTICE.md](NOTICE.md).
