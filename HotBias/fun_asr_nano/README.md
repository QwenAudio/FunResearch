# Fun-ASR-Nano integration

Run these commands from the HotBias repository root. Use the toolkit-format
`Fun-ASR-Nano-2512` model and a compatible FunASR environment. If Nano is not
already registered in that environment, pass `--nano-model-code` pointing to
`model.py` in the [Fun-ASR repository](https://github.com/QwenAudio/Fun-ASR).

For a single audio file, the final aligned checkpoint is ready for inference:

```bash
python fun_asr_nano/infer.py \
  --model-dir /path/to/Fun-ASR-Nano-2512 \
  --nano-model-code /path/to/Fun-ASR/model.py \
  --checkpoint /path/to/hotbias-checkpoints/nano_aligned.pt \
  --audio /path/to/audio.wav \
  --hotword '池州' --hotword '赤州'
```

The output contains the base and HotBias transcriptions. Candidate words are
deduplicated and retained in the supplied order up to the default cap of 40.
This decision never reads a transcript. Base Top-50 and token-Trie gating are
enabled. The standalone inference command runs without a reference transcript.

For target alignment, `initialize_adapter.py` creates a training-only Nano initialization
from the Qwen3 transfer package. The six dimension-dependent target projections
start fresh; the shared core and compatible projection biases/scales are loaded.
`prepare_manifest.py` validates a training JSONL and wraps it in the selection
format used by `train.py`.

The released three checkpoints do not include the six randomly initialized
Nano projection matrices from the original run before training. Recreating an
initialization from `source_transfer.pt` copies the same 142 compatible tensors
and initializes those six matrices anew; training follows the reported method
but is not expected to reproduce the published final bytes exactly.

```bash
python fun_asr_nano/initialize_adapter.py \
  --transfer-checkpoint /path/to/hotbias-checkpoints/source_transfer.pt \
  --output-checkpoint outputs/nano_init.pt
python fun_asr_nano/prepare_manifest.py \
  --train-jsonl /path/to/train.jsonl \
  --output outputs/nano_selection.json
torchrun --standalone --nproc_per_node=8 fun_asr_nano/train.py \
  --model-dir /path/to/Fun-ASR-Nano-2512 \
  --nano-model-code /path/to/Fun-ASR/model.py \
  --initial-checkpoint outputs/nano_init.pt \
  --selection-manifest outputs/nano_selection.json \
  --local-audio-root /path/to/audio-root \
  --output-dir outputs/nano_train \
  --epochs 1 --gradient-accumulation 8
```

Each UTF-8 training JSONL line needs `raw_audio_path` or `audio_path`, an
`asr_hotwords` list or comma-separated string, and a transcript in `text`,
`label`, or the last assistant message. `duration` is seconds and is used to
report selected hours. Relative audio paths resolve against
`--local-audio-root`; `oss://bucket/key` resolves to that root plus `key`.

```json
{"audio_path":"audio/001.wav","text":"请打开池州地图","asr_hotwords":["池州","赤州"],"duration":3.2}
```

For a labeled batch evaluation, the manifest is a JSON array of dataset
specifications. The `jsonl` and optional `audio_root` paths resolve relative
to the manifest directory. Evaluation rows use the same audio and hotword
fields; `text` or `label` supplies the reference for metrics only. The
per-sample predictions are written to `predictions.jsonl`.

```json
[{"name":"example","jsonl":"eval.jsonl","audio_root":"/path/to/audio-root","hotword_field":"asr_hotwords","stages":[3]}]
```

```bash
python fun_asr_nano/evaluate.py \
  --model-dir /path/to/Fun-ASR-Nano-2512 \
  --nano-model-code /path/to/Fun-ASR/model.py \
  --checkpoint /path/to/hotbias-checkpoints/nano_aligned.pt \
  --manifest /path/to/eval_manifest.json \
  --output-dir outputs/nano_eval
```

The default route uses Top-50/Trie gating. The historical evaluation also
used an external character-CTC gate. To enable its character-mode acoustic
gate, add these two arguments to `infer.py` or `evaluate.py`:

```bash
--ctc-temporal-gate --ctc-model-dir /path/to/wav2vec2-large-xlsr-53-chinese-zh-cn
```

This CTC model is a separate frozen dependency and is not one of the three
HotBias checkpoints. If its directory contains only a legacy
`pytorch_model.bin`, Transformers requires PyTorch 2.6 or newer; a verified
`safetensors` copy of the same CTC model also works. The CTC option has unit
coverage for character alignment and safe model loading, but was not run end
to end in the local PyTorch 2.3 environment. Even with that option enabled,
the reference-independent candidate cap changes cases with more than 40
hotwords from the historical evaluator, so published paper metrics need a new
evaluation before comparison. `infer.py` creates temporary files for its
single utterance and ignores reference-based metrics.
