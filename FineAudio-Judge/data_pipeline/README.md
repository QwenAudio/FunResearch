# Training-data construction

This directory contains the three training-data construction routes used by
FineAudio-Judge, together with controlled edits, cross-route generation, and
mismatch pairing. A prompt refers to the text description or audio-generation
instruction associated with an audio sample.

## Construction routes

| Route | Method | Inputs |
|---|---|---|
| Route 1 | Describe real-world audio through audio-grounded analysis and caption refinement. | Vript/FineVideo media, or a custom audio/video JSONL manifest. |
| Route 2 | Compose speech, sound effects, ambience, and music according to scene graphs and timelines. | AudioSet-Strong-compatible assets, SpeechCraft/AISHELL-3, and ParaSpeechCaps with EARS/EXPRESSO audio. |
| Route 3 | Create audio-generation prompts and synthesize audio with Audio-Oscar. | A text model for prompt construction and an Audio-Oscar deployment. |

Positive and negative construction have separate entry points:

| Route | Positive construction | Controlled negatives |
|---|---|---|
| Route 1 | [positive/pipeline.py](route1_real_world/positive/pipeline.py) | [negative/pipeline.py](route1_real_world/negative/pipeline.py) |
| Route 2 | [positive/pipeline.py](route2_deterministic_composition/positive/pipeline.py) | [negative/pipeline.py](route2_deterministic_composition/negative/pipeline.py) |
| Route 3 | [positive/pipeline.py](route3_prompt_generation/positive/pipeline.py) | [negative/pipeline.py](route3_prompt_generation/negative/pipeline.py) |

Route 1 negatives edit prompt claims while retaining the original audio.
Route 2 negatives either edit the prompt or modify and re-render the audio
composition. Route 3 negatives either edit the prompt or generate counterfactual
audio with Audio-Oscar while retaining the original evaluation prompt.

## Setup and usage

Run commands from the directory containing `pyproject.toml` and `data_pipeline/`:

```bash
python -m pip install -e .
python -m pip install -r data_pipeline/requirements.txt
```

Provide your own datasets, model checkpoints or endpoints, and compute. Local
model stages require a suitable CUDA environment, and media processing requires
FFmpeg/ffprobe. Audio-Oscar services use their own model-specific environments.

For example, start Route 1 from a custom media manifest:

```bash
python data_pipeline/route1_real_world/positive/pipeline.py \
  --input-manifest /path/to/media.jsonl \
  --omni-model YOUR_AUDIO_LANGUAGE_MODEL \
  --text-model YOUR_TEXT_MODEL \
  --output-dir outputs/route1/positive
```

Each input row contains `id`, `dataset`, and `audio_path` or `video_path`.
Use the entry points above with `--help` to inspect their arguments and stages.

Negative construction consumes completed positive outputs. Route 1 and Route 3
also require a separate rubric manifest for the exact source prompts, joined
by sample ID and dataset where applicable. Preserve rubric evidence spans and
dependency fields, and supply rubric structure without perceptual answer labels.
Route 2 derives construction rubrics from its scene graphs and does not call
an LLM during negative construction.

## Configuration

| File | Purpose |
|---|---|
| `route1_real_world/positive/config.json` | Duration and selection settings, language checks, captioning, and speaker alignment. |
| `route2_deterministic_composition/positive/config.json` | Example recipe for constructing 10 Route 2 samples. Pass it with `--recipe`; edit the counts, seeds, and composition settings for a custom run. |
| `route3_prompt_generation/positive/config.json` | Text-model endpoint and prompt generation, rewriting, audit, and deduplication settings. |
| `route3_prompt_generation/positive/audio_oscar.env.example` | Audio-Oscar checkout, model paths, service addresses, and generation settings. |
| `training_recipe.json` | Recorded training construction quotas and policies. |

For Route 2, select the example recipe with
`--recipe data_pipeline/route2_deterministic_composition/positive/config.json`
when running `data_pipeline/route2_deterministic_composition/positive/pipeline.py`
from the repository root. Supply source-data paths and the caption endpoint
through the entry point's command-line arguments.

## Additional pairing methods

| Method | Description | Entry point |
|---|---|---|
| Cross-route generation | Generate new Audio-Oscar audio from Route 1/2 prompts. These are intended matching candidates. | [pipeline.py](additional_negatives/cross_route_generation/pipeline.py) |
| Constrained mismatch pairing | Pair existing prompts and audio across or within routes, with donor non-reuse and compatibility-based selection. | [pipeline.py](additional_negatives/mismatch_pairing/pipeline.py) |

## Main outputs

Paths below are relative to each run's output directory:

| Construction | Pair manifest |
|---|---|
| Route 1 positive | `combined/audio_caption_pairs.jsonl` |
| Route 1 negative | `output/path1_negative_unlabeled.jsonl` |
| Route 2 positive | `pairs.jsonl` |
| Route 2 negative | `prompt_side/pairs.jsonl`, `audio_side/pairs.jsonl` |
| Route 3 positive | `pairs.jsonl` |
| Route 3 negative | `manifests/negative_pairs.jsonl` |
| Cross-route generation | `manifests/generated_pairs.jsonl` |
| Mismatch pairing | `mismatch_pairs.jsonl` |

Each run also saves audit reports and provenance. Rubric decomposition,
perceptual labeling, and final SFT-data assembly are separate steps.
See [examples](../examples/README.md) for audio–prompt pairs from all three routes.
Source materials and generated audio remain subject to their applicable
upstream licenses and model/service terms.
