# Third-party notices

FineAudio-Judge code is released under the Apache License 2.0.
Third-party software, models, services, and source materials used by this
project remain subject to their respective licenses and terms of use.

## Audio-Oscar integration

Route 3 and cross-route audio generation use
[Audio-Oscar](https://github.com/ziye26/Audio-Oscar), under the
Apache-2.0 license.

This repository does not include the complete Audio-Oscar source code.
It provides a local patch at
`data_pipeline/route3_prompt_generation/positive/patches/audio_oscar.patch`,
which is applied to the upstream revision required by the generation runner.

The patch adjusts floating-point tolerance during peak normalization in
`executor/mixer.py` and makes the Demucs connection settings configurable
through environment variables in `music/AceStep.py` and `song/demucs.py`.

The Audio-Oscar planner uses Qwen3.5-35B-A3B. Audio-generation models used
through Audio-Oscar include CosyVoice3, VoxCPM2, MMAudio, StableAudio,
MeanAudio, ACE-Step, and HeartMuLa. These models retain their respective
licenses and terms; Audio-Oscar's software license does not replace them.

## Models and services

| Model or service | Usage | Terms |
|---|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Judge base model; Route 1 audio-grounded caption generation and speaker attribution | Apache-2.0 |
| Qwen3.5-35B-A3B | Route 3 prompt generation; Audio-Oscar planner; rubric decomposition for training and evaluation data | Upstream model terms apply |
| Qwen3.5-Omni-Plus | Training-data rubric annotation; Rubric-QA reference-label annotation; evaluation baseline | Upstream model or service terms apply |
| Gemini 2.5 Pro | Rubric-QA reference-label annotation and evaluation baseline | Upstream service terms apply |
| CosyVoice3, VoxCPM2, MMAudio, StableAudio, MeanAudio, ACE-Step, HeartMuLa | Audio generation within Audio-Oscar | Each model's upstream licenses and terms apply |
| SeedAudio 1.0 | Human-preference candidates and generator-OOD evaluation audio | Upstream service terms apply |
| Qwen-Audio-3.0-Gen-Preview | Generator-OOD evaluation audio | Upstream service terms apply |
| AnyAudio-Judge | Evaluation baseline and comparison rubrics | Upstream repository and model terms apply |

The third-party model weights listed above are not bundled with this
code release. Depending on the pipeline stage, users must supply model
checkpoints for local inference or configure external service endpoints.
The applicable model licenses and service terms govern their use.

## Python dependencies

Dependencies listed in `pyproject.toml` and `data_pipeline/requirements.txt`
(including ms-swift, vLLM, PyTorch, Transformers, PEFT, Whisper, SpeechBrain,
Demucs, FunASR, and ModelScope) are installed separately and are governed
by their own licenses.

## Training-data sources and construction

| Route | Sources and construction |
|---|---|
| Route 1: Real-world audio | Native audio from FineVideo and Vript videos, with audio-grounded captions generated using Qwen3-Omni-30B-A3B-Instruct |
| Route 2: Deterministic composition | AudioSet-Strong assets and speech audio with annotations from the resources described below |
| Route 3: Prompt-driven generation | Text descriptions generated using Qwen3.5-35B-A3B, followed by audio generation through Audio-Oscar |


Qwen3.5-35B-A3B is used to decompose prompts into rubric questions.
Qwen3.5-Omni-Plus is used to annotate the resulting training-data questions.

## Benchmark sources and annotation

| Resource | Usage |
|---|---|
| FineVideo | Real-audio source |
| ESC-50, UrbanSound8K, InstructTTSEval | Sound-event and speech assets for composed audio |
| ASG-Bench | Scene-description text resources |
| Audio-Oscar | Audio generation |
| Qwen-Audio-3.0-Gen-Preview | Audio generation |
| SeedAudio 1.0 | Audio generation, including human-preference candidates |

Qwen3.5-35B-A3B is used to decompose evaluation prompts into rubric questions.
Gemini 2.5 Pro and Qwen3.5-Omni-Plus are used to annotate the resulting
Rubric-QA questions.

## Released examples

This GitHub repository includes 30 audio–prompt examples from the
training-data construction pipeline, organized as follows:

| Directory | Source or construction method | Pairs |
|---|---|---:|
| `examples/audio_prompt_pairs/A1/route1/` | FineVideo; `route1_demo_01` through `route1_demo_05` | 5 |
| `examples/audio_prompt_pairs/A2/route1/` | Vript; `route1_demo_06` through `route1_demo_10` | 5 |
| `examples/audio_prompt_pairs/A2/route2/` | Deterministic composition | 10 |
| `examples/audio_prompt_pairs/A2/route3/` | Audio-Oscar generation | 10 |

The benchmark is released separately on Hugging Face. The A1/A2 directory
names above refer to the GitHub examples.

## Applicable data licenses and terms

The data in `examples/audio_prompt_pairs/A1/` remains subject to the
[Terms of Use of FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo#terms-of-use-for-finevideo)
and the applicable original source licenses.

The data in `examples/audio_prompt_pairs/A2/` is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), to the extent
that this project holds or is authorized to license the relevant rights.

For the Hugging Face benchmark, FineVideo-derived data likewise remains
subject to the FineVideo Terms of Use and original source licenses.
The remaining benchmark data is released under CC BY-NC 4.0, to the extent
that this project holds or is authorized to license the relevant rights.

Third-party audio, text, and other source materials retain their
upstream licenses and terms of use. Generated audio also remains
subject to the applicable model or service terms. The project's data
license does not replace or waive these requirements.

See the root README and the notices accompanying each data release
for the project's data-license statement and disclaimer.
