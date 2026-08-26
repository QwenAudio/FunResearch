<div align="center">

# Spoken Function Calling

### A New Perspective on Spoken Language Understanding for Large Audio Language Models

Yuezhang Peng, Yuxin Liu, Changfeng Gao, Zhifu Gao, Xiangang Li, Xie Chen<sup>†</sup>

<sup>†</sup>Corresponding author.

**ACM Multimedia 2026 (MM '26)**

[Paper](https://arxiv.org/pdf/2608.05126) · [Dataset documentation](data/README.md) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation) · [License](#license)

</div>


## Overview

Spoken Function Calling (SFC) reframes spoken language understanding as schema-constrained tool invocation. Given a spoken request, environmental context, and a dynamic set of function definitions, a model produces directly executable function calls. Missing required arguments are represented by the explicit `NAN` sentinel instead of being hallucinated.

This repository is the official implementation of **Spoken Function Calling: A New Perspective on Spoken Language Understanding for Large Audio Language Models**. It contains:

- **SFC-Bench**, a spoken function-calling benchmark built from 300 daily-life functions across 5 domains and 21 scenarios;
- four difficulty levels covering single-intent, multi-intent, and multi-turn interactions;
- in-domain and out-of-domain splits over 232 ID and 68 OOD functions;
- paired SFC and traditional SLU representations for the Level 1 comparison;
- supervised fine-tuning guidance and GRPO post-training code for Qwen2.5-Omni-7B;
- the paper's fine-grained SFC reward, an exact-match reward baseline, and multilevel evaluation tools.

## SFC-Bench

### Difficulty levels

| Level | Multi-intent | Multi-turn | Description |
|---|:---:|:---:|---|
| Level 1 | No | No | A direct, single-function request |
| Level 2 | Yes | No | Multiple functions requested in one turn |
| Level 3-1 | No | Yes | A multi-turn interaction with semantic completion |
| Level 3-2 | Yes | Yes | Multi-function, multi-turn interaction with the highest complexity |

### Released data

| Representation | Train-ID | Test-ID | Test-OOD | Notes |
|---|---:|---:|---:|---|
| Processed SFC | 5,534 | 2,590 | 771 | Turn-level records used by ms-swift |
| Processed SLU | 2,647 | 902 | 321 | Paired Level 1 SLU comparison |

The release also contains more than 7K raw benchmark records. Multi-turn examples are expanded into turn-level records in the processed SFC files, so processed counts differ from raw scenario counts. Exact per-file counts and checksums are available in [`data/manifest.json`](data/manifest.json).

### Data organization

```text
data/
├── raw/final_data_1228/          # Raw JSONL grouped by level and split
├── audio/final_data_audio/       # WAV files
├── processed/
│   ├── data_msswift/             # SFC training/evaluation format
│   └── data_slu/                 # Traditional intent-slot format
└── manifest.json                 # Counts, sizes, and MD5 checksums
```

A processed SFC record has the following structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "... function definitions ... environment ... <audio>"
    }
  ],
  "audios": ["audio/final_data_audio/level_1/test_id/id_1.wav"],
  "solution": "[\"add_reminder(content=\\\"submit report\\\", time=\\\"NAN\\\")\"]"
}
```

See [`data/README.md`](data/README.md) for the complete format description.


## Repository Layout

```text
SpokenFC/
├── configs/                      # Environment example and training config
├── data/                         # Raw, audio, and processed SFC-Bench data
├── ms-swift/                     # Vendored ms-swift source
├── project/
│   ├── eval/                     # Inference and multilevel metrics
│   ├── plugins/sfc_reward.py     # EM and fine-grained SFC rewards
│   ├── scripts/                  # Environment, path, and manifest utilities
│   └── train/                    # GRPO training entry points
├── MS_SWIFT_VERSION
├── VERSION
└── README.md
```

## Quick Start

### 1. Create an environment

The code targets Python 3.10+ and a CUDA environment suitable for Qwen2.5-Omni-7B. Install PyTorch, vLLM, and DeepSpeed versions compatible with your CUDA runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the vendored training framework and evaluation dependencies.
python -m pip install -e ./ms-swift
python -m pip install "vllm>=0.5.1,<0.10.2" "deepspeed>=0.14" \
  openai tqdm opencc-python-reimplemented regex
```

If the repository is distributed with audio through Git LFS, retrieve the audio files before training or evaluation:

```bash
git lfs install
git lfs pull
```

### 2. Configure paths

```bash
export MODEL_PATH=/path/to/Qwen2.5-Omni-7B
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

source project/scripts/setup_env.sh
```

All dataset paths are derived from the repository root. To use an external data directory or ms-swift installation, set `DATA_ROOT` or `MS_SWIFT_ROOT` before sourcing `setup_env.sh`.

If you moved the data from another machine and the JSON files contain stale absolute audio paths, normalize them once and regenerate the manifest:

```bash
bash project/scripts/normalize_paths.sh
bash project/scripts/generate_manifest.sh
```

## Training

### Supervised fine-tuning

The paper uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT. SFT is intentionally not duplicated in this repository. Use the files under `data/processed/data_msswift/` to prepare LLaMA-Factory multimodal conversations, then use the resulting checkpoint as the starting model for GRPO or evaluation.

### GRPO with exact-match reward

```bash
export MODEL_PATH=/path/to/base-or-sft-checkpoint
source project/scripts/setup_env.sh
bash project/train/grpo_em.sh
```

### GRPO with the fine-grained SFC reward

The fine-grained reward decomposes a function call into function-name, parameter-key, and parameter-value components, providing denser feedback than exact match.

```bash
export MODEL_PATH=/path/to/base-or-sft-checkpoint
source project/scripts/setup_env.sh
bash project/train/grpo_mixed.sh
```

Outputs are written to `outputs/grpo_em/` and `outputs/grpo_mixed/` by default.

> **Paper configuration.** The checked-in scripts default to BF16 and 8 generations as a lower-cost starting point. The final SpokenFC-7B setting reported in the paper uses FP16 and 32 generations per prompt. To match that setting, change `--torch_dtype` to `float16` and `--num_generations` to `32` in `project/train/grpo_mixed.sh`, then adjust batch size and tensor parallelism for the available hardware.

## Evaluation

The evaluation script starts a local vLLM OpenAI-compatible server when needed, runs end-to-end audio inference on Test-ID and Test-OOD, and computes function-name accuracy, parameter-value precision/recall/F1, and overall exact accuracy by difficulty level.

```bash
export MODEL_PATH=/path/to/base-or-trained-checkpoint
export VLLM_HOST=127.0.0.1
export VLLM_PORT=12355
export VLLM_MODEL_NAME=Qwen2.5-Omni-7B

source project/scripts/setup_env.sh
bash project/eval/eval_all.sh
```

Results are saved under `project/eval/results/` unless `EVAL_OUTPUT_DIR` is overridden.

You can also run the metric implementation directly on an existing inference file:

```bash
python project/eval/metrics_multilevel.py \
  project/eval/results/sfc-eval_test_id.jsonl \
  --format sfc \
  --output_file project/eval/results/sfc-eval_test_id_analyse.jsonl
```

## Citation

If you use SFC-Bench, SpokenFC-7B, or this codebase, please cite:

```bibtex
@article{peng2026spoken,
  title={Spoken Function Calling: A New Perspective on Spoken Language Understanding for Large Audio Language Models},
  author={Peng, Yuezhang and Liu, Yuxin and Gao, Changfeng and Gao, Zhifu and Li, Xiangang and Chen, Xie},
  journal={arXiv preprint arXiv:2608.05126},
  year={2026}
}
```

## License

Except for third-party components that retain their original licenses, the original materials in this repository are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE). In particular, the vendored [`ms-swift/`](ms-swift/) source remains subject to its own license and notices.

## Acknowledgements

This project builds on [ms-swift](https://github.com/modelscope/ms-swift), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and Qwen2.5-Omni. We thank the maintainers and contributors of these projects.

