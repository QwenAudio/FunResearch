# ECoM-Reasoning

Efficient Chain-of-Modality (ECoM) reasoning for spoken language models: a text LLM
(Qwen2.5-1.5B) + a Whisper speech encoder + a CosyVoice speech codec, trained to *reason
before it speaks*. Reasoning traces are **compressed** (LLMLingua token-skipping) so the
model learns an efficient chain of modality instead of a verbose one.

Audio samples: open [`index.html`](index.html) (assets in [`docs/`](docs/)).

## Install

If you are using this project inside **FunResearch**, enter the project directory first:

```bash
cd ECoM-Reasoning
```

If you are using it as a standalone repository, clone it and initialise submodules:

```bash
git clone <repo-url> ECoM-Reasoning && cd ECoM-Reasoning
git submodule update --init --recursive
```

When this project is copied into a larger repository and is not itself a Git repository,
initialise the external dependencies directly:

```bash
git clone https://github.com/X-LANCE/SLAM-LLM.git SLAM-LLM
mkdir -p third_party
git clone https://github.com/microsoft/LLMLingua.git third_party/LLMLingua
```

Then set up the environment:

```bash
conda create -n ecom python=3.11 -y && conda activate ecom
pip install -e SLAM-LLM             # the slam_llm framework + core deps
pip install -r requirements.txt

# patched LLMLingua (needed by the data-construction script)
git -C third_party/LLMLingua apply ../../patches/llmlingua-dynamiccache.patch
pip install -e third_party/LLMLingua
```

`SLAM-LLM/` and `third_party/LLMLingua/` are external dependencies; if their remotes are unreachable,
initialise those directories from your own checkouts.

## Run

Run everything **from the repo root** with the framework on the path:

```bash
export PYTHONPATH=./SLAM-LLM/src:$PYTHONPATH
```

1. Put model weights under `checkpoints/` — see [`checkpoints/README.md`](checkpoints/README.md).
2. Build the compressed data and point the manifests — see [`data/README.md`](data/README.md).
3. Edit the paths at the top of each script, then launch:

```bash
bash src/scripts/finetune_com.sh     # train
bash src/scripts/inference_com.sh    # inference (streaming CosyVoice decoding)
```

## Acknowledgements

- [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM.git) — base speech-LLM framework (`slam_llm`).
- [LLMLingua](https://github.com/microsoft/LLMLingua) — prompt compression for efficient reasoning traces.
- CosyVoice — speech tokenizer / vocoder used for codec decoding.
