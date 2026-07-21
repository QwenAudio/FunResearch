# Data

Data-construction script + training/validation manifests for ECoM Reasoning.

## 1. Compress the reasoning traces

`tokenskip_math_parallel.py` compresses the `question` / `solution` fields of a math-QA jsonl
at several keep-ratios (LLMLingua scorer, one worker per GPU), adding `question_compress_<r>` and
`solution_compress_<r>` keys to every row. Requires the patched LLMLingua (see the top-level README).

```bash
export MATH_DATASET_DIR=/path/to/Math-Dataset     # reads $MATH_DATASET_DIR/train.jsonl
export LLMLINGUA2_PATH=/path/to/llmlingua-2-xlm-roberta-large-meetingbank
# LLAMA2_7B_PATH / QWEN25_1_5B_PATH are also needed for the llmlingua1 / *_qwen modes
python data/tokenskip_math_parallel.py
```

Output: `$MATH_DATASET_DIR/Compression/train_compress_all_ratios_<mode>.jsonl`.

## 2. Point the manifests at your data

Replace the `/path/to/...` placeholders in `train_mix_dataset.yaml` and `test_math_dataset.yaml`
with your jsonl paths. Per-subset fields:

| field | meaning |
|-------|---------|
| `dataset_path` | jsonl file for this subset |
| `data_struct` | `sats` (spoken chat), `com` (chat + user text), `com-reason` / `sats-reason` (with reasoning) |
| `compress.stage` / `compress.rate` | which field is compressed, and which `*_compress_<rate>` key to use |
| `sampling_rate` | per-subset sampling weight |
| `system_prompt` | prompt prepended to each conversation |
