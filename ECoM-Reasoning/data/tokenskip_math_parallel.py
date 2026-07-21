### Compress question / solution text by ratio, with multi-GPU parallelism and five compression modes:
###   - llmlingua2        : LLMLingua-2 (BERT-style token classification)
###   - llmlingua1        : LLMLingua-1 (LLM scoring, Llama-2-7b-hf)
###   - llmlingua1_qwen   : LLMLingua-1 (LLM scoring, Qwen2.5-1.5B)
###   - longllmlingua     : LongLLMLingua (LLM scoring + long-context params, Llama-2-7b-hf)
###   - longllmlingua_qwen: LongLLMLingua (LLM scoring + long-context params, Qwen2.5-1.5B)

import os
import json
import copy
import multiprocessing as mp
from tqdm import tqdm
from llmlingua import PromptCompressor


# ---------------- Config ----------------
NUM_GPUS = 4
TMP_DIR = "/tmp/compress_ratio_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# Compression modes: "llmlingua2" | "llmlingua1" | "llmlingua1_qwen" | "longllmlingua" | "longllmlingua_qwen"
# Multiple modes run sequentially, each loading its own weights and writing a separate file.
MODES = [
    "llmlingua2"
    # "llmlingua1",
    # "llmlingua1_qwen",
    # "longllmlingua",
    # "longllmlingua_qwen",
]

# Per-mode scorer/compressor weights (override via env vars, or edit the placeholder defaults)
MODEL_PATHS = {
    "llmlingua2":         os.environ.get("LLMLINGUA2_PATH", "checkpoints/llmlingua-2-xlm-roberta-large-meetingbank"),
    "llmlingua1":         os.environ.get("LLAMA2_7B_PATH",  "/path/to/Llama-2-7b-hf"),
    "llmlingua1_qwen":    os.environ.get("QWEN25_1_5B_PATH", "checkpoints/Qwen2.5-1.5B"),
    "longllmlingua":      os.environ.get("LLAMA2_7B_PATH",  "/path/to/Llama-2-7b-hf"),
    "longllmlingua_qwen": os.environ.get("QWEN25_1_5B_PATH", "checkpoints/Qwen2.5-1.5B"),
}

for _m in MODES:
    assert _m in MODEL_PATHS, f"Unknown MODE: {_m}"


# ---------------- IO ----------------
def load_jsonl(file, encoding="utf-8"):
    data = []
    with open(file, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------- Compressor ----------------
def build_compressor(llmlingua_path, mode, device="cuda"):
    """Build the compressor; enable the llmlingua-2 backend based on mode."""
    llm_lingua = PromptCompressor(
        model_name=llmlingua_path,
        use_llmlingua2=(mode == "llmlingua2"),
        device_map=device,
    )
    return llm_lingua


def compress_text(llm_lingua, text, compression_ratio, mode, question=""):
    """
    Compress a single text segment by ratio.
    - llmlingua2     : BERT-style token-classification compression
    - llmlingua1 / llmlingua1_qwen      : LLM-scoring compression; needs concate_question=False
    - longllmlingua / longllmlingua_qwen: LLM scoring + long-context params; conditional when a question is given

    Robustness: on a per-sample failure (e.g. LLMLingua's short-text IndexError), fall back to
    "compress everything" (empty string) so a worker never crashes, and so compressed length stays
    monotonic across rates (returning the original would make low rates paradoxically longer).
    """
    if not text or not text.strip():
        return ""

    try:
        if mode == "llmlingua2":
            res = llm_lingua.compress_prompt(text, rate=compression_ratio)
        elif mode in ("llmlingua1", "llmlingua1_qwen"):
            res = llm_lingua.compress_prompt(
                text,
                rate=compression_ratio,
                concate_question=False,
            )
        elif mode in ("longllmlingua", "longllmlingua_qwen"):
            # LongLLMLingua requires a question; when compressing the question itself (unconditional),
            # fall back to LLMLingua-1 unconditional LLM-scoring compression.
            if question and question.strip():
                # LongLLMLingua recommended params (from the README Quick Start)
                res = llm_lingua.compress_prompt(
                    text,
                    question=question,
                    rate=compression_ratio,
                    condition_in_question="after_condition",
                    reorder_context="sort",
                    dynamic_context_compression_ratio=0.3,
                    condition_compare=True,
                    context_budget="+100",
                    rank_method="longllmlingua",
                    concate_question=False,
                )
            else:
                res = llm_lingua.compress_prompt(
                    text,
                    rate=compression_ratio,
                    concate_question=False,
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except (IndexError, RuntimeError, AssertionError) as e:
        # LLMLingua may raise IndexError on short text / extreme rates;
        # fall back to "compress everything" (empty string) to keep ratio semantics monotonic.
        print(
                f"[WARN] compress fail (mode={mode}, rate={compression_ratio}, "
                f"len={len(text)}): {type(e).__name__}: {e}; fallback to empty (full compress).",
            flush=True,
        )
        return ""

    if isinstance(res, dict):
        return res.get("compressed_prompt", "")
    return res


# ---------------- worker ----------------
def worker(
    gpu_id: int,
    num_gpus: int,
    in_path: str,
    tmp_out: str,
    llmlingua_path: str,
    ratio_list: list,
    mode: str,
):
    """Single-GPU worker: handles rows where idx % num_gpus == gpu_id."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm_lingua = build_compressor(llmlingua_path, mode, device="cuda")

    data = load_jsonl(in_path)

    with open(tmp_out, "w", encoding="utf-8") as w:
        for idx, item in enumerate(tqdm(data, desc=f"[{mode}] gpu{gpu_id}", position=gpu_id)):
            if idx % num_gpus != gpu_id:
                continue

            item = copy.deepcopy(item)
            q = item.get("question", "")
            a = item.get("solution", "")

            for compression_ratio in ratio_list:
                # the question itself has no condition, so leave question= empty
                compressed_q = compress_text(llm_lingua, q, compression_ratio, mode, question="")
                # solution is conditioned on the question (only LongLLMLingua actually uses it)
                compressed_a = compress_text(llm_lingua, a, compression_ratio, mode, question=q)

                item[f"question_compress_{compression_ratio}"] = compressed_q
                item[f"solution_compress_{compression_ratio}"] = compressed_a

            w.write(json.dumps({"idx": idx, "obj": item}, ensure_ascii=False) + "\n")


# ---------------- merge ----------------
def merge_tmp(total_len: int, tmp_files):
    out = [None] * total_len
    for p in tmp_files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[rec["idx"]] = rec["obj"]
    missing = sum(x is None for x in out)
    if missing:
        raise RuntimeError(f"merge failed, missing {missing} items")
    return out


# ---------------- Run one mode ----------------
def run_one_mode(mode: str, input_path: str, output_dir: str, ratio_list: list, total_len: int):
    """Launch NUM_GPUS workers for one mode, compress in parallel, and merge outputs."""
    llmlingua_path = MODEL_PATHS[mode]
    print(f"\n========== [MODE] {mode} | [model] {llmlingua_path} ==========", flush=True)

    tmp_files = [
        os.path.join(TMP_DIR, f"compress_ratio.{mode}.gpu{i}.jsonl")
        for i in range(NUM_GPUS)
    ]
    for p in tmp_files:
        if os.path.exists(p):
            os.remove(p)

    procs = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(
            target=worker,
            args=(gpu_id, NUM_GPUS, input_path, tmp_files[gpu_id], llmlingua_path, ratio_list, mode),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"[{mode}] worker gpu{procs.index(p)} failed, exitcode={p.exitcode}")

    merged = merge_tmp(total_len, tmp_files)

    base = os.path.basename(input_path).rsplit(".", 1)[0]
    output_path = os.path.join(output_dir, f"{base}_compress_all_ratios_{mode}.jsonl")
    save_jsonl(merged, output_path)
    print(f"[{mode}] Saved -> {output_path}, rows: {len(merged)}", flush=True)


# ---------------- main ----------------
def main():
    # Directory holding train_3k.jsonl; compressed outputs go to <input_dir>/Compression.
    input_dir = os.environ.get("MATH_DATASET_DIR", "/path/to/Math-Dataset")
    input_path = os.path.join(input_dir, "train_3k.jsonl")
    output_dir = os.path.join(input_dir, "Compression")

    ratio_list = [0.4] # [0.2, 0.4, 0.6, 0.8]

    mp.set_start_method("spawn", force=True)

    data = load_jsonl(input_path)
    total_len = len(data)
    print(f"Loaded {total_len} rows from {input_path}")
    print(f"Will run modes in sequence: {MODES}")

    for mode in MODES:
        try:
            run_one_mode(mode, input_path, output_dir, ratio_list, total_len)
        except Exception as e:
            # a single mode crashing must not stop the remaining modes
            print(f"[ERROR] mode={mode} failed: {type(e).__name__}: {e}", flush=True)

    print("\nAll modes done.")


if __name__ == "__main__":
    main()
