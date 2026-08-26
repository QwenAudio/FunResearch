#!/bin/bash
# GRPO training with exact-match reward (external_r1v_acc).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../scripts/setup_env.sh"

: "${MODEL_PATH:?Please set MODEL_PATH before training (export MODEL_PATH=/path/to/Qwen2.5-Omni-7B)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

cd "${MS_SWIFT_ROOT}"

swift rlhf \
    --rlhf_type grpo \
    --loss_scale last_round \
    --model "${MODEL_PATH}" \
    --external_plugins "${SFC_REWARD_PLUGIN}" \
    --reward_funcs external_r1v_acc \
    --load_from_cache_file false \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA_OOD}" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 2048 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 3 \
    --eval_steps 400 \
    --save_steps 400 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "${OUTPUT_ROOT}/grpo_em" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --num_iterations 1 \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --beta 0.001 \
    --deepspeed zero2 \
    --log_completions true \
    --async_generate false \
    --sleep_level 0 \
    --vllm_tensor_parallel_size 4 \
    --report_to "${REPORT_TO}"
