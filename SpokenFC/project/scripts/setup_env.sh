#!/bin/bash
# Source this file before training or evaluation:
#   source project/scripts/setup_env.sh

# Repository root (auto-detect from this script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Data paths
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export RAW_DATA_DIR="${DATA_ROOT}/raw/final_data_1228"
export AUDIO_DATA_DIR="${DATA_ROOT}/audio/final_data_audio"
export MSSWIFT_DATA_DIR="${DATA_ROOT}/processed/data_msswift"
export SLU_DATA_DIR="${DATA_ROOT}/processed/data_slu"

# Training datasets (SFC / Function Call)
export TRAIN_DATA="${TRAIN_DATA:-${MSSWIFT_DATA_DIR}/train_id.json}"
export VAL_DATA_ID="${VAL_DATA_ID:-${MSSWIFT_DATA_DIR}/test_id.json}"
export VAL_DATA_OOD="${VAL_DATA_OOD:-${MSSWIFT_DATA_DIR}/test_ood.json}"

# SLU datasets
export SLU_TRAIN_DATA="${SLU_TRAIN_DATA:-${SLU_DATA_DIR}/train_id.json}"
export SLU_VAL_DATA_ID="${SLU_VAL_DATA_ID:-${SLU_DATA_DIR}/test_id.json}"
export SLU_VAL_DATA_OOD="${SLU_VAL_DATA_OOD:-${SLU_DATA_DIR}/test_ood.json}"

# ms-swift framework (override to use your own installation)
export MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-${REPO_ROOT}/ms-swift}"
export SFC_REWARD_PLUGIN="${REPO_ROOT}/project/plugins/sfc_reward.py"

# Model path (set before training/eval)
export MODEL_PATH="${MODEL_PATH:-}"

# Output directories
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/project/eval/results}"

# Qwen2.5-Omni multimodal env vars
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT:-0}"
