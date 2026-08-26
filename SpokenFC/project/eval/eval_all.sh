#!/bin/bash
# End-to-end SFC evaluation: inference on test_id / test_ood, then metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../scripts/setup_env.sh"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH before running eval}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-12355}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen2.5-Omni-7B}"

EVAL_SETTING="${EVAL_SETTING:-sfc-eval}"
mkdir -p "${EVAL_OUTPUT_DIR}"

INFER_SCRIPT="${SCRIPT_DIR}/infer.py"
METRICS_SCRIPT="${SCRIPT_DIR}/metrics_multilevel.py"

INPUT_ID="${MSSWIFT_DATA_DIR}/test_id.json"
INPUT_OOD="${MSSWIFT_DATA_DIR}/test_ood.json"
OUTPUT_ID="${EVAL_OUTPUT_DIR}/${EVAL_SETTING}_test_id.jsonl"
OUTPUT_OOD="${EVAL_OUTPUT_DIR}/${EVAL_SETTING}_test_ood.jsonl"
ANALYSIS_ID="${EVAL_OUTPUT_DIR}/${EVAL_SETTING}_test_id_analyse.jsonl"
ANALYSIS_OOD="${EVAL_OUTPUT_DIR}/${EVAL_SETTING}_test_ood_analyse.jsonl"

echo "=== Spoken Function Calling Evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Output: ${EVAL_OUTPUT_DIR}"

# Step 1: Start vLLM server in background (if not already running)
if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Starting vLLM server on port ${VLLM_PORT}..."
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        nohup vllm serve "${MODEL_PATH}" \
            --served-model-name "${VLLM_MODEL_NAME}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --dtype bfloat16 \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.9 \
            > "${EVAL_OUTPUT_DIR}/vllm.log" 2>&1 &
    VLLM_PID=$!
    echo "Waiting for vLLM to be ready (pid=${VLLM_PID})..."
    for _ in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo "vLLM is ready."
            break
        fi
        sleep 10
    done
fi

# Step 2: Inference
echo "Running inference on test_id..."
python "${INFER_SCRIPT}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --model-name "${VLLM_MODEL_NAME}" \
    --input_file "${INPUT_ID}" \
    --output_file "${OUTPUT_ID}" \
    --repo-root "${REPO_ROOT}"

echo "Running inference on test_ood..."
python "${INFER_SCRIPT}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --model-name "${VLLM_MODEL_NAME}" \
    --input_file "${INPUT_OOD}" \
    --output_file "${OUTPUT_OOD}" \
    --repo-root "${REPO_ROOT}"

# Step 3: Metrics
echo "Computing metrics (test_id)..."
python "${METRICS_SCRIPT}" \
    --input_file "${OUTPUT_ID}" \
    --output_file "${ANALYSIS_ID}" \
    --format sfc

echo "Computing metrics (test_ood)..."
python "${METRICS_SCRIPT}" \
    --input_file "${OUTPUT_OOD}" \
    --output_file "${ANALYSIS_OOD}" \
    --format sfc

echo "=== Evaluation complete ==="
echo "Results: ${ANALYSIS_ID}"
echo "Results: ${ANALYSIS_OOD}"
