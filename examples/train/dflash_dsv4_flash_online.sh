#!/bin/bash
# Online DFlash Training Script for DeepSeek-V4-Flash (Sliding Window + Muon)
#
# Runs the full online DFlash training pipeline: data preparation, vLLM server launch,
# and training (with hidden states generated on-the-fly from the live server).
#
# This script uses two separate virtual environments:
#   - speculators venv: for data preparation and training
#   - vLLM venv: for the vLLM server (launch_vllm.py calls os.execvp with sys.executable)
#
# Usage:
#   Test run (100 samples):  MODE=test bash examples/train/dflash_dsv4_flash_online.sh
#   Full run:                MODE=full bash examples/train/dflash_dsv4_flash_online.sh

set -euo pipefail

# ============ Configuration ============
MODEL="deepseek-ai/DeepSeek-V4-Flash"
BASE_DIR="/mnt/data/engine/rahul-tuli/dsv4-dflash-training-v2"
DATA_DIR="$BASE_DIR/data"
CHECKPOINT_DIR="$BASE_DIR/checkpoints"
HIDDEN_STATES_DIR="$BASE_DIR/hidden_states"
VLLM_PORT=8000
SEQ_LENGTH=8192
EPOCHS=5
LR=6e-4
CHECKPOINT_FREQ=0.2

# DFlash-specific parameters
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="3 13 23 32 42"

# Sliding window attention — alternate layers use sliding window
SLIDING_WINDOW=2048
SLIDING_WINDOW_INDICES="0 2 4"

# Optimizer — Muon for 2D weight matrices, AdamW for the rest
OPTIMIZER="muon"
MUON_LR=0.02

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1,2,3"
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4

# Virtual environments
SPECULATORS_VENV="/home/rahul-tuli/speculators/.venv"
VLLM_VENV="/home/rahul-tuli/vllm/.venv"

# Datasets (sharegpt format)
ULTRACHAT="/mnt/data/engine/hezhao/ultrachat_DeepSeek-V4-Flash_20260502_043453.jsonl"
MAGPIE="/mnt/data/engine/hezhao/magpie_DeepSeek-V4-Flash_20260502_043453.jsonl"

# Mode: "test" for 100-sample validation, "full" for complete training run
MODE="${MODE:-test}"
# =======================================

# Mode-specific settings
if [ "$MODE" = "test" ]; then
    MAX_SAMPLES_FLAG="--max-samples 100"
    LOGGER_FLAG="--logger trackio"
    echo "=== Running in TEST mode (100 samples, no experiment tracking) ==="
elif [ "$MODE" = "full" ]; then
    MAX_SAMPLES_FLAG=""
    LOGGER_FLAG="--logger trackio"
    echo "=== Running in FULL mode (all samples, trackio enabled) ==="
else
    echo "ERROR: MODE must be 'test' or 'full', got '$MODE'"
    exit 1
fi

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$HIDDEN_STATES_DIR"

# Step 1: Prepare data (speculators venv)
echo "=== Step 1: Preparing data ==="
source "$SPECULATORS_VENV/bin/activate"
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$ULTRACHAT" \
    --data "$MAGPIE" \
    --output "$DATA_DIR" \
    $MAX_SAMPLES_FLAG \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background (vLLM venv)
echo "=== Step 2: Launching vLLM server ==="
source "$VLLM_VENV/bin/activate"
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --data-parallel-size 4 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train against the live vLLM server (speculators venv)
echo "=== Step 3: Training ==="
source "$SPECULATORS_VENV/bin/activate"

run_training() {
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
        --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
        scripts/train.py \
        --verifier-name-or-path "$MODEL" \
        --data-path "$DATA_DIR" \
        --hidden-states-path "$HIDDEN_STATES_DIR" \
        --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
        --save-path "$CHECKPOINT_DIR" \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --scheduler-type cosine \
        --total-seq-len "$SEQ_LENGTH" \
        --speculator-type "$SPECULATOR_TYPE" \
        --block-size "$BLOCK_SIZE" \
        --max-anchors "$MAX_ANCHORS" \
        --num-layers "$NUM_LAYERS" \
        --target-layer-ids $TARGET_LAYER_IDS \
        --sliding-window "$SLIDING_WINDOW" \
        --sliding-window-indices $SLIDING_WINDOW_INDICES \
        --optimizer "$OPTIMIZER" \
        --muon-lr "$MUON_LR" \
        --checkpoint-freq "$CHECKPOINT_FREQ" \
        --on-missing generate \
        --on-generate delete \
        $LOGGER_FLAG \
        --run-name "dflash-dsv4-flash-online-test"
}

if [ "$MODE" = "full" ]; then
    run_training 2>&1 | tee "$BASE_DIR/train.log"
else
    run_training
fi

echo "Done. Checkpoints saved to $CHECKPOINT_DIR/"
