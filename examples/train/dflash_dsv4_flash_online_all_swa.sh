#!/bin/bash
# Online DFlash Training Script for DeepSeek-V4-Flash (All-SWA + Muon)
#
# Runs the full online DFlash training pipeline: data preparation, vLLM server
# launch, and training with hidden states generated on-the-fly from the live
# server (--on-missing generate). All 5 draft layers use sliding window
# attention (window=2048), matching vLLM's DFlash all-SWA inference support.
#
# Two virtual environments are used:
#   - speculators venv: data preparation and training
#   - vLLM venv:        the vLLM server (launch_vllm.py re-execs sys.executable)
#
# DeepSeek-V4 hidden-state generation can occasionally exceed vLLM's default
# per-request timeout (180s), so --request-timeout is raised (REQUEST_TIMEOUT).
#
# Usage:
#   Test run (100 samples):  MODE=test bash examples/train/dflash_dsv4_flash_online_all_swa.sh
#   Full run (all samples):  MODE=full bash examples/train/dflash_dsv4_flash_online_all_swa.sh

set -euo pipefail

# ============ Configuration ============
MODEL="deepseek-ai/DeepSeek-V4-Flash"
BASE_DIR="/mnt/data/engine/rahul-tuli/dsv4-dflash-training-online-all-swa"
DATA_DIR="$BASE_DIR/data"
CHECKPOINT_DIR="$BASE_DIR/checkpoints"
# Reuse the already-generated hidden states (cache hits skip vLLM extraction).
# With --on-generate delete, existing hs_*.safetensors are read-only cache hits
# (never deleted); only freshly-generated misses are removed after use, so disk
# stays bounded. The loader verifies token_ids match, so stale/misaligned files
# are regenerated rather than silently reused.
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/mnt/data/engine/rahul-tuli/dsv4-dflash-training/hidden_states}"
VLLM_PORT="${VLLM_PORT:-8200}"
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

# Sliding window attention — ALL 5 layers use sliding window
SLIDING_WINDOW=2048
SLIDING_WINDOW_INDICES="0 1 2 3 4"

# Optimizer — Muon for 2D weight matrices, AdamW for the rest
OPTIMIZER="muon"
MUON_LR=0.02

# vLLM hidden-state request timeout (seconds). DSv4 can be slow; default is 180.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
# Distributed (NCCL) collective timeout for the trainer. A rank blocked on a slow
# hidden-state request must not trip the 10-minute NCCL default and abort training.
export SPECULATORS_DIST_TIMEOUT_SEC="${SPECULATORS_DIST_TIMEOUT_SEC:-3600}"

# vLLM runs DP=4 (+ expert parallel) on 4 dedicated GPUs: 4 replicas load-balance
# extraction requests (4x throughput) with experts sharded via EP. Training gets
# the other 4 GPUs (disjoint — no compute contention with vLLM).
VLLM_GPUS="0,1,2,3"
VLLM_DATA_PARALLEL=4
VLLM_TENSOR_PARALLEL=1
VLLM_GPU_MEM_UTIL=0.9
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4

# Virtual environments
SPECULATORS_VENV="/home/rahul-tuli/speculators/.venv"
VLLM_VENV="/home/rahul-tuli/vllm/.venv"

# Datasets (sharegpt format)
ULTRACHAT="/mnt/data/engine/hezhao/ultrachat_DeepSeek-V4-Flash_20260502_043453.jsonl"
MAGPIE="/mnt/data/engine/hezhao/magpie_DeepSeek-V4-Flash_20260502_043453.jsonl"

# Mode: "test" for 100-sample validation, "full" for the complete training run
MODE="${MODE:-test}"
# =======================================

# Mode-specific settings
if [ "$MODE" = "test" ]; then
    MAX_SAMPLES_FLAG="--max-samples 100"
    RUN_NAME="dflash-dsv4-flash-online-all-swa-test"
    echo "=== Running in TEST mode (100 samples) ==="
elif [ "$MODE" = "full" ]; then
    MAX_SAMPLES_FLAG=""
    RUN_NAME="dflash-dsv4-flash-online-all-swa"
    echo "=== Running in FULL mode (all samples) ==="
else
    echo "ERROR: MODE must be 'test' or 'full', got '$MODE'"
    exit 1
fi

echo "  Sliding window: size=$SLIDING_WINDOW, layers=ALL ($SLIDING_WINDOW_INDICES)"
echo "  Optimizer:      $OPTIMIZER (muon_lr=$MUON_LR, adamw_lr=$LR)"
echo "  Request timeout: ${REQUEST_TIMEOUT}s"

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
# DP=4 x TP=2 (+ expert parallel) across all 8 GPUs: 4 replicas load-balance the
# extraction requests (4x throughput) while TP=2 shards each replica's weights/KV.
# gpu-memory-utilization is capped so the co-located trainer has headroom.
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --data-parallel-size "$VLLM_DATA_PARALLEL" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL" \
    --enable-expert-parallel \
    --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
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
        --request-timeout "$REQUEST_TIMEOUT" \
        --logger trackio \
        --run-name "$RUN_NAME"
}

run_training 2>&1 | tee "$BASE_DIR/train-${MODE}.log"

echo "Done. Checkpoints saved to $CHECKPOINT_DIR/"
