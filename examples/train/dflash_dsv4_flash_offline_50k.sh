#!/bin/bash
# Offline DFlash Training Script for DeepSeek-V4-Flash (Sliding Window + Muon)
#
# Trains a DFlash speculator using pre-generated hidden states (no vLLM server needed).
# Hidden states must already exist at HIDDEN_STATES_DIR as hs_<idx>.safetensors files.
#
# Usage:
#   bash examples/train/dflash_dsv4_flash_offline_50k.sh

set -euo pipefail

# ============ Configuration ============
MODEL="deepseek-ai/DeepSeek-V4-Flash"
BASE_DIR="/mnt/data/engine/rahul-tuli/dsv4-dflash-training"
DATA_DIR="$BASE_DIR/data-50k"
CHECKPOINT_DIR="$BASE_DIR/checkpoints-swa-muon"
HIDDEN_STATES_DIR="$BASE_DIR/hidden_states"
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

# GPU assignments (all GPUs available for training — no vLLM server needed)
TRAIN_GPUS="0,1,2,3,4,5,6,7"
NUM_TRAIN_GPUS=8

# Virtual environment
SPECULATORS_VENV="/home/rahul-tuli/speculators/.venv"

# Logging
LOGGER_FLAG="--logger trackio"
# =======================================

source "$SPECULATORS_VENV/bin/activate"

echo "=== Offline DFlash Training for DeepSeek-V4-Flash ==="
echo "  Data:           $DATA_DIR"
echo "  Hidden states:  $HIDDEN_STATES_DIR"
echo "  Checkpoints:    $CHECKPOINT_DIR"
echo "  GPUs:           $TRAIN_GPUS ($NUM_TRAIN_GPUS GPUs)"
echo "  Optimizer:      $OPTIMIZER (muon_lr=$MUON_LR, adamw_lr=$LR)"
echo "  Sliding window: size=$SLIDING_WINDOW, layers=$SLIDING_WINDOW_INDICES"

mkdir -p "$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
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
    --on-missing skip \
    $LOGGER_FLAG 2>&1 | tee "$BASE_DIR/train-swa-muon.log"

echo "Done. Checkpoints saved to $CHECKPOINT_DIR/"
