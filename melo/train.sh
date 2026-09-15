#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
MODEL_NAME=$(basename "$(dirname "$CONFIG")")
PORT=10902

# Prevent PCIe P2P bus deadlocks on consumer/workstation motherboards
export NCCL_P2P_DISABLE=1
export RCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Memory allocation & architecture targets
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
export MIOPEN_FIND_MODE=FAST

torchrun --nproc_per_node="$GPUS" \
         --master_port="$PORT" \
         train.py --c "$CONFIG" --model "$MODEL_NAME"