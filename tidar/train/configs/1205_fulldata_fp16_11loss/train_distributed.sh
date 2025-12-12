#!/bin/bash

# TiDAR Training Script

set -e

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho

CONFIG_FILE="tidar/train/configs/1205_fulldata_fp16_11loss/train_config.json"
DS_CONFIG="tidar/train/configs/1205_fulldata_fp16_11loss/ds_config_distributed_bf16.json"

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         tidar/train/train_tidar_deepspeed.py \
            --deepspeed \
            --deepspeed_config "${DS_CONFIG}" \
            --config_path "${CONFIG_FILE}"

echo "Training completed!"
