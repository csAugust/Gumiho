#!/bin/bash

# TiDAR Training Script

set -e

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         tidar/train/train_tidar_deepspeed.py \
           --deepspeed \
           --deepspeed_config "tidar/train/ds_config_distributed.json" \
           --config_path "tidar/train/train_config.json"

echo "Training completed!"
