#!/bin/bash

# TiDAR Training Script

set -e

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho

# Default values
NUM_GPUS=8
CONFIG_FILE="tidar/train/train_config.json"
DS_CONFIG="tidar/train/ds_config.json"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus=$NUM_GPUS tidar/train/train_tidar_deepspeed.py \
    --deepspeed \
    --deepspeed_config "${DS_CONFIG}" \
    --config_path "${CONFIG_FILE}"

echo "Training completed!"
