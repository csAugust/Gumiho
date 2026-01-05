#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho

# Configuration file path
CONFIG_FILE="gumiho/train/configs/base/train_config.json"
DS_CONFIG="gumiho/train/configs/base/ds_config.json"

# Run training with single config path parameter
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed gumiho/train/main_deepspeed.py \
    --deepspeed \
    --deepspeed_config "${DS_CONFIG}" \
    --config_path "${CONFIG_FILE}"

echo "Training completed!"