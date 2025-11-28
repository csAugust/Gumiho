#!/bin/bash
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
cd ./gumiho/train

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho

# Configuration file path
CONFIG_FILE="/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/scripts/train_config.json"

# Run training with single config path parameter
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main_deepspeed.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --config_path "${CONFIG_FILE}"
