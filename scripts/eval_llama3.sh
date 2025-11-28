#!/bin/bash
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho

# Configuration file path
CONFIG_FILE="/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/scripts/eval_config.json"

# Run evaluation with configuration
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m gumiho.evaluation.gen_gumiho_answer_llama3chat \
  --config_path "$CONFIG_FILE"


echo "Evaluation completed!"
