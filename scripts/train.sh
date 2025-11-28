#!/bin/bash
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
cd ./gumiho/train

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho
logger_name="wandb"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main_deepspeed.py --deepspeed --deepspeed_config ds_config.json \
    --tmpdir "/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/train_data" \
    --cpdir "./ckpts-cloudml" \
    --configpath "/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/gumiho/train/Gumiho-LLaMA3-Instruct-8B.json" \
    --basepath "/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct" \
    --logger_file ${logger_name} \
    --p_w 0.1 \
    --mlp_v_w 1.0 \
    --mlp_p_w 100 \
    --max_len 2048 \
    --model_name l3_8b \
    --start_epoch 0 
