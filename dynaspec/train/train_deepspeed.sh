#!/bin/bash
# Hybrid-AgileDrafter DeepSpeed 训练启动脚本

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8
CONFIG_PATH="dynaspec/train/train_config.json"
DS_CONFIG_PATH="dynaspec/train/ds_config_hybrid.json"

# 可选：从检查点恢复训练
# EXISTING_MODEL_PATH="./checkpoints/hybrid_agile_ds/epoch_10/pytorch_model.bin"

# 启动 DeepSpeed 训练
deepspeed --num_gpus=$NUM_GPUS \
    dynaspec/train/train_hybrid_deepspeed.py \
    --config_path $CONFIG_PATH \
    --deepspeed_config $DS_CONFIG_PATH \
    # --existing_model_path $EXISTING_MODEL_PATH  # 如需从检查点恢复，取消注释

echo "Training completed!"
