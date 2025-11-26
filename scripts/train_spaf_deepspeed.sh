#!/bin/bash
# SpAF Training with DeepSpeed
# This script launches multi-GPU training for SpAF model

# Configuration
BASE_MODEL_PATH="meta-llama/Llama-2-7b-hf"  # Change this to your model path
TRAIN_DATA_PATH="./train_data"  # Change this to your training data path
EVAL_DATA_PATH="./eval_data"  # Optional: evaluation data path
CHECKPOINT_DIR="./spaf_checkpoints"
DEEPSPEED_CONFIG="gumiho/train/ds_config_spaf.json"

# Training hyperparameters
NUM_EPOCHS=3
MAX_SEQ_LENGTH=512
CUTOFF_LAYER=16  # For 7B model (32 layers), use 16; for 13B, use 20; for 70B, use 40
ADAPTER_DIM_RATIO=0.25
ALIGNMENT_WEIGHT=1.0
ADAPTER_WEIGHT=1.0
ALIGNMENT_LOSS_TYPE="mse"

# Optimization
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01

# Logging
LOGGING_STEPS=100
SAVE_STEPS=1000
EVAL_STEPS=500

# Weights & Biases (optional)
USE_WANDB=false
WANDB_PROJECT="spaf_training"
WANDB_RUN_NAME="spaf_llama2_7b"

# Number of GPUs
NUM_GPUS=2  # Change this based on your setup

# Build command
CMD="deepspeed --num_gpus=${NUM_GPUS} gumiho/train/train_spaf_deepspeed.py \
    --deepspeed_config ${DEEPSPEED_CONFIG} \
    --base_model_path ${BASE_MODEL_PATH} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --cutoff_layer ${CUTOFF_LAYER} \
    --adapter_dim_ratio ${ADAPTER_DIM_RATIO} \
    --alignment_weight ${ALIGNMENT_WEIGHT} \
    --adapter_weight ${ADAPTER_WEIGHT} \
    --alignment_loss_type ${ALIGNMENT_LOSS_TYPE} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS}"

# Add eval data if provided
if [ -n "$EVAL_DATA_PATH" ]; then
    CMD="${CMD} --eval_data_path ${EVAL_DATA_PATH}"
fi

# Add wandb if enabled
if [ "$USE_WANDB" = true ]; then
    CMD="${CMD} --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN_NAME}"
fi

# Print command
echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD
