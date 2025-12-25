#!/bin/bash

set -e  # Exit on error

export PYTHONPATH=$PYTHONPATH:/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho
# ls /mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/train_data/Qwen2.5-1.5B-Instruct/ultrachat_200k/0_207860_mufp16
# ls /mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/train_data/Qwen2.5-1.5B-Instruct/ShareGPT_V4.3_unfiltered_cleaned_split.json/0_68200_mufp16

# Default parameters
OUTDIR="/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/train_data/1212"
START=0
# END=10
# END=68200
END=207800
NUM_PROCESSES=4
GPUS="0,1|2,3|4,5|6,7"
MODEL_PATH="/mnt/bos-text/models/hf_models/Qwen2.5-1.5B-Instruct"
# MODEL_PATH="/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct"
DATASET_PATH="/mnt/user-ssd/chenzhiyang1/workspace/Datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json"
# DATASET_PATH="/mnt/user-ssd/chenzhiyang1/workspace/Datasets/ultrachat_200k"
MAX_LENGTH=4096
SYSTEM_PROMPT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --num-processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --system-prompt)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "Starting data generation with the following parameters:"
echo "  Output directory: $OUTDIR"
echo "  Data range: $START to $END"
echo "  Number of processes: $NUM_PROCESSES"
echo "  GPU assignments: $GPUS"
echo "  Model path: $MODEL_PATH"
echo "  Dataset path: $DATASET_PATH"
echo "  Max length: $MAX_LENGTH"
if [ -n "$SYSTEM_PROMPT" ]; then
    echo "  System prompt: (custom)"
else
    echo "  System prompt: (default)"
fi
echo ""

# Build the command
CMD="python3 -m tidar.data.ge_data.allocation \
    --outdir \"$OUTDIR\" \
    --start \"$START\" \
    --end \"$END\" \
    --num_processes \"$NUM_PROCESSES\" \
    --gpus \"$GPUS\" \
    --model_path \"$MODEL_PATH\" \
    --dataset_path \"$DATASET_PATH\" \
    --max_length \"$MAX_LENGTH\""

# Add system_prompt if provided
if [ -n "$SYSTEM_PROMPT" ]; then
    CMD="$CMD --system_prompt \"$SYSTEM_PROMPT\""
fi

# Run the data generation script
eval $CMD

echo ""
echo "Data generation completed!"
echo "Output directory: $OUTDIR"
