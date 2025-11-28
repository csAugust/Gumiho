#!/bin/bash

set -e  # Exit on error

# Default parameters
OUTDIR="/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/train_data"
START=0
END=67999
NUM_PROCESSES=2
GPUS="0,1,2,3|4,5,6,7"
MODEL_PATH="/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct"
DATASET_PATH="/mnt/user-ssd/chenzhiyang1/workspace/Datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json"
MAX_LENGTH=2048
SYSTEM_PROMPT=""

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate training data for LLaMA3 model"
    echo ""
    echo "Options:"
    echo "  --outdir DIR          Output directory for training data (default: ./train_data)"
    echo "  --start N             Start index for data range (default: 0)"
    echo "  --end N               End index for data range (default: 67999)"
    echo "  --num-processes N     Number of parallel processes (default: 2)"
    echo "  --gpus GPUS           GPU assignments (comma-separated, pipe-separated for multiple processes)"
    echo "                        (default: '0,1,2,3|4,5,6,7')"
    echo "  --model-path PATH     Path to the model (default: /mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct)"
    echo "  --dataset-path PATH   Path to the dataset (default: /mnt/user-ssd/chenzhiyang1/workspace/Datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json)"
    echo "  --max-length N        Maximum sequence length (default: 2048)"
    echo "  --system-prompt TEXT  System prompt for the model (optional, uses default if not provided)"
    echo "  -h, --help            Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --outdir ./my_train_data --end 10000"
    echo "  $0 --num-processes 1 --gpus '0,1,2,3'"
    echo "  $0 --model-path /path/to/model --max-length 4096"
    echo ""
}

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
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
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
CMD="python3 -m gumiho.ge_data.allocation \
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
