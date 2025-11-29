#!/bin/bash

# TiDAR Training Script

set -e

# Default values
NUM_GPUS=8
CONFIG_PATH="tidar/train_config.json"
DS_CONFIG="tidar/ds_config.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --ds-config)
            DS_CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-gpus N         Number of GPUs to use (default: 8)"
            echo "  --config-path PATH   Path to training config (default: tidar/train_config.json)"
            echo "  --ds-config PATH     Path to DeepSpeed config (default: tidar/ds_config.json)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default settings"
            echo "  $0 --num-gpus 4                       # Train with 4 GPUs"
            echo "  $0 --config-path my_config.json       # Use custom config"
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
echo "=========================================="
echo "TiDAR Training Configuration"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config path: $CONFIG_PATH"
echo "DeepSpeed config: $DS_CONFIG"
echo "=========================================="
echo ""

# Check if config files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$DS_CONFIG" ]; then
    echo "Error: DeepSpeed config file not found: $DS_CONFIG"
    exit 1
fi

# Launch training with DeepSpeed
echo "Starting training..."
echo ""

deepspeed --num_gpus=$NUM_GPUS tidar/train_tidar_deepspeed.py \
    --config_path "$CONFIG_PATH" \
    --deepspeed_config "$DS_CONFIG"

echo ""
echo "Training completed!"
