#!/usr/bin/env python3
# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example script showing how to prepare data and launch SpAF DeepSpeed training.

This script demonstrates:
1. How to prepare and tokenize training data
2. How to configure SpAF for multi-GPU training
3. How to launch training with DeepSpeed
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_and_save_data(
    dataset_name='wikitext',
    output_dir='./train_data',
    model_path='meta-llama/Llama-2-7b-hf',
    max_samples=10000,
    max_length=512,
):
    """
    Prepare training data and save to disk for DeepSpeed training.
    
    Args:
        dataset_name: Name of dataset to load.
        output_dir: Directory to save tokenized data.
        model_path: Path to base model for tokenizer.
        max_samples: Maximum number of samples.
        max_length: Maximum sequence length.
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    else:
        dataset = load_dataset(dataset_name, split='train')
    
    # Limit samples
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Tokenize and save
    os.makedirs(output_dir, exist_ok=True)
    
    tokenized_data = []
    for idx, example in enumerate(dataset):
        text = example.get('text', example.get('content', ''))
        
        if not text or len(text.strip()) == 0:
            continue
        
        # Tokenize
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        tokenized_data.append({
            'input_ids': tokens['input_ids'][0].tolist(),
            'attention_mask': tokens['attention_mask'][0].tolist(),
        })
        
        # Save in batches
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(dataset):
            batch_file = os.path.join(output_dir, f'batch_{idx // 1000}.pt')
            torch.save(tokenized_data, batch_file)
            print(f"Saved {len(tokenized_data)} samples to {batch_file}")
            tokenized_data = []
    
    print(f"Data preparation completed! Saved to {output_dir}")


def create_launch_script(
    base_model_path='meta-llama/Llama-2-7b-hf',
    train_data_path='./train_data',
    num_gpus=2,
    output_file='launch_spaf_training.sh',
):
    """
    Create a launch script for multi-GPU training.
    
    Args:
        base_model_path: Path to base model.
        train_data_path: Path to training data.
        num_gpus: Number of GPUs to use.
        output_file: Output script file name.
    """
    script_content = f'''#!/bin/bash
# Auto-generated SpAF training launch script

# Set these variables
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPUs
export OMP_NUM_THREADS=1

# DeepSpeed configuration
deepspeed --num_gpus={num_gpus} gumiho/train/train_spaf_deepspeed.py \\
    --deepspeed_config gumiho/train/ds_config_spaf.json \\
    --base_model_path {base_model_path} \\
    --train_data_path {train_data_path} \\
    --checkpoint_dir ./spaf_checkpoints \\
    --num_epochs 3 \\
    --max_seq_length 512 \\
    --cutoff_layer 16 \\
    --adapter_dim_ratio 0.25 \\
    --alignment_weight 1.0 \\
    --adapter_weight 1.0 \\
    --learning_rate 1e-4 \\
    --weight_decay 0.01 \\
    --logging_steps 100 \\
    --save_steps 1000
'''
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    os.chmod(output_file, 0o755)
    print(f"Created launch script: {output_file}")
    print(f"Run with: bash {output_file}")


def main():
    print("="*60)
    print("SpAF Multi-GPU Training Setup")
    print("="*60)
    
    # Step 1: Prepare data
    print("\nStep 1: Preparing training data...")
    prepare_and_save_data(
        dataset_name='wikitext',
        output_dir='./train_data',
        model_path='meta-llama/Llama-2-7b-hf',
        max_samples=1000,  # Small sample for testing
        max_length=512,
    )
    
    # Step 2: Create launch script
    print("\nStep 2: Creating launch script...")
    create_launch_script(
        base_model_path='meta-llama/Llama-2-7b-hf',
        train_data_path='./train_data',
        num_gpus=2,
        output_file='launch_spaf_training.sh',
    )
    
    print("\n" + "="*60)
    print("Setup completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review and edit launch_spaf_training.sh as needed")
    print("2. Make sure you have the base model downloaded")
    print("3. Run: bash launch_spaf_training.sh")
    print("\nFor distributed training across multiple nodes:")
    print("Use: deepspeed --hostfile=hostfile.txt gumiho/train/train_spaf_deepspeed.py ...")
    print("\nMonitoring:")
    print("- Logs will be saved to: ./spaf_checkpoints/training.log")
    print("- Checkpoints will be saved to: ./spaf_checkpoints/")
    print("- Use --use_wandb flag for Weights & Biases logging")


if __name__ == '__main__':
    main()
