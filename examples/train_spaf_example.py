#!/usr/bin/env python3
# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example training script for SpAF model.

This script demonstrates how to train the SpAF components (Adapters + AlignmentHead)
on a base LLM.
"""

import os
import json
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gumiho.model.spaf_model import SpAFModel
from gumiho.train.train_spaf import (
    SpAFDataset,
    setup_spaf_training,
    train_spaf_epoch,
    evaluate_spaf
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_dataset(dataset_name='wikitext', split='train', max_samples=10000):
    """
    Prepare training dataset.
    
    Args:
        dataset_name: Name of dataset to load.
        split: Dataset split ('train', 'validation', 'test').
        max_samples: Maximum number of samples to use.
        
    Returns:
        Tokenized dataset.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Limit samples
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    logger.info(f"Dataset size: {len(dataset)}")
    return dataset


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """
    Tokenize dataset.
    
    Args:
        dataset: HuggingFace dataset.
        tokenizer: Tokenizer to use.
        max_length: Maximum sequence length.
        
    Returns:
        List of tokenized examples.
    """
    logger.info("Tokenizing dataset...")
    
    tokenized_data = []
    for example in dataset:
        # Get text field (adjust based on your dataset)
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
    
    logger.info(f"Tokenized {len(tokenized_data)} examples")
    return tokenized_data


def main():
    # Configuration
    config_path = 'gumiho/train/spaf_config.json'
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override with your settings
    config['base_model_path'] = 'meta-llama/Llama-2-7b-hf'  # Change this
    config['spaf_settings']['cutoff_layer'] = 16  # For 7B model (32 layers)
    config['training']['batch_size'] = 4
    config['training']['num_epochs'] = 1
    config['output']['output_dir'] = './spaf_checkpoints'
    
    # Create output directory
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Initialize SpAF model
    logger.info("Initializing SpAF model...")
    model = SpAFModel.from_pretrained(
        base_model_path=config['base_model_path'],
        spaf_config=config['spaf_settings'],
        torch_dtype=torch.float16 if config['training']['fp16'] else torch.float32,
        device_map='auto',
    )
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_raw = prepare_dataset('wikitext', 'train', max_samples=1000)
    eval_raw = prepare_dataset('wikitext', 'validation', max_samples=100)
    
    train_tokenized = tokenize_dataset(train_raw, tokenizer, config['training']['max_seq_length'])
    eval_tokenized = tokenize_dataset(eval_raw, tokenizer, config['training']['max_seq_length'])
    
    # Create datasets and dataloaders
    train_dataset = SpAFDataset(train_tokenized, config['training']['max_seq_length'])
    eval_dataset = SpAFDataset(eval_tokenized, config['training']['max_seq_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
    )
    
    # Setup training
    logger.info("Setting up training...")
    optimizer, loss_fn = setup_spaf_training(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        loss_config=config['training']['loss_weights'],
    )
    
    # Training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    best_eval_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*50}")
        
        # Train
        train_losses = train_spaf_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            log_interval=config['training']['logging_steps'],
        )
        
        logger.info(f"\nTrain Losses:")
        logger.info(f"  Total: {train_losses['total']:.4f}")
        logger.info(f"  Adapter: {train_losses['adapter_avg']:.4f}")
        logger.info(f"  Alignment: {train_losses['alignment']:.4f}")
        
        # Evaluate
        if (epoch + 1) % max(1, config['training']['num_epochs'] // 5) == 0:
            logger.info("\nEvaluating...")
            eval_losses = evaluate_spaf(
                model=model,
                dataloader=eval_loader,
                loss_fn=loss_fn,
                device=device,
            )
            
            logger.info(f"\nEval Losses:")
            logger.info(f"  Total: {eval_losses['total']:.4f}")
            logger.info(f"  Adapter: {eval_losses['adapter_avg']:.4f}")
            logger.info(f"  Alignment: {eval_losses['alignment']:.4f}")
            
            # Save best model
            if eval_losses['total'] < best_eval_loss:
                best_eval_loss = eval_losses['total']
                save_path = f"{config['output']['output_dir']}/best_model"
                logger.info(f"\nSaving best model to {save_path}")
                model.save_spaf_components(save_path)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_steps'] == 0:
            save_path = f"{config['output']['output_dir']}/epoch_{epoch+1}"
            logger.info(f"\nSaving checkpoint to {save_path}")
            model.save_spaf_components(save_path)
    
    logger.info("\n" + "="*50)
    logger.info("Training completed!")
    logger.info("="*50)


if __name__ == '__main__':
    main()
