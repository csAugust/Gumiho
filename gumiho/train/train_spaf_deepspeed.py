# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
SpAF Training Script with DeepSpeed Support

This module implements multi-GPU training for SpAF model using DeepSpeed.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from loguru import logger
from typing import Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import config loader
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gumiho.train.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader()
spaf_training_config = config_loader.get_spaf_training_config()
paths_config = config_loader.get_paths()

# Parse arguments
parser = argparse.ArgumentParser(description='SpAF Training with DeepSpeed')

# Model paths
parser.add_argument('--base_model_path', type=str, required=True,
                    help='Path to base LLM model')
parser.add_argument('--spaf_config_path', type=str, 
                    default='gumiho/train/spaf_config.json',
                    help='Path to SpAF configuration file')
parser.add_argument('--checkpoint_dir', type=str, default='./spaf_checkpoints',
                    help='Directory to save checkpoints')

# Training parameters
parser.add_argument('--train_data_path', type=str, required=True,
                    help='Path to training data')
parser.add_argument('--eval_data_path', type=str, default=None,
                    help='Path to evaluation data')
parser.add_argument('--num_epochs', type=int, default=spaf_training_config.get('num_epochs', 3),
                    help='Number of training epochs')
parser.add_argument('--max_seq_length', type=int, default=spaf_training_config.get('max_seq_length', 512),
                    help='Maximum sequence length')
parser.add_argument('--batch_size', type=int, default=spaf_training_config.get('batch_size', 8),
                    help='Training batch size per GPU')

# SpAF parameters
parser.add_argument('--cutoff_layer', type=int, default=None,
                    help='Cutoff layer for SpAF (default: num_layers // 2)')
parser.add_argument('--adapter_dim_ratio', type=float, default=spaf_training_config.get('adapter_dim_ratio', 0.25),
                    help='Adapter dimension ratio')
parser.add_argument('--alignment_weight', type=float, default=spaf_training_config.get('alignment_weight', 1.0),
                    help='Weight for alignment loss')
parser.add_argument('--adapter_weight', type=float, default=spaf_training_config.get('adapter_weight', 1.0),
                    help='Weight for adapter loss')
parser.add_argument('--alignment_loss_type', type=str, default=spaf_training_config.get('alignment_loss_type', 'mse'),
                    choices=['mse', 'cosine'],
                    help='Type of alignment loss')

# Optimization
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='Weight decay')

# Logging and checkpointing
parser.add_argument('--logging_steps', type=int, default=spaf_training_config.get('logging_steps', 100),
                    help='Log every N steps')
parser.add_argument('--save_steps', type=int, default=spaf_training_config.get('save_steps', 1000),
                    help='Save checkpoint every N steps')
parser.add_argument('--eval_steps', type=int, default=spaf_training_config.get('eval_steps', 500),
                    help='Evaluate every N steps')
parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help='Path to checkpoint to resume from')

# Distributed training
parser.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank for distributed training")

# Wandb logging
parser.add_argument('--use_wandb', action='store_true',
                    help='Use Weights & Biases for logging')
parser.add_argument('--wandb_project', type=str, default='spaf_training',
                    help='W&B project name')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='W&B run name')

# DeepSpeed config
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# Initialize distributed training
deepspeed.init_distributed()
rank = dist.get_rank()
world_size = dist.get_world_size()

# Setup logging
if rank == 0:
    logger.remove()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file = os.path.join(args.checkpoint_dir, 'training.log')
    logger.add(log_file, level="INFO", mode="w")
    logger.info(f"Arguments: {args}")
    logger.info(f"World size: {world_size}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"spaf_rank{rank}",
            config=vars(args)
        )

# Import SpAF modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gumiho.model.spaf_model import SpAFModel
from gumiho.train.train_spaf import SpAFDataset, SpAFLoss


def load_tokenized_data(data_path: str, max_samples: int = None):
    """Load pre-tokenized data from files."""
    logger.info(f"Loading data from {data_path}")
    
    if os.path.isfile(data_path):
        # Single file
        data = torch.load(data_path)
        if isinstance(data, list):
            tokenized_data = data
        else:
            tokenized_data = [data]
    elif os.path.isdir(data_path):
        # Directory with multiple files
        tokenized_data = []
        for file_name in os.listdir(data_path):
            if file_name.endswith('.pt') or file_name.endswith('.pth'):
                file_path = os.path.join(data_path, file_name)
                data = torch.load(file_path)
                if isinstance(data, list):
                    tokenized_data.extend(data)
                else:
                    tokenized_data.append(data)
    else:
        raise ValueError(f"Data path {data_path} is neither a file nor a directory")
    
    if max_samples and len(tokenized_data) > max_samples:
        tokenized_data = tokenized_data[:max_samples]
    
    if rank == 0:
        logger.info(f"Loaded {len(tokenized_data)} samples")
    
    return tokenized_data


def evaluate(
    model_engine,
    eval_loader: DataLoader,
    loss_fn: SpAFLoss,
    head_engine: nn.Module,
    epoch: int,
    global_step: int,
) -> Dict[str, float]:
    """Evaluate the model."""
    model_engine.eval()
    
    total_losses = {
        'total': 0.0,
        'adapter_avg': 0.0,
        'alignment': 0.0,
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            
            # Forward pass (frozen base model)
            base_outputs = model_engine.module.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden_states = base_outputs.hidden_states
            
            # Compute adapter logits
            adapter_logits_list = []
            for idx, layer in enumerate(model_engine.module.base_model.model.layers):
                if layer.adapter is not None and idx < len(all_hidden_states) - 1:
                    layer_hidden = all_hidden_states[idx + 1]
                    adapter_logits = layer.adapter(layer_hidden)
                    adapter_logits_list.append(adapter_logits)
                else:
                    adapter_logits_list.append(None)
            
            # Compute alignment prediction
            target_tokens = input_ids[:, 1:]
            if target_tokens.size(1) > 0:
                alignment_pred = model_engine.module.alignment_head(token_ids=target_tokens)
                alignment_target = all_hidden_states[model_engine.module.cutoff_layer][:, :-1, :]
            else:
                alignment_pred = None
                alignment_target = None
            
            # Compute loss
            loss, loss_dict = loss_fn(
                adapter_logits=adapter_logits_list,
                alignment_pred=alignment_pred,
                alignment_target=alignment_target,
                target_ids=input_ids,
            )
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key in total_losses:
                    total_losses[key] += value
            num_batches += 1
    
    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    
    # Gather losses from all ranks
    for key in avg_losses:
        tensor = torch.tensor(avg_losses[key], device=rank)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        avg_losses[key] = (tensor / world_size).item()
    
    if rank == 0:
        logger.info(f"\nEvaluation at epoch {epoch}, step {global_step}:")
        logger.info(f"  Eval Total Loss: {avg_losses['total']:.4f}")
        logger.info(f"  Eval Adapter Loss: {avg_losses['adapter_avg']:.4f}")
        logger.info(f"  Eval Alignment Loss: {avg_losses['alignment']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'eval/total_loss': avg_losses['total'],
                'eval/adapter_loss': avg_losses['adapter_avg'],
                'eval/alignment_loss': avg_losses['alignment'],
                'epoch': epoch,
                'step': global_step,
            })
    
    model_engine.train()
    return avg_losses


def main():
    # Load SpAF configuration
    if rank == 0:
        logger.info(f"Loading SpAF configuration from {args.spaf_config_path}")
    
    if os.path.exists(args.spaf_config_path):
        with open(args.spaf_config_path, 'r') as f:
            spaf_full_config = json.load(f)
        spaf_config = spaf_full_config.get('spaf_settings', {})
    else:
        spaf_config = {}
    
    # Override with command line arguments
    if args.cutoff_layer is not None:
        spaf_config['cutoff_layer'] = args.cutoff_layer
    spaf_config['adapter_dim_ratio'] = args.adapter_dim_ratio
    spaf_config['enable_spaf'] = True
    
    # Initialize SpAF model
    if rank == 0:
        logger.info("Initializing SpAF model...")
    
    model = SpAFModel.from_pretrained(
        base_model_path=args.base_model_path,
        spaf_config=spaf_config,
        torch_dtype=torch.float16,
    )
    
    # Freeze base model
    model.freeze_base_model()
    
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    # Load training data
    train_data = load_tokenized_data(args.train_data_path)
    train_dataset = SpAFDataset(train_data, args.max_seq_length)
    
    # Load evaluation data if provided
    eval_loader = None
    if args.eval_data_path:
        eval_data = load_tokenized_data(args.eval_data_path)
        eval_dataset = SpAFDataset(eval_data, args.max_seq_length)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
    
    # Setup loss function
    loss_weights = {
        'alignment': args.alignment_weight,
        'adapter_base': args.adapter_weight,
    }
    loss_fn = SpAFLoss(
        cutoff_layer=model.cutoff_layer,
        num_layers=model.num_layers,
        loss_weights=loss_weights,
        alignment_loss_type=args.alignment_loss_type,
    )
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.get_trainable_parameters(),
        training_data=train_dataset,
    )
    
    # Setup frozen head for computing targets (similar to original code)
    head = model.base_model.lm_head
    head_engine = head.half().to(rank)
    head_engine.eval()
    for param in head_engine.parameters():
        param.requires_grad = False
    
    if rank == 0:
        logger.info("Starting training...")
    
    global_step = 0
    best_eval_loss = float('inf')
    
    # Training loop
    for epoch in range(args.num_epochs):
        model_engine.train()
        
        epoch_losses = {
            'total': 0.0,
            'adapter_avg': 0.0,
            'alignment': 0.0,
        }
        num_batches = 0
        
        if rank == 0:
            epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        else:
            epoch_iterator = train_loader
        
        for batch_idx, batch in enumerate(epoch_iterator):
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            
            model_engine.zero_grad()
            
            # Forward pass with frozen base model
            with torch.no_grad():
                base_outputs = model_engine.module.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                all_hidden_states = base_outputs.hidden_states
            
            # Compute adapter logits (with gradients)
            adapter_logits_list = []
            for idx, layer in enumerate(model_engine.module.base_model.model.layers):
                if layer.adapter is not None and idx < len(all_hidden_states) - 1:
                    layer_hidden = all_hidden_states[idx + 1].detach()
                    adapter_logits = layer.adapter(layer_hidden)
                    adapter_logits_list.append(adapter_logits)
                else:
                    adapter_logits_list.append(None)
            
            # Compute alignment head prediction (with gradients)
            target_tokens = input_ids[:, 1:]
            if target_tokens.size(1) > 0:
                alignment_pred = model_engine.module.alignment_head(token_ids=target_tokens)
                alignment_target = all_hidden_states[model_engine.module.cutoff_layer][:, :-1, :].detach()
            else:
                alignment_pred = None
                alignment_target = None
            
            # Compute loss
            loss, loss_dict = loss_fn(
                adapter_logits=adapter_logits_list,
                alignment_pred=alignment_pred,
                alignment_target=alignment_target,
                target_ids=input_ids,
            )
            
            # Backward and step
            model_engine.backward(loss)
            model_engine.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value
            num_batches += 1
            global_step += 1
            
            # Logging
            if rank == 0 and (batch_idx + 1) % args.logging_steps == 0:
                avg_loss = epoch_losses['total'] / num_batches
                logger.info(
                    f"Epoch {epoch+1}, Step {global_step}: "
                    f"Loss={loss_dict['total']:.4f}, "
                    f"Avg Loss={avg_loss:.4f}"
                )
                
                if args.use_wandb:
                    wandb.log({
                        'train/loss': loss_dict['total'],
                        'train/adapter_loss': loss_dict.get('adapter_avg', 0),
                        'train/alignment_loss': loss_dict.get('alignment', 0),
                        'train/avg_loss': avg_loss,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': global_step,
                    })
            
            # Evaluation
            if eval_loader and (global_step % args.eval_steps == 0):
                eval_losses = evaluate(
                    model_engine, eval_loader, loss_fn, head_engine,
                    epoch, global_step
                )
                
                # Save best model
                if rank == 0 and eval_losses['total'] < best_eval_loss:
                    best_eval_loss = eval_losses['total']
                    save_path = os.path.join(args.checkpoint_dir, 'best_model')
                    logger.info(f"Saving best model to {save_path}")
                    model_engine.module.save_spaf_components(save_path)
            
            # Save checkpoint
            if rank == 0 and (global_step % args.save_steps == 0):
                save_path = os.path.join(args.checkpoint_dir, f'checkpoint_step_{global_step}')
                logger.info(f"Saving checkpoint to {save_path}")
                model_engine.module.save_spaf_components(save_path)
        
        # End of epoch
        avg_epoch_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        
        if rank == 0:
            logger.info(f"\nEpoch {epoch+1} completed:")
            logger.info(f"  Avg Total Loss: {avg_epoch_losses['total']:.4f}")
            logger.info(f"  Avg Adapter Loss: {avg_epoch_losses['adapter_avg']:.4f}")
            logger.info(f"  Avg Alignment Loss: {avg_epoch_losses['alignment']:.4f}")
            
            # Save epoch checkpoint
            save_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}')
            logger.info(f"Saving epoch checkpoint to {save_path}")
            model_engine.module.save_spaf_components(save_path)
    
    if rank == 0:
        logger.info("Training completed!")
        if args.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    main()
