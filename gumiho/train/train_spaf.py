# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
SpAF Training Script

This module implements the training logic for SpAF model with multi-task joint loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpAFLoss(nn.Module):
    """
    Multi-task joint loss for SpAF training.
    
    Combines:
    - Adapter losses: CrossEntropy for token prediction at each layer
    - Alignment loss: MSE or Cosine Similarity for hidden state alignment
    
    Args:
        cutoff_layer (int): The layer index where SpAF starts.
        num_layers (int): Total number of layers in the model.
        loss_weights (dict): Dictionary containing loss weights.
        alignment_loss_type (str): Type of alignment loss ('mse' or 'cosine').
    """
    
    def __init__(
        self,
        cutoff_layer: int,
        num_layers: int,
        loss_weights: Optional[Dict] = None,
        alignment_loss_type: str = 'mse',
    ):
        super().__init__()
        
        self.cutoff_layer = cutoff_layer
        self.num_layers = num_layers
        self.alignment_loss_type = alignment_loss_type
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'alignment': 1.0,
                'adapter_base': 1.0,  # Base weight for adapters
            }
        
        self.w_align = loss_weights.get('alignment', 1.0)
        self.w_adapter_base = loss_weights.get('adapter_base', 1.0)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def compute_alignment_loss(
        self,
        pred_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alignment loss between predicted and target hidden states.
        
        Args:
            pred_hidden: Predicted hidden states from AlignmentHead.
            target_hidden: Target hidden states from the model.
            
        Returns:
            Alignment loss value.
        """
        if self.alignment_loss_type == 'mse':
            return F.mse_loss(pred_hidden, target_hidden)
        elif self.alignment_loss_type == 'cosine':
            # Cosine similarity loss: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(
                pred_hidden.flatten(0, 1),
                target_hidden.flatten(0, 1),
                dim=-1
            )
            return 1.0 - cos_sim.mean()
        else:
            raise ValueError(f"Unknown alignment loss type: {self.alignment_loss_type}")
    
    def forward(
        self,
        adapter_logits: List[Optional[torch.Tensor]],
        alignment_pred: Optional[torch.Tensor],
        alignment_target: Optional[torch.Tensor],
        target_ids: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task joint loss.
        
        Args:
            adapter_logits: List of adapter logits for each layer.
            alignment_pred: Predicted hidden states from AlignmentHead.
            alignment_target: Target hidden states at cutoff layer.
            target_ids: Target token IDs.
            
        Returns:
            Dictionary containing total loss and individual loss components.
        """
        losses = {}
        total_loss = 0.0
        
        # Adapter losses
        adapter_loss_sum = 0.0
        num_adapters = 0
        
        for idx, logits in enumerate(adapter_logits):
            if logits is not None and idx >= self.cutoff_layer:
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                
                # Flatten for loss computation
                loss = self.ce_loss(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                adapter_loss_sum += loss
                num_adapters += 1
                losses[f'adapter_{idx}'] = loss.item()
        
        if num_adapters > 0:
            avg_adapter_loss = adapter_loss_sum / num_adapters
            losses['adapter_avg'] = avg_adapter_loss.item()
            total_loss += self.w_adapter_base * avg_adapter_loss
        
        # Alignment loss
        if alignment_pred is not None and alignment_target is not None:
            alignment_loss = self.compute_alignment_loss(
                alignment_pred, alignment_target
            )
            losses['alignment'] = alignment_loss.item()
            total_loss += self.w_align * alignment_loss
        
        losses['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, losses


class SpAFDataset(Dataset):
    """
    Dataset for SpAF training.
    
    This dataset preprocesses text data to extract:
    - Input token sequences
    - Target token sequences
    - Intermediate hidden states (computed on-the-fly or precomputed)
    
    Args:
        tokenized_data: Tokenized text data.
        max_length: Maximum sequence length.
    """
    
    def __init__(
        self,
        tokenized_data: List[Dict],
        max_length: int = 512,
    ):
        self.data = tokenized_data
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_ids = item['input_ids'][:self.max_length]
        attention_mask = item.get('attention_mask', [1] * len(input_ids))[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


def train_spaf_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SpAFLoss,
    device: str = 'cuda',
    log_interval: int = 100,
) -> Dict[str, float]:
    """
    Train SpAF model for one epoch.
    
    Args:
        model: SpAFModel instance.
        dataloader: Training data loader.
        optimizer: Optimizer.
        loss_fn: SpAF loss function.
        device: Device to train on.
        log_interval: Logging interval.
        
    Returns:
        Dictionary containing average losses.
    """
    model.train()
    
    total_losses = {
        'total': 0.0,
        'adapter_avg': 0.0,
        'alignment': 0.0,
    }
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass with frozen base model
        with torch.no_grad():
            # Get all hidden states from base model
            base_outputs = model.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden_states = base_outputs.hidden_states
        
        # Compute adapter logits (with gradients)
        adapter_logits_list = []
        for idx, layer in enumerate(model.base_model.model.layers):
            if layer.adapter is not None and idx < len(all_hidden_states) - 1:
                layer_hidden = all_hidden_states[idx + 1].detach()
                adapter_logits = layer.adapter(layer_hidden)
                adapter_logits_list.append(adapter_logits)
            else:
                adapter_logits_list.append(None)
        
        # Compute alignment head prediction
        # Use the next token as target for alignment
        target_tokens = input_ids[:, 1:]  # Shift by 1
        if target_tokens.size(1) > 0:
            alignment_pred = model.alignment_head(token_ids=target_tokens)
            # Target is the hidden state at cutoff_layer - 1 for the previous position
            alignment_target = all_hidden_states[model.cutoff_layer][:, :-1, :].detach()
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key, value in loss_dict.items():
            if key in total_losses:
                total_losses[key] += value
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_losses['total'] / num_batches
            logger.info(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Current Loss: {loss_dict['total']:.4f}"
            )
    
    # Compute average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    
    return avg_losses


def setup_spaf_training(
    model,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    loss_config: Optional[Dict] = None,
):
    """
    Setup SpAF model for training.
    
    Args:
        model: SpAFModel instance.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        loss_config: Configuration for loss function.
        
    Returns:
        Tuple of (optimizer, loss_fn).
    """
    # Freeze base model
    model.freeze_base_model()
    
    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Setup loss function
    if loss_config is None:
        loss_config = {
            'alignment': 1.0,
            'adapter_base': 1.0,
        }
    
    loss_fn = SpAFLoss(
        cutoff_layer=model.cutoff_layer,
        num_layers=model.num_layers,
        loss_weights=loss_config,
        alignment_loss_type='mse',
    )
    
    return optimizer, loss_fn


def evaluate_spaf(
    model,
    dataloader: DataLoader,
    loss_fn: SpAFLoss,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate SpAF model.
    
    Args:
        model: SpAFModel instance.
        dataloader: Evaluation data loader.
        loss_fn: SpAF loss function.
        device: Device to evaluate on.
        
    Returns:
        Dictionary containing average losses.
    """
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'adapter_avg': 0.0,
        'alignment': 0.0,
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            base_outputs = model.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden_states = base_outputs.hidden_states
            
            # Compute adapter logits
            adapter_logits_list = []
            for idx, layer in enumerate(model.base_model.model.layers):
                if layer.adapter is not None and idx < len(all_hidden_states) - 1:
                    layer_hidden = all_hidden_states[idx + 1]
                    adapter_logits = layer.adapter(layer_hidden)
                    adapter_logits_list.append(adapter_logits)
                else:
                    adapter_logits_list.append(None)
            
            # Compute alignment prediction
            target_tokens = input_ids[:, 1:]
            if target_tokens.size(1) > 0:
                alignment_pred = model.alignment_head(token_ids=target_tokens)
                alignment_target = all_hidden_states[model.cutoff_layer][:, :-1, :]
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
    
    # Compute average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    
    return avg_losses
