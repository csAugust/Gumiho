# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
SpAF (Speculative Adapter Fusion) Modules

This module contains the Adapter and AlignmentHead components for SpAF architecture.
"""

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class Adapter(nn.Module):
    """
    A lightweight Adapter module to be inserted into each Transformer Block.
    
    The Adapter uses a bottleneck architecture with down-projection, activation, and up-projection.
    It predicts the final output token based on intermediate hidden states.
    
    Args:
        hidden_size (int): The hidden size of the base model.
        vocab_size (int): The vocabulary size for prediction.
        adapter_dim_ratio (float): Ratio to determine the intermediate dimension (default: 0.25).
        hidden_act (str): Activation function name (default: "silu").
    """
    
    def __init__(self, hidden_size, vocab_size, adapter_dim_ratio=0.25, hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_dim = max(1, int(hidden_size * adapter_dim_ratio))
        
        # Bottleneck architecture
        self.down_proj = nn.Linear(hidden_size, self.intermediate_dim, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.up_proj = nn.Linear(self.intermediate_dim, hidden_size, bias=False)
        
        # Prediction head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, hidden_states, return_features=False):
        """
        Forward pass of the Adapter.
        
        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch, seq_len, hidden_size).
            return_features (bool): If True, return intermediate features along with logits.
            
        Returns:
            torch.Tensor: Logits of shape (batch, seq_len, vocab_size) if return_features=False,
                         or tuple of (logits, features) if return_features=True.
        """
        # Bottleneck transformation
        down = self.down_proj(hidden_states)
        activated = self.act_fn(down)
        features = self.up_proj(activated)
        
        # Predict tokens
        logits = self.lm_head(features)
        
        if return_features:
            return logits, features
        return logits


class AlignmentHead(nn.Module):
    """
    AlignmentHead module that generates pseudo hidden states from token embeddings.
    
    This module is trained to mimic the hidden states at layer k-1 given a token embedding,
    enabling parallel verification in SpAF.
    
    Args:
        vocab_size (int): The vocabulary size.
        embedding_dim (int): The dimension of token embeddings.
        hidden_size (int): The target hidden size (should match model's hidden size at layer k-1).
        num_layers (int): Number of MLP layers (default: 2).
        hidden_dim_ratio (float): Ratio for intermediate dimension (default: 2.0).
        hidden_act (str): Activation function name (default: "silu").
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, 
                 hidden_dim_ratio=2.0, hidden_act="silu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        intermediate_dim = int(hidden_size * hidden_dim_ratio)
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, intermediate_dim, bias=True),
                ACT2FN[hidden_act],
                nn.LayerNorm(intermediate_dim)
            ])
            input_dim = intermediate_dim
        
        # Final projection to target hidden size
        layers.append(nn.Linear(input_dim, hidden_size, bias=True))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, token_ids=None, token_embeddings=None):
        """
        Forward pass of the AlignmentHead.
        
        Args:
            token_ids (torch.LongTensor, optional): Token IDs of shape (batch, seq_len).
            token_embeddings (torch.Tensor, optional): Pre-computed token embeddings of shape 
                                                        (batch, seq_len, embedding_dim).
            
        Returns:
            torch.Tensor: Pseudo hidden states of shape (batch, seq_len, hidden_size).
        """
        if token_embeddings is None:
            if token_ids is None:
                raise ValueError("Either token_ids or token_embeddings must be provided")
            token_embeddings = self.token_embedding(token_ids)
        
        # Generate pseudo hidden states
        pseudo_hidden_states = self.mlp(token_embeddings)
        
        return pseudo_hidden_states
    
    def load_embedding_from_model(self, model_embedding_weight):
        """
        Initialize token embeddings from a pretrained model.
        
        Args:
            model_embedding_weight (torch.Tensor): Embedding weight from pretrained model.
        """
        self.token_embedding.weight.data.copy_(model_embedding_weight)
        
    def freeze_embeddings(self):
        """Freeze the token embedding layer."""
        self.token_embedding.weight.requires_grad = False
