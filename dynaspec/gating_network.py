# -*- coding: utf-8 -*-
"""
门控网络 (Gating Network) 用于 Hybrid-AgileDrafter
根据 LLM 隐藏状态动态选择使用哪个块（MLP 或 Transformer）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    门控网络：基于 LLM 隐藏状态选择最佳的块
    
    Args:
        input_dim (int): LLM 隐藏状态的维度
        num_choices (int): 总块数 N (K 个 MLP 块 + M 个 Transformer 块)
    """
    
    def __init__(self, input_dim: int, num_choices: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_choices = num_choices
        
        # 两层 MLP：input_dim -> input_dim // 2 -> num_choices
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_choices)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_state: LLM 隐藏状态
                - shape: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
        
        Returns:
            logits: 原始 logits，形状为 [batch_size, num_choices]
        """
        # 如果输入是 3D (batch_size, seq_len, input_dim)，取最后一个 token 的隐藏状态
        if hidden_state.dim() == 3:
            gating_input = hidden_state[:, -1, :]  # [batch_size, input_dim]
        elif hidden_state.dim() == 2:
            gating_input = hidden_state  # [batch_size, input_dim]
        else:
            raise ValueError(f"Unsupported hidden_state dimension: {hidden_state.dim()}")

        logits = self.mlp(gating_input)  # [batch_size, num_choices]
        return logits
