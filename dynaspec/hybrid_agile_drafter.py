# -*- coding: utf-8 -*-
"""
Hybrid-AgileDrafter: 动态异构序列模型
使用可微门控网络选择性组合 MLP 和 Transformer 块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# 导入 Gumiho 仓库中的现有组件
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gumiho.model.cnets import LlamaMLP, LlamaDecoderLayer, LlamaRMSNorm
from dynaspec.gating_network import GatingNetwork


class HybridAgileDrafter(nn.Module):
    """
    Hybrid-AgileDrafter: 动态异构草稿模型
    
    架构:
        1. 门控网络：基于 LLM 隐藏状态选择最佳块
        2. 混合块序列：K 个轻量级 MLP 块 + M 个 Transformer 层
        3. 加权融合：使用 Gumbel-Softmax 加权融合各块的输出
    
    Args:
        config: 模型配置对象，应包含以下属性：
            - hidden_size: 隐藏层维度
            - vocab_size: 词汇表大小
            - num_mlp_blocks: MLP 块数量 (K)
            - num_transformer_blocks: Transformer 块数量 (M)
            - gumbel_temperature: Gumbel-Softmax 温度 (默认 1.0)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # 从配置中读取块数量
        self.num_mlp_blocks = getattr(config, 'num_mlp_blocks', 2)
        self.num_transformer_blocks = getattr(config, 'num_transformer_blocks', 3)
        self.total_blocks = self.num_mlp_blocks + self.num_transformer_blocks
        
        # Gumbel-Softmax 温度
        self.temperature = getattr(config, 'gumbel_temperature', 1.0)
        self.is_training = True  # 默认为训练模式
        
        # 构建混合块序列
        self.hybrid_blocks = nn.ModuleList()
        
        # 1. 添加 K 个 MLP 块
        for _ in range(self.num_mlp_blocks):
            mlp_block = LlamaMLP(config)
            self.hybrid_blocks.append(mlp_block)
        
        # 2. 添加 M 个 Transformer 层
        for idx in range(self.num_transformer_blocks):
            # LlamaDecoderLayer 需要一个 index 参数
            # 为了简化，我们使用 idx，但注意第一个层（index=0）可能没有 input_layernorm
            transformer_block = LlamaDecoderLayer(config, index=idx+1)
            self.hybrid_blocks.append(transformer_block)
        
        # 层归一化（用于残差连接）
        self.layer_norms = nn.ModuleList([
            LlamaRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
            for _ in range(self.total_blocks)
        ])
        
        # 门控网络
        self.gating_network = GatingNetwork(
            input_dim=self.hidden_size,
            num_choices=self.total_blocks
        )
        
        # 输出头：将块输出映射到词汇表
        self.output_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # 初始投影层：用于处理输入
        self.input_projection = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(
        self,
        inputs: torch.Tensor,
        llm_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            inputs: 输入 token 的嵌入或隐藏状态 [batch_size, seq_len, hidden_size]
            llm_hidden_state: LLM 的隐藏状态，用于门控决策 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码（可选）
            position_ids: 位置 ID（可选）
        
        Returns:
            final_logits: 加权融合后的最终 logits [batch_size, seq_len, vocab_size]
            gating_probs: 门控概率分布 [batch_size, total_blocks]
        """
        batch_size, seq_len, _ = inputs.shape
        
        # ===== 第一步：门控决策 =====
        gating_logits = self.gating_network(llm_hidden_state)  # [B, N]
        
        # ===== 第二步：Gumbel-Softmax =====
        # 训练时用软概率（hard=False），推理时用硬选择（hard=True）
        gating_probs = F.gumbel_softmax(
            gating_logits,
            tau=self.temperature,
            hard=not self.is_training,
            dim=-1
        )  # [B, N]
        
        # ===== 第三步：渐进式处理 =====
        # 初始投影
        x = self.input_projection(inputs)  # [B, SeqLen, HiddenSize]
        
        # 存储每个块的输出 logits
        block_logits_list = []
        
        for i, block in enumerate(self.hybrid_blocks):
            # 保存残差
            residual = x
            
            # 应用块
            if i < self.num_mlp_blocks:
                # MLP 块：直接前向传播
                block_output = block(x)  # [B, SeqLen, HiddenSize]
            else:
                # Transformer 块：需要额外参数
                # LlamaDecoderLayer 返回一个元组 (hidden_states, ...)
                layer_outputs = block(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False
                )
                block_output = layer_outputs[0]  # 取第一个元素（hidden_states）
            
            # 残差连接
            x = residual + block_output
            
            # 层归一化
            x = self.layer_norms[i](x)
            
            # 通过输出头生成该块的 logits
            block_logits = self.output_head(x)  # [B, SeqLen, VocabSize]
            block_logits_list.append(block_logits)
        
        # ===== 第四步：加权融合 =====
        # 堆叠所有块的 logits
        stacked_logits = torch.stack(block_logits_list, dim=1)  # [B, N, SeqLen, VocabSize]
        
        # 重塑门控概率以进行广播
        gating_probs_reshaped = gating_probs.view(batch_size, self.total_blocks, 1, 1)
        # [B, N, 1, 1]
        
        # 加权求和
        final_logits = torch.sum(
            stacked_logits * gating_probs_reshaped,
            dim=1
        )  # [B, SeqLen, VocabSize]
        
        return final_logits, gating_probs
    
    def train(self, mode: bool = True):
        """重写 train 方法以更新 is_training 标志"""
        super().train(mode)
        self.is_training = mode
        return self
    
    def eval(self):
        """重写 eval 方法以更新 is_training 标志"""
        super().eval()
        self.is_training = False
        return self
