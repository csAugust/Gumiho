# -*- coding: utf-8 -*-
"""
Hybrid-AgileDrafter 损失计算函数
包含模仿损失和成本预期损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_hybrid_loss(
    draft_logits: torch.Tensor,
    target_labels: torch.Tensor,
    gating_probs: torch.Tensor,
    num_mlp_blocks: int,
    num_transformer_blocks: int,
    cost_lambda: float = 0.01,
    mlp_cost: float = 1.0,
    transformer_cost: float = 5.0,
    loss_mask: torch.Tensor = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算 Hybrid-AgileDrafter 的总损失
    
    损失组成：
        1. 模仿损失 (Imitation Loss): 交叉熵损失，使草稿模型模仿 LLM
        2. 成本预期损失 (Expected Cost Loss): 鼓励模型选择计算成本更低的块
    
    Args:
        draft_logits: 草稿模型输出的 logits [batch_size, seq_len, vocab_size]
        target_labels: 目标 token IDs [batch_size, seq_len]
        gating_probs: 门控概率分布 [batch_size, total_blocks]
        num_mlp_blocks: MLP 块数量
        num_transformer_blocks: Transformer 块数量
        cost_lambda: 成本损失的权重系数
        mlp_cost: 每个 MLP 块的单位成本
        transformer_cost: 每个 Transformer 块的单位成本
        loss_mask: 损失掩码，用于忽略 padding token [batch_size, seq_len] (可选)
    
    Returns:
        total_loss: 总损失
        loss_dict: 包含各项损失的字典
    """
    batch_size, seq_len, vocab_size = draft_logits.shape
    total_blocks = num_mlp_blocks + num_transformer_blocks
    
    # ===== 1. 模仿损失（交叉熵） =====
    # 将 logits 和 labels 展平以计算交叉熵
    draft_logits_flat = draft_logits.view(-1, vocab_size)  # [B*SeqLen, VocabSize]
    target_labels_flat = target_labels.view(-1)  # [B*SeqLen]
    
    if loss_mask is not None:
        # 应用损失掩码
        loss_mask_flat = loss_mask.view(-1)  # [B*SeqLen]
        # 计算每个位置的交叉熵
        ce_loss_unreduced = F.cross_entropy(
            draft_logits_flat,
            target_labels_flat,
            reduction='none'
        )  # [B*SeqLen]
        # 只在非掩码位置计算损失
        draft_loss = (ce_loss_unreduced * loss_mask_flat).sum() / (loss_mask_flat.sum() + 1e-8)
    else:
        # 不使用掩码，直接计算平均交叉熵
        draft_loss = F.cross_entropy(draft_logits_flat, target_labels_flat)
    
    # ===== 2. 成本预期损失 =====
    # 定义每个块的成本向量
    costs = torch.zeros(total_blocks, device=gating_probs.device)
    costs[:num_mlp_blocks] = mlp_cost  # 前 K 个是 MLP 块
    costs[num_mlp_blocks:] = transformer_cost  # 后 M 个是 Transformer 块
    
    # 计算批量平均预期成本
    # gating_probs: [B, N], costs: [N]
    expected_cost = (gating_probs * costs.unsqueeze(0)).sum(dim=-1).mean()
    
    # ===== 3. 总损失 =====
    total_loss = draft_loss + cost_lambda * expected_cost
    
    # ===== 4. 构建损失字典 =====
    loss_dict = {
        "total_loss": total_loss,
        "draft_loss": draft_loss,
        "expected_cost": expected_cost,
        "cost_loss": cost_lambda * expected_cost
    }
    
    # 添加门控统计信息（用于监控）
    with torch.no_grad():
        # 计算每个块被选择的平均概率
        for i in range(total_blocks):
            block_type = "mlp" if i < num_mlp_blocks else "transformer"
            block_idx = i if i < num_mlp_blocks else i - num_mlp_blocks
            loss_dict[f"gating/{block_type}_{block_idx}_prob"] = gating_probs[:, i].mean()
        
        # 计算 MLP 块和 Transformer 块的总概率
        mlp_total_prob = gating_probs[:, :num_mlp_blocks].sum(dim=-1).mean()
        transformer_total_prob = gating_probs[:, num_mlp_blocks:].sum(dim=-1).mean()
        loss_dict["gating/mlp_total_prob"] = mlp_total_prob
        loss_dict["gating/transformer_total_prob"] = transformer_total_prob
    
    return total_loss, loss_dict


def compute_accuracy(
    draft_logits: torch.Tensor,
    target_labels: torch.Tensor,
    loss_mask: torch.Tensor = None,
    topk: Tuple[int, ...] = (1, 3, 5)
) -> Dict[str, float]:
    """
    计算草稿模型的准确率
    
    Args:
        draft_logits: 草稿模型输出的 logits [batch_size, seq_len, vocab_size]
        target_labels: 目标 token IDs [batch_size, seq_len]
        loss_mask: 损失掩码 [batch_size, seq_len] (可选)
        topk: 要计算的 top-k 准确率
    
    Returns:
        acc_dict: 包含各项准确率的字典
    """
    with torch.no_grad():
        batch_size, seq_len, vocab_size = draft_logits.shape
        
        # 展平
        draft_logits_flat = draft_logits.view(-1, vocab_size)  # [B*SeqLen, VocabSize]
        target_labels_flat = target_labels.view(-1)  # [B*SeqLen]
        
        if loss_mask is not None:
            loss_mask_flat = loss_mask.view(-1)  # [B*SeqLen]
            # 只在有效位置计算准确率
            valid_indices = loss_mask_flat.bool()
            draft_logits_flat = draft_logits_flat[valid_indices]
            target_labels_flat = target_labels_flat[valid_indices]
            num_valid = valid_indices.sum().item()
        else:
            num_valid = batch_size * seq_len
        
        if num_valid == 0:
            # 没有有效样本
            return {f"top{k}_acc": 0.0 for k in topk}
        
        # 计算 top-k 准确率
        acc_dict = {}
        max_k = max(topk)
        _, pred_topk = draft_logits_flat.topk(max_k, dim=-1, largest=True, sorted=True)
        # pred_topk: [N, max_k]
        
        target_expanded = target_labels_flat.unsqueeze(-1).expand_as(pred_topk)
        # target_expanded: [N, max_k]
        
        correct = pred_topk.eq(target_expanded)  # [N, max_k]
        
        for k in topk:
            correct_k = correct[:, :k].any(dim=-1).float().sum().item()
            acc_dict[f"top{k}_acc"] = correct_k / num_valid
        
        return acc_dict
