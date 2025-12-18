# -*- coding: utf-8 -*-
"""
Hybrid-AgileDrafter 模型包装器
集成 HybridAgileDrafter 与基础 LLM 模型
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gumiho.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from dynaspec.hybrid_agile_drafter import HybridAgileDrafter
from transformers import AutoTokenizer, AutoConfig


class HybridAgileDrafterModel(nn.Module):
    """
    Hybrid-AgileDrafter 完整模型
    
    包含：
        1. 基础 LLM 模型（用于获取隐藏状态）
        2. HybridAgileDrafter（草稿模型）
    
    Args:
        base_model: 基础 LLM 模型
        base_model_name_or_path: 基础模型路径
        config: HybridAgileDrafter 配置
    """
    
    def __init__(
        self,
        base_model,
        base_model_name_or_path: str,
        config
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path,
            use_fast=False
        )
        
        # 设置 HybridAgileDrafter 配置
        hybrid_config = type('HybridConfig', (), {})()
        hybrid_config.hidden_size = self.hidden_size
        hybrid_config.vocab_size = self.vocab_size
        hybrid_config.num_mlp_blocks = getattr(config, 'num_mlp_blocks', 2)
        hybrid_config.num_transformer_blocks = getattr(config, 'num_transformer_blocks', 3)
        hybrid_config.gumbel_temperature = getattr(config, 'gumbel_temperature', 1.0)
        hybrid_config.rms_norm_eps = getattr(self.config, 'rms_norm_eps', 1e-6)
        
        # 复制其他可能需要的配置
        for attr in ['num_attention_heads', 'num_key_value_heads', 'intermediate_size', 
                     'hidden_act', 'max_position_embeddings', 'rope_scaling', 
                     'pretraining_tp', 'rope_theta']:
            if hasattr(self.config, attr):
                setattr(hybrid_config, attr, getattr(self.config, attr))
        
        # 初始化 HybridAgileDrafter
        self.draft_model = HybridAgileDrafter(hybrid_config)
        
        # 获取设备
        self.device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        self.draft_model.to(self.device)
    
    def get_tokenizer(self):
        """获取 tokenizer"""
        return self.tokenizer
    
    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        从预训练模型加载
        
        Args:
            base_model_path: 基础模型路径
            config_path: HybridAgileDrafter 配置文件路径（可选）
            **kwargs: 其他参数
        """
        # 加载基础模型
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {Type}")
        
        # 加载配置
        if config_path:
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = type('Config', (), config_dict)()
        else:
            # 使用默认配置
            config = type('Config', (), {
                'num_mlp_blocks': 2,
                'num_transformer_blocks': 3,
                'gumbel_temperature': 1.0
            })()
        
        model = cls(base_model, base_model_path, config)
        return model
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_orig: bool = False
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            past_key_values: 过去的 KV 缓存
            position_ids: 位置 IDs
            output_orig: 是否输出原始 LLM 的 logits
        
        Returns:
            outputs: 基础模型输出
            orig (可选): 原始 LLM 的 logits
            hidden_states: LLM 隐藏状态
        """
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]
        
        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states
    
    def draft_forward(
        self,
        input_ids: torch.Tensor,
        llm_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        草稿模型前向传播
        
        Args:
            input_ids: 输入 token IDs
            llm_hidden_states: LLM 隐藏状态
            attention_mask: 注意力掩码
            position_ids: 位置 IDs
        
        Returns:
            draft_logits: 草稿模型的 logits
            gating_probs: 门控概率
        """
        # 获取输入的嵌入
        inputs_embeds = self.base_model.model.embed_tokens(input_ids)
        
        # 调用 HybridAgileDrafter
        draft_logits, gating_probs = self.draft_model(
            inputs=inputs_embeds,
            llm_hidden_state=llm_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        return draft_logits, gating_probs
