# -*- coding: utf-8 -*-

from safetensors import safe_open
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gumiho.model.cnets import LlamaMLP, LlamaDecoderLayer, LlamaRMSNorm, ParallelMLPs, _expand_mask, _make_causal_mask
from dynaspec.model.gating_network import GatingNetwork
from transformers.activations import ACT2FN


def load_single_tensor(base_model_path, tensor_name):
    try:
        with open(os.path.join(base_model_path, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            emb_path = index_json["weight_map"][tensor_name]
        with safe_open(os.path.join(base_model_path, emb_path),
                        framework="pt",
                        device="cpu") as f:
            tensor_slice = f.get_slice(tensor_name)
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(base_model_path, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            emb_path = index_json["weight_map"][tensor_name]
        weights = torch.load(os.path.join(base_model_path, emb_path))
        tensor = weights[tensor_name].float()
    return tensor


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
    
    def __init__(self, config, args, tokenizer):
        super().__init__()
        
        self.config = config
        self.args = args
        self.tokenizer = tokenizer

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        
        self.num_mlp_blocks = getattr(config, 'num_mlp_blocks', 2)
        self.num_transformer_blocks = getattr(config, 'num_transformer_blocks', 3)
        self.total_blocks = self.num_mlp_blocks + self.num_transformer_blocks
        self.temperature = getattr(config, 'gumbel_temperature', 1.0)
        self.is_training = True  # 默认为训练模式
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
        
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.act = ACT2FN[self.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.serial_head_num = args.serial_head_num
        self.transformer_blocks = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(self.num_transformer_blocks)])

        self.mlp = ParallelMLPs(config.hidden_size, self.num_mlp_blocks)

        # self.hybrid_blocks = nn.ModuleList()

        # # 1. 添加 K 个 MLP 块
        # for _ in range(self.num_mlp_blocks):
        #     mlp_block = LlamaMLP(config)
        #     self.hybrid_blocks.append(mlp_block)
        
        # # 2. 添加 M 个 Transformer 层
        # for idx in range(self.num_transformer_blocks):
        #     # LlamaDecoderLayer 需要一个 index 参数
        #     # 为了简化，我们使用 idx，但注意第一个层（index=0）可能没有 input_layernorm
        #     transformer_block = LlamaDecoderLayer(config, index=idx+1)
        #     self.hybrid_blocks.append(transformer_block)
        
        # # 层归一化（用于残差连接）
        # self.layer_norms = nn.ModuleList([
        #     LlamaRMSNorm(self.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        #     for _ in range(self.total_blocks)
        # ])
        
        # # 门控网络
        # self.gating_network = GatingNetwork(
        #     input_dim=self.hidden_size,
        #     num_choices=self.total_blocks
        # )
        
        # # 输出头：将块输出映射到词汇表
        # self.output_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # # 初始投影层：用于处理输入
        # self.input_projection = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0
        seq_length_with_past = seq_length

        # ===== prepare position_ids & attention_mask =====
        inputs_embeds = self.embed_tokens(input_ids).to(hidden_states.dtype)
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.transformer_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        
        ret_hidden_states = []
        if self.is_training:
            mlp_inputs = []
            inputs_embeds_shifted = torch.roll(inputs_embeds, shifts=-1, dims=1)
            hidden_states_mlp = self.fc(torch.cat((inputs_embeds_shifted, hidden_states), dim=-1))
            for i in range(self.serial_head_num):
                mlp_inputs.append(torch.roll(hidden_states_mlp, shifts=-(i), dims=1))
            
            mlp_inputs = torch.cat(mlp_inputs, dim=-1)  # (bs, sl, 2*hidden_dim)
            for i in range(self.num_mlp_blocks):
                ret_hidden_states.append(self.mlp.mlp[i](mlp_inputs))

        ret_hidden_states.append(hidden_states)
        if self.is_training:
            if use_cache:
                return hidden_states, next_decoder_cache
            else:
                return hidden_states
        else:
            if use_cache:
                return ret_hidden_states, next_decoder_cache
            else:
                return ret_hidden_states


        # # ===== 第一步：门控决策 =====
        # gating_logits = self.gating_network(hidden_states)  # [B, N]
        
        # # ===== 第二步：Gumbel-Softmax =====
        # # 训练时用软概率（hard=False），推理时用硬选择（hard=True）
        # gating_probs = F.gumbel_softmax(
        #     gating_logits,
        #     tau=self.temperature,
        #     hard=not self.is_training,
        #     dim=-1
        # )  # [B, N]
        
        # # ===== 第三步：渐进式处理 =====
        # # 初始投影
        # x = self.input_projection(inputs)  # [B, SeqLen, HiddenSize]
        
        # # 存储每个块的输出 logits
        # block_logits_list = []
        
        # for i, block in enumerate(self.hybrid_blocks):
        #     # 保存残差
        #     residual = x
            
        #     # 应用块
        #     if i < self.num_mlp_blocks:
        #         # MLP 块：直接前向传播
        #         block_output = block(x)  # [B, SeqLen, HiddenSize]
        #     else:
        #         # Transformer 块：需要额外参数
        #         # LlamaDecoderLayer 返回一个元组 (hidden_states, ...)
        #         layer_outputs = block(
        #             x,
        #             attention_mask=attention_mask,
        #             position_ids=position_ids,
        #             past_key_value=None,
        #             output_attentions=False,
        #             use_cache=False
        #         )
        #         block_output = layer_outputs[0]  # 取第一个元素（hidden_states）
            
        #     # 残差连接
        #     x = residual + block_output
            
        #     # 层归一化
        #     x = self.layer_norms[i](x)
            
        #     # 通过输出头生成该块的 logits
        #     block_logits = self.output_head(x)  # [B, SeqLen, VocabSize]
        #     block_logits_list.append(block_logits)
        
        # # ===== 第四步：加权融合 =====
        # # 堆叠所有块的 logits
        # stacked_logits = torch.stack(block_logits_list, dim=1)  # [B, N, SeqLen, VocabSize]
        
        # # 重塑门控概率以进行广播
        # gating_probs_reshaped = gating_probs.view(batch_size, self.total_blocks, 1, 1)
        # # [B, N, 1, 1]
        
        # # 加权求和
        # final_logits = torch.sum(
        #     stacked_logits * gating_probs_reshaped,
        #     dim=1
        # )  # [B, SeqLen, VocabSize]
        
        # return final_logits, gating_probs
    
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
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

