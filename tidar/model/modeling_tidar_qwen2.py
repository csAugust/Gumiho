"""
TiDAR-Qwen2 Model Implementation
"""

import random
from functools import partial
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Any

from .kv_cache import KVCache, initialize_past_key_values, rollback_past_key_values
from .modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from .configuration_tidar_qwen2 import TiDARQwen2Config

import os
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
):
    """
    Create a causal mask for bi-directional self-attention.

    Args:
        input_ids_shape (torch.Size): The shape of input_ids tensor, typically (batch_size, tgt_len).
        dtype (torch.dtype): The data type of the mask.
        device (torch.device): The device on which the mask will be placed.
        past_key_values_length (int, optional): The length of past key values. Default is 0.

    Returns:
        torch.Tensor: The causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (torch.Tensor): The attention mask tensor of shape `[bsz, seq_len]`.
        dtype (torch.dtype): The data type of the mask.
        tgt_len (Optional[int], optional): The target sequence length. If None, it defaults to the source sequence length.

    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class TiDARAttention(Qwen2Attention):
    """
    TiDAR Attention layer with support for hybrid attention patterns.
    
    This attention layer works with the hybrid attention mask created by TiDARModel.
    The actual attention pattern is controlled by the attention_mask passed to forward().
    """
    
    def __init__(self, config: TiDARQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.tidar_config = config.tidar_config
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[List[KVCache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with custom KVCache support."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from .modeling_qwen2 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # [MODIFIED] Using custom KVCache mechanism for preallocated GPU memory optimization
        if past_key_value is not None:
            # past_key_value is a list of two KVCache objects [key_cache, value_cache]
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)

        # Get attention interface
        from .modeling_qwen2 import eager_attention_forward, repeat_kv, ALL_ATTENTION_FUNCTIONS
        
        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                from transformers.utils import logging
                logger = logging.get_logger(__name__)
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TiDARDecoderLayer(Qwen2DecoderLayer):
    """
    TiDAR Decoder Layer with TiDAR attention.
    """
    
    def __init__(self, config: TiDARQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace self_attn with TiDAR version
        self.self_attn = TiDARAttention(config=config, layer_idx=layer_idx)


class TiDARModel(Qwen2Model):
    """
    TiDAR Model with hybrid attention mask support.
    
    This model implements the core TiDAR mechanism:
    1. Hybrid attention mask that combines:
       - Causal attention for clean tokens (autoregressive mode)
       - Block-wise bidirectional attention for masked tokens (diffusion mode)
       - Full attention from masked to clean tokens (conditioning)
       - No attention from clean to masked tokens (no leakage)
    
    2. Single forward pass that simultaneously:
       - Validates draft tokens (autoregressive head)
       - Generates new draft tokens (diffusion head)
    """
    
    def __init__(self, config: TiDARQwen2Config):
        super().__init__(config)
        # Replace all decoder layers with TiDAR versions
        self.layers = nn.ModuleList(
            [TiDARDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    
    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask

    def _create_hybrid_attention_mask_train(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[List[List[KVCache]]],
    ) -> torch.Tensor:
        """
        Create TiDAR's hybrid attention mask.
        
        The hybrid attention mask has the following structure for a sequence of length 2S:
        
        ```
        [ Causal Mask (S×S)    | No Attention (S×S)  ]
        [ Full Attention (S×S) | Block-wise Bi (S×S) ]
        ```
        
        Where:
        - Causal Mask (S×S): Standard autoregressive mask for clean tokens
        - Block-wise Bi (S×S): Block-wise bidirectional mask for masked tokens
        - Full Attention (S×S): Masked tokens can attend to all clean tokens
        - No Attention (S×S): Clean tokens cannot attend to masked tokens
        
        Args:
            attention_mask: Original attention mask (for padding)
            input_tensor: Input hidden states [batch_size, seq_length, hidden_size]
            cache_position: Position cache
            past_key_values: Past key values cache
            
        Returns:
            Hybrid attention mask [batch_size, 1, seq_length, seq_length]
        """
        batch_size, seq_length = input_tensor.shape[:2]
        clean_length = int(seq_length * self.config.clean_ratio)
        masked_length = seq_length - clean_length
        
        # If not a TiDAR-formatted sequence, use standard causal mask
        if clean_length == 0 or masked_length == 0:
            return super()._update_causal_mask(
                attention_mask, input_tensor, cache_position, past_key_values
            )
        
        min_dtype = torch.finfo(input_tensor.dtype).min
        device = input_tensor.device
        
        # Create hybrid mask matrix
        hybrid_mask = torch.full(
            (seq_length, seq_length), fill_value=min_dtype, dtype=input_tensor.dtype, device=device
        )
        
        # 1. Top-left: Causal mask for clean tokens (autoregressive mode)
        if clean_length > 0:
            causal_mask = torch.triu(
                torch.ones(clean_length, clean_length, dtype=torch.bool, device=device),
                diagonal=1
            )
            hybrid_mask[:clean_length, :clean_length] = causal_mask.float() * min_dtype
        
        # 2. Bottom-right: Block-wise bidirectional mask for masked tokens (diffusion mode)
        # 3. Bottom-left: Full attention from masked to clean tokens (conditioning)
        if masked_length > 0:
            block_size = self.config.block_size
            for i in range(clean_length, seq_length, block_size):
                block_end = min(i + block_size, seq_length)
                # 2. Bottom-right:
                # Within each block, tokens can attend to each other (bidirectional)
                hybrid_mask[i:block_end, i:block_end] = 0
                # 3. Bottom-left:
                if i > clean_length: # start from second block
                    hybrid_mask[i:block_end, :i-clean_length] = 0
        
        # Expand to batch dimension
        hybrid_mask = hybrid_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Merge with user-provided attention mask (for padding)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            
            # Convert attention_mask to same dtype as hybrid_mask
            attention_mask = attention_mask.to(hybrid_mask.dtype)
            hybrid_mask = hybrid_mask + attention_mask
        
        return hybrid_mask
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[List[KVCache]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ):
        """
        Forward pass with TiDAR support.
        
        Supports two input formats:
        1. Standard format: [token1, token2, ..., tokenN]
        2. TiDAR format: [clean_token1, ..., clean_tokenS, MASK, MASK, ..., MASK]
        
        For TiDAR format:
        - First S tokens are clean tokens (autoregressive mode)
        - Last S tokens are masked tokens (diffusion mode)
        - The hybrid attention mask ensures proper attention patterns
        """
        # Detect if this is TiDAR format
        if self.config.use_tidar and input_ids is not None:
            assert position_ids is not None
            assert attention_mask is not None
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_ids.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if cache_position is None:
            cache_position = torch.arange(
                seq_length_with_past, seq_length_with_past + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        causal_mask = None
        if self.config.use_tidar:
            if self.config.is_training:
                causal_mask = self._create_hybrid_attention_mask_train(
                    attention_mask, inputs_embeds, cache_position, past_key_values
                )
            else:
                causal_mask = attention_mask + torch.ones_like(attention_mask, device=inputs_embeds.device)
        else:
            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class TiDARQwen2ForCausalLM(Qwen2ForCausalLM):
    """
    TiDAR-Qwen2 for Causal Language Modeling.
    
    This model combines:
    1. Autoregressive head: Validates draft tokens
    2. Diffusion head: Generates new draft tokens
    3. Single forward pass: Both tasks in parallel
    """
    config_class = TiDARQwen2Config 
    _no_split_modules = ["TiDARDecoderLayer"]

    def __init__(self, config: TiDARQwen2Config):
        super().__init__(config)
        self.model = TiDARModel(config)
        
        # TiDAR-specific components
        self.tidar_config = config.tidar_config
        print(self.tidar_config)
        
        # Additional heads for TiDAR (can be added later)
        # self.validation_head = nn.Linear(config.hidden_size, config.vocab_size)
        # self.draft_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        block_size: int = 8,
        clean_ratio: float = 0.5,
        use_tidar: bool = True,
        is_training: bool = True,
        **kwargs
    ):
        """
        Load TiDAR model from either a TiDAR checkpoint or Qwen2.5 pretrained weights.
        
        This method automatically detects the model type:
        - If it's a TiDAR checkpoint (has TiDARQwen2Config), loads directly
        - If it's a Qwen2.5 model, converts it to TiDAR format
        
        Args:
            pretrained_model_name_or_path: Path to model checkpoint or model identifier
            block_size: Block size for block-wise bidirectional attention (only used when converting from Qwen2.5)
            clean_ratio: Ratio of clean tokens in the input sequence (only used when converting from Qwen2.5)
            use_tidar: Whether to enable TiDAR mode (only used when converting from Qwen2.5)
            **kwargs: Additional arguments passed to transformers.from_pretrained
            
        Returns:
            TiDARQwen2ForCausalLM: Model loaded from checkpoint
        """
        
        # Try to load configuration to detect model type
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
            # Check if it's already a TiDAR model
            if isinstance(config, TiDARQwen2Config) or config.model_type == "tidar_qwen2":
                # Direct loading of TiDAR checkpoint
                if kwargs.get('local_rank', 0) == 0 or not torch.distributed.is_initialized():
                    print(f"Loading TiDAR checkpoint from: {pretrained_model_name_or_path}")
                
                # Use parent class's from_pretrained for direct loading
                model = super(TiDARQwen2ForCausalLM, cls).from_pretrained(
                    pretrained_model_name_or_path,
                    **kwargs
                )
                return model
            
            else:
                # Converting from Qwen2.5 to TiDAR
                if kwargs.get('local_rank', 0) == 0 or not torch.distributed.is_initialized():
                    print(f"Converting Qwen2.5 model to TiDAR from: {pretrained_model_name_or_path}")
                
                # Convert to TiDAR configuration
                tidar_config = TiDARQwen2Config.from_qwen2_config(
                    config,
                    block_size=block_size,
                    clean_ratio=clean_ratio,
                    use_tidar=use_tidar,
                    is_training=is_training,
                )
                
                # Initialize TiDAR model with converted config
                model = cls(tidar_config)
                
                # Load Qwen2.5 weights and transfer to TiDAR model
                qwen_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    **kwargs
                )
                
                # Transfer weights (strict=False allows for architecture differences)
                model.load_state_dict(qwen_model.state_dict(), strict=False)
                
                return model
                
        except Exception as e:
            # Fallback: try direct loading as TiDAR checkpoint
            if kwargs.get('local_rank', 0) == 0 or not torch.distributed.is_initialized():
                print(f"Attempting direct loading as TiDAR checkpoint: {pretrained_model_name_or_path}")
            
            return super(TiDARQwen2ForCausalLM, cls).from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )
    
    def save_pretrained(
        self,
        save_directory: str,
        save_config: bool = True,
        **kwargs
    ):
        """
        Save TiDAR model and configuration to a directory.
        
        This creates a proper TiDAR checkpoint that can be loaded directly
        without needing to convert from Qwen2.5 again.
        
        Args:
            save_directory: Directory to save the model
            save_config: Whether to save the configuration file
            **kwargs: Additional arguments passed to parent's save_pretrained
            
        Example:
            ```python
            # Convert from Qwen2.5 and save
            model = TiDARQwen2ForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                block_size=8,
                clean_ratio=0.5
            )
            model.save_pretrained("./tidar_checkpoints/initial")
            
            # Later, load directly
            model = TiDARQwen2ForCausalLM.from_pretrained("./tidar_checkpoints/initial")
            ```
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save using parent class method (saves weights and config)
        super().save_pretrained(save_directory, save_config=save_config, **kwargs)
        
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(f"✓ TiDAR model saved to: {save_directory}")
        else:
            print(f"✓ TiDAR model saved to: {save_directory}")
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def naivegenerate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        tokenizer = None,
        **kwargs
    ):
        self.model.config.use_tidar = False
        logits_processor = self._prepare_logits_processor(temperature, top_p, top_k)

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_new_tokens):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token+=1

            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
        return input_ids

    @torch.no_grad()
    def tidar_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        block_size: int = None,
        tokenizer = None,
        **kwargs
    ):
        """
        TiDAR-specific generation with parallel validation and drafting.
        
        This implements TiDAR's core mechanism:
        1. Prefill: Process initial prompt and generate first draft tokens
        2. Generation loop:
           a. Build input: [current_draft_tokens, mask_block_0, mask_block_1, ..., mask_block_K]
              - Each mask_block corresponds to a potential acceptance scenario
           b. Single forward pass with KV cache reuse:
              - Validation: validates draft tokens (autoregressive)
              - Parallel drafting: generates K+1 conditional next drafts
           c. Accept/reject based on validation (rejection sampling)
           d. Table-lookup: select next draft from the appropriate conditional branch
        3. Update KV cache and repeat
        
        Args:
            input_ids: Initial input tokens [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            draft_len: Number of draft tokens per iteration (default: block_size)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token ids [batch_size, seq_len + num_generated]
        """
        
        self.model.config.use_tidar = True
        # Get draft length from config if not specified
        if block_size is None:
            block_size = self.config.block_size
        
        # Prepare logits processor for sampling
        logits_processor = self._prepare_logits_processor(temperature, top_p, top_k)
        
        # Initialize variables
        batch_size = input_ids.shape[0]
        assert batch_size == 1

        device = input_ids.device
        accepted_tokens = input_ids.clone()
        current_draft_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        # [MODIFIED] Initialize custom KVCache
        past_key_values, past_key_values_data_list, current_length_data = initialize_past_key_values(self)
        
        # Get mask token ID
        mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id else 151643
        eos_token_id = self.config.eos_token_id
        
        # Track number of generated tokens
        num_generated = 0
        input_len = accepted_tokens.shape[1]
        
        # Prefill: Single forward pass with [prompt, MASK_BLOCK]
        # This simultaneously:
        # 1. Populates KV cache for the prompt
        # 2. Generates initial draft tokens from the mask block
        mask_block = torch.full(
            (batch_size, block_size),
            mask_token_id,
            dtype=torch.long,
            device=device
        )
        prefill_input = torch.cat([accepted_tokens, mask_block], dim=1)
        
        # Position IDs: sequential for the entire prefill input
        prefill_position_ids = torch.arange(
            prefill_input.shape[1],
            dtype=torch.long,
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        # prefill_position_ids[:, input_len:] += 1

        # Forward pass
        prefill_attention_mask = self._create_tidar_attention_mask(
            current_input_len=input_len,
            num_mask_blocks=1,
            mask_block_size=block_size,
            kv_len=0,
            batch_size=batch_size,
            is_prefill=True,
            device=device,
            dtype=self.dtype
        )
        prefill_outputs = self.model(
            input_ids=prefill_input,
            attention_mask=prefill_attention_mask,
            position_ids=prefill_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        prefill_logits = self.lm_head(prefill_outputs.last_hidden_state)

        # Extract draft logits from mask positions and sample initial draft
        prefill_output_logits = prefill_logits[:, input_len:, :]
        prefill_output_tokens = self._sample_from_logits(
            prefill_output_logits,
            block_size,
            logits_processor
        )
        prefill_output_token = prefill_output_tokens[:, 0]
        # accepted_tokens = torch.cat([accepted_tokens, prefill_output_token.unsqueeze(0)], dim=1)
        # num_generated += 1
        current_draft_tokens = prefill_output_tokens

        # Reset the current_length of each KVCache to input_len
        rollback_past_key_values(past_key_values, input_len)
        
        # Main generation loop
        while num_generated < max_new_tokens:
            # Step 1: Forward pass with current draft + multiple mask blocks
            # This returns validation_logits and all conditional draft_logits
            validation_logits, all_draft_logits = self._forward_with_conditional_masks(
                current_draft_tokens,
                past_key_values,
                block_size,
                mask_token_id
            )
            
            # Step 2: Validate current draft tokens (rejection sampling)
            accept_length, resampled_token = self._validate_draft_tokens(
                current_draft_tokens,
                validation_logits,
                logits_processor,
                device
            )
            
            # Step 3: Update accepted tokens
            if accept_length > 0:
                accepted_tokens = torch.cat([
                    accepted_tokens,
                    current_draft_tokens[:, :accept_length]
                ], dim=1)
                num_generated += accept_length
            # if resampled_token is not None:
            #     accepted_tokens = torch.cat([accepted_tokens, resampled_token], dim=1)
            #     num_generated += 1
            
            # Step 4: Table-lookup to select next draft tokens
            # Select from all_draft_logits based on actual_accept_length
            next_draft_logits = all_draft_logits[accept_length - 1]  # Shape: [batch, draft_len, vocab]
            # current_draft_tokens = torch.cat([resampled_token, self._sample_from_logits(
            #     next_draft_logits,
            #     block_size,
            #     logits_processor
            # )], dim=1)

            current_draft_tokens = self._sample_from_logits(
                next_draft_logits,
                block_size,
                logits_processor
            )

            
            # Step 5: Update KV cache
            # We need to update past_key_values to reflect the newly accepted tokens
            # This requires a forward pass with the accepted tokens
            if accept_length != block_size:
                rollback_past_key_values(past_key_values, accepted_tokens.shape[1])

            # Step 6: Check termination conditions
            if eos_token_id is not None:
                if eos_token_id in accepted_tokens[0, input_len:].tolist():
                    break
            
            if num_generated >= max_new_tokens:
                break
        
        return accepted_tokens
    
    def _prepare_logits_processor(self, temperature, top_p, top_k):
        """Prepare logits processor for sampling."""
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
        
        processor_list = LogitsProcessorList()
        
        if temperature > 1e-5:
            if temperature >= 1e-5 and temperature != 1.0:
                processor_list.append(TemperatureLogitsWarper(temperature))
            if 1e-8 <= top_p < 1.0:
                processor_list.append(TopPLogitsWarper(top_p))
            if top_k > 0:
                processor_list.append(TopKLogitsWarper(top_k))
            return processor_list
        else:
            return None
    
    def _validate_draft_tokens(self, draft_tokens, validation_logits, logits_processor, device):
        """
        Validate draft tokens using rejection sampling.
        
        Args:
            draft_tokens: Draft tokens to validate [batch_size, draft_len]
            validation_logits: Logits for validation [batch_size, draft_len, vocab_size]
            logits_processor: Logits processor for sampling
            device: Device to use
            
        Returns:
            accept_length: Number of accepted tokens
            resampled_token: Resampled token if rejection occurred, else None
        """
        
        batch_size, current_input_length = draft_tokens.shape
        accept_length = 1
        last_verified_token = None
        
        for i in range(current_input_length):
            val_logits = validation_logits[:, i, :]
            if logits_processor is not None:
                val_logits = logits_processor(None, val_logits)
            probs = torch.softmax(val_logits, dim=-1)

            if logits_processor is None:
                # Greedy: check if draft token matches argmax
                predicted_token = val_logits.argmax(dim=-1, keepdim=True)
                if i == current_input_length - 1:
                    last_verified_token = None
                else:
                    draft_token = draft_tokens[:, i+1]
                    if (draft_token == predicted_token).all():
                        accept_length += 1
                    else:
                        # Rejection: use the predicted token
                        last_verified_token = predicted_token
                        break
            else:
                if i == current_input_length - 1:
                    last_verified_token = None
                else:
                    # Sampling-based rejection sampling
                    draft_token_id = draft_token[0, 0].item()
                    p_draft = probs[0, draft_token_id].item()
                    
                    # Simplified rejection sampling: accept with probability p_draft
                    r = random.random()
                    if r <= p_draft:
                        accept_length += 1
                    else:
                        # Rejection: resample from adjusted distribution
                        # Set the rejected token's probability to 0 and renormalize
                        probs[0, draft_token_id] = 0
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        last_verified_token = torch.multinomial(probs, 1)
                        break
        
        return accept_length, last_verified_token
    
    def _forward_with_conditional_masks(
        self,
        current_draft_tokens: torch.LongTensor,
        past_key_values: KVCache,
        mask_block_size: int,
        mask_token_id: int
    ):
        """
        Forward pass with multiple conditional mask blocks for parallel drafting.
        
        This performs the core TiDAR operation: validating current draft tokens
        while simultaneously generating (draft_len + 1) conditional next drafts.
        
        Args:
            current_draft_tokens: Current draft tokens to validate [batch, draft_len]
            past_key_values: KV cache containing accepted tokens
            draft_len: Number of draft tokens per block
            mask_token_id: ID of the mask token
            
        Returns:
            validation_logits: Logits for validating current draft [batch, draft_len, vocab]
            all_draft_logits: List of draft logits for each acceptance scenario
                             [(batch, draft_len, vocab)] * (draft_len + 1)
        """
        batch_size, input_token_length = current_draft_tokens.shape
        device = current_draft_tokens.device
        num_scenarios = mask_block_size  # 0 to draft_len accepted tokens
        
        # Build input: [current_draft_tokens, mask_block_0, mask_block_1, ..., mask_block_K]
        # Each mask_block has draft_len MASK tokens
        mask_blocks = torch.full(
            (batch_size, num_scenarios * mask_block_size),
            mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        input_ids = torch.cat([current_draft_tokens, mask_blocks], dim=1)
        
        # Build position_ids
        # current_draft_tokens: positions start from KV cache length
        # mask_block_j: positions start from KV cache length + j
        kv_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else past_key_values[0][0].shape[2]
        
        position_ids = []
        # Positions for current_draft_tokens
        draft_positions = torch.arange(kv_len, kv_len + mask_block_size, dtype=torch.long, device=device)
        position_ids.append(draft_positions)
        
        # Positions for each mask_block_j
        for j in range(num_scenarios):
            block_start = kv_len + j + 1
            block_positions = torch.arange(
                block_start,
                block_start + mask_block_size,
                dtype=torch.long,
                device=device
            )
            position_ids.append(block_positions)
        
        position_ids = torch.cat(position_ids, dim=0).unsqueeze(0).expand(batch_size, -1)
        
        # Build complex attention mask
        # This is the crucial part that implements TiDAR's parallel drafting
        attention_mask = self._create_tidar_attention_mask(
            current_input_len=input_token_length,
            num_mask_blocks=num_scenarios,
            mask_block_size=mask_block_size,
            kv_len=kv_len,
            batch_size=batch_size,
            is_prefill=False,
            device=device,
            dtype=self.dtype
        )
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        # Get logits
        logits = self.lm_head(outputs.last_hidden_state)
        
        # Extract validation logits (for current_draft_tokens)
        validation_logits = logits[:, :input_token_length, :]
        
        # Extract draft logits for each scenario
        all_draft_logits = []
        for j in range(num_scenarios):
            start_idx = input_token_length + j * mask_block_size
            end_idx = start_idx + mask_block_size
            scenario_logits = logits[:, start_idx:end_idx, :]
            all_draft_logits.append(scenario_logits)
        
        return validation_logits, all_draft_logits
    
    def _create_tidar_attention_mask(
        self,
        current_input_len: int,
        num_mask_blocks: int,
        mask_block_size: int,
        kv_len: int,
        batch_size: int,
        is_prefill: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        """
        Create complex attention mask for TiDAR's conditional parallel drafting.
        
        The mask structure:
        - current_draft_tokens: attend causally to each other and to KV cache
        - mask_block_j: attends to KV cache + first j tokens of current_draft + itself (bidirectional)
        - No cross-attention between different mask blocks
        
        Args:
            current_draft_len: Length of current draft tokens
            num_mask_blocks: Number of conditional mask blocks (draft_len + 1)
            draft_len: Tokens per mask block
            kv_len: Length of KV cache
            batch_size: Batch size
            device: Device
            
        Returns:
            attention_mask: [batch, 1, total_len, total_len + kv_len]
        """
        total_len = current_input_len + num_mask_blocks * mask_block_size
        total_seq_len = kv_len + total_len
        
        # Create mask matrix
        # We'll use 0 for allowed attention, large negative for blocked
        min_dtype = torch.finfo(dtype).min
        mask = torch.full(
            (total_len, total_seq_len),
            min_dtype,
            dtype=dtype,
            device=device
        )
        
        # 1. All positions can attend to KV cache
        mask[:, :kv_len] = 0
        
        # 2. current_draft_tokens: causal attention to themselves
        for i in range(current_input_len):
            # Can attend to KV cache (already set above)
            # Can attend to previous draft tokens
            mask[i, kv_len:kv_len + i + 1] = 0
        
        # 3. Each mask_block_j
        if is_prefill:
            assert num_mask_blocks == 1
            mask[current_input_len:current_input_len+mask_block_size, :] = 0
        else:
            for j in range(num_mask_blocks):
                block_start = current_input_len + j * mask_block_size
                block_end = block_start + mask_block_size
                
                # Can attend to KV cache (already set)
                # Can attend to first j tokens of current_draft
                mask[block_start:block_end, kv_len:kv_len + j + 1] = 0
                
                # Block-wise bidirectional attention within the block
                mask[block_start:block_end, kv_len + block_start:kv_len + block_end] = 0
        
        # Expand to batch dimension
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        return mask
    
    def _sample_from_logits(
        self,
        logits: torch.FloatTensor,
        num_tokens: int,
        logits_processor
    ):
        """
        Sample tokens from logits.
        
        Args:
            logits: Logits to sample from [batch, num_tokens, vocab]
            num_tokens: Number of tokens to sample
            logits_processor: Logits processor for sampling
            
        Returns:
            tokens: Sampled tokens [batch, num_tokens]
        """
        batch_size = logits.shape[0]
        tokens = []
        
        for i in range(num_tokens):
            token_logits = logits[:, i, :]
            
            if logits_processor is not None:
                token_logits = logits_processor(None, token_logits)
                probs = torch.softmax(token_logits, dim=-1)
                token = torch.multinomial(probs, 1)
            else:
                token = token_logits.argmax(dim=-1, keepdim=True)
            
            tokens.append(token)
        
        return torch.cat(tokens, dim=1)


AutoConfig.register("tidar_qwen2", TiDARQwen2Config)
AutoModelForCausalLM.register(TiDARQwen2Config, TiDARQwen2ForCausalLM)