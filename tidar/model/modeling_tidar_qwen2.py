"""
TiDAR-Qwen2 Model Implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Any
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from .modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2RMSNorm,
    Qwen2MLP,
)
from .configuration_tidar_qwen2 import TiDARQwen2Config


class TiDARAttention(Qwen2Attention):
    """
    TiDAR Attention layer with support for hybrid attention patterns.
    
    This attention layer works with the hybrid attention mask created by TiDARModel.
    The actual attention pattern is controlled by the attention_mask passed to forward().
    """
    
    def __init__(self, config: TiDARQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.tidar_config = config.tidar_config


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
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        """
        Override to support TiDAR's hybrid attention mask.
        
        Args:
            attention_mask: Original attention mask (for padding)
            input_tensor: Input hidden states
            cache_position: Position cache
            past_key_values: Past key values cache
            output_attentions: Whether to output attentions
            
        Returns:
            Hybrid attention mask for TiDAR
        """
        # If TiDAR is disabled, use standard causal mask
        if not self.config.use_tidar:
            return super()._update_causal_mask(
                attention_mask, input_tensor, cache_position, past_key_values, output_attentions
            )
        
        return self._create_hybrid_attention_mask(
            attention_mask, input_tensor, cache_position, past_key_values
        )
    
    def _create_hybrid_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
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
        if masked_length > 0:
            block_size = self.config.block_size
            for i in range(clean_length, seq_length, block_size):
                block_end = min(i + block_size, seq_length)
                # Within each block, tokens can attend to each other (bidirectional)
                hybrid_mask[i:block_end, i:block_end] = 0
        
        # 3. Bottom-left: Full attention from masked to clean tokens (conditioning)
        if masked_length > 0 and clean_length > 0:
            hybrid_mask[clean_length:, :clean_length] = 0
        
        # 4. Top-right: No attention from clean to masked tokens (no leakage)
        if clean_length > 0 and masked_length > 0:
            hybrid_mask[:clean_length, clean_length:] = min_dtype
        
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
        past_key_values: Optional[Cache] = None,
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
            seq_length = input_ids.shape[1]
            clean_length = int(seq_length * self.config.clean_ratio)
            
            # If TiDAR format, ensure position encoding is correct
            if position_ids is None and clean_length > 0 and clean_length < seq_length:
                # Create TiDAR-style position encoding
                # Both clean and masked tokens use positions 0, 1, 2, ..., S-1
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Call parent forward
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs
        )


class TiDARQwen2ForCausalLM(Qwen2ForCausalLM):
    """
    TiDAR-Qwen2 for Causal Language Modeling.
    
    This model combines:
    1. Autoregressive head: Validates draft tokens
    2. Diffusion head: Generates new draft tokens
    3. Single forward pass: Both tasks in parallel
    """
    
    def __init__(self, config: TiDARQwen2Config):
        super().__init__(config)
        # Replace model with TiDAR version
        self.model = TiDARModel(config)
        
        # TiDAR-specific components
        self.tidar_config = config.tidar_config
        
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
        import os
        from transformers import AutoConfig, AutoModelForCausalLM
        
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
                    use_tidar=use_tidar
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
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        Prepare inputs for generation with TiDAR support.
        """
        # Call parent method
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )
        
        # TiDAR-specific input preparation can be added here
        if self.config.use_tidar and past_key_values is not None:
            # TiDAR KV cache management logic
            pass
        
        return model_inputs
    
    @torch.no_grad()
    def tidar_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        **kwargs
    ):
        """
        TiDAR-specific generation with parallel validation and drafting.
        
        This implements TiDAR's core mechanism:
        1. Build input: [accepted_tokens, draft_tokens, MASK, MASK, ...]
        2. Single forward pass:
           - Validation head validates draft tokens (autoregressive)
           - Draft head generates new draft tokens (diffusion)
        3. Accept/reject based on validation
        4. Update and repeat
        
        Args:
            input_ids: Initial input tokens
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token ids
        """
        # This will be implemented in the inference module
        # For now, fall back to standard generation
        print("Warning: TiDAR generation not yet implemented. Using standard generation.")
        return self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
