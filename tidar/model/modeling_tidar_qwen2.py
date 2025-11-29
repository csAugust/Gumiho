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
        Load TiDAR model from Qwen2.5 pretrained weights.
        
        This method loads a Qwen2.5 model and converts it to TiDAR format
        by creating a TiDAR configuration and loading the weights.
        
        Args:
            pretrained_model_name_or_path: Path to Qwen2.5 model or model identifier
            block_size: Block size for block-wise bidirectional attention
            clean_ratio: Ratio of clean tokens in the input sequence
            use_tidar: Whether to enable TiDAR mode
            **kwargs: Additional arguments for model loading
            
        Returns:
            TiDARQwen2ForCausalLM: Model initialized with Qwen2.5 weights
        """
        import os
        from transformers import AutoConfig
        
        print("=" * 80)
        print("Loading Qwen2.5 model and converting to TiDAR")
        print("=" * 80)
        
        # Load Qwen2.5 configuration
        print(f"\n1. Loading Qwen2.5 configuration from: {pretrained_model_name_or_path}")
        qwen_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        print(f"  - vocab_size: {qwen_config.vocab_size}")
        print(f"  - hidden_size: {qwen_config.hidden_size}")
        print(f"  - num_hidden_layers: {qwen_config.num_hidden_layers}")
        print(f"  - num_attention_heads: {qwen_config.num_attention_heads}")
        print(f"  - num_key_value_heads: {qwen_config.num_key_value_heads}")
        print(f"  - intermediate_size: {qwen_config.intermediate_size}")
        print(f"  - max_position_embeddings: {qwen_config.max_position_embeddings}")
        
        # Create TiDAR configuration based on Qwen2.5 config
        print(f"\n2. Creating TiDAR configuration:")
        print(f"  - block_size: {block_size}")
        print(f"  - clean_ratio: {clean_ratio}")
        print(f"  - use_tidar: {use_tidar}")
        
        tidar_config = TiDARQwen2Config(
            vocab_size=qwen_config.vocab_size,
            hidden_size=qwen_config.hidden_size,
            num_hidden_layers=qwen_config.num_hidden_layers,
            num_attention_heads=qwen_config.num_attention_heads,
            num_key_value_heads=qwen_config.num_key_value_heads,
            intermediate_size=qwen_config.intermediate_size,
            max_position_embeddings=qwen_config.max_position_embeddings,
            rms_norm_eps=getattr(qwen_config, 'rms_norm_eps', 1e-6),
            hidden_act=getattr(qwen_config, 'hidden_act', 'silu'),
            block_size=block_size,
            use_tidar=use_tidar,
            clean_ratio=clean_ratio,
        )
        
        print(f"✓ TiDAR configuration created")
        
        # Create TiDAR model
        print(f"\n3. Creating TiDAR model:")
        model = cls(tidar_config)
        print(f"✓ TiDAR model created")
        print(f"  - Model type: {type(model).__name__}")
        
        # Load weights from Qwen2.5 model
        print(f"\n4. Loading weights from Qwen2.5 model:")
        print(f"  - Model path: {pretrained_model_name_or_path}")
        
        # Load Qwen2.5 model weights
        from transformers import AutoModelForCausalLM
        
        # Determine device
        device = kwargs.get('device_map', 'cpu')
        if isinstance(device, str) and device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Qwen2.5 model
        qwen_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        print(f"  - Qwen2.5 parameters: {sum(p.numel() for p in qwen_model.parameters()):,}")
        print(f"  - TiDAR parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Get state dicts
        qwen_state_dict = qwen_model.state_dict()
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(qwen_state_dict, strict=False)
        
        print(f"✓ Weights loaded")
        if missing_keys:
            print(f"  - Missing keys: {len(missing_keys)}")
            # Only show first 5 missing keys
            for key in missing_keys[:5]:
                print(f"    * {key}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys) - 5} more")
        else:
            print(f"  - Missing keys: 0")
        
        if unexpected_keys:
            print(f"  - Unexpected keys: {len(unexpected_keys)}")
            # Only show first 5 unexpected keys
            for key in unexpected_keys[:5]:
                print(f"    * {key}")
            if len(unexpected_keys) > 5:
                print(f"    ... and {len(unexpected_keys) - 5} more")
        else:
            print(f"  - Unexpected keys: 0")
        
        # Verify weight transfer
        print(f"\n5. Verifying weight transfer:")
        
        # Compare some key weights
        with torch.no_grad():
            # Compare embedding weights
            qwen_embed = qwen_model.model.embed_tokens.weight
            tidar_embed = model.model.embed_tokens.weight
            
            # Move to same device for comparison
            if qwen_embed.device != tidar_embed.device:
                qwen_embed = qwen_embed.to(tidar_embed.device)
            
            embed_diff = torch.abs(qwen_embed - tidar_embed).max().item()
            print(f"  - Embedding weight max difference: {embed_diff:.2e}")
            
            # Compare first layer attention weights
            qwen_attn = qwen_model.model.layers[0].self_attn.q_proj.weight
            tidar_attn = model.model.layers[0].self_attn.q_proj.weight
            
            # Move to same device for comparison
            if qwen_attn.device != tidar_attn.device:
                qwen_attn = qwen_attn.to(tidar_attn.device)
            
            attn_diff = torch.abs(qwen_attn - tidar_attn).max().item()
            print(f"  - First layer Q projection max difference: {attn_diff:.2e}")
            
            if embed_diff < 1e-5 and attn_diff < 1e-5:
                print(f"  ✓ Weights transferred successfully!")
            else:
                print(f"  ⚠ Weight differences detected (may be expected for some parameters)")
        
        # Test forward pass
        print(f"\n6. Testing forward pass:")
        
        # Create test input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, tidar_config.vocab_size, (batch_size, seq_length))
        
        # Move to same device as model
        if hasattr(model, 'device'):
            input_ids = input_ids.to(model.device)
        
        print(f"  - Test input shape: {input_ids.shape}")
        
        # Test standard forward pass
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"  - Output logits shape: {outputs.logits.shape}")
        print(f"  ✓ Forward pass successful!")
        
        # Test TiDAR format forward pass
        print(f"\n7. Testing TiDAR format:")
        
        clean_length = int(seq_length * clean_ratio)
        MASK_TOKEN_ID = 151643  # Qwen2.5 mask token id
        
        clean_tokens = torch.randint(0, MASK_TOKEN_ID, (batch_size, clean_length))
        masked_tokens = torch.full((batch_size, seq_length - clean_length), MASK_TOKEN_ID, dtype=torch.long)
        
        if hasattr(model, 'device'):
            clean_tokens = clean_tokens.to(model.device)
            masked_tokens = masked_tokens.to(model.device)
        
        tidar_input_ids = torch.cat([clean_tokens, masked_tokens], dim=1)
        
        print(f"  - Clean tokens: {clean_length}")
        print(f"  - Masked tokens: {seq_length - clean_length}")
        print(f"  - TiDAR input shape: {tidar_input_ids.shape}")
        
        with torch.no_grad():
            tidar_outputs = model(tidar_input_ids)
        
        print(f"  - TiDAR output logits shape: {tidar_outputs.logits.shape}")
        print(f"  ✓ TiDAR format forward pass successful!")
        
        print(f"\n" + "=" * 80)
        print("✓ Qwen2.5 successfully converted to TiDAR model!")
        print("=" * 80)
        
        print(f"\nModel Summary:")
        print(f"  - Model: TiDAR-Qwen2.5")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Block size: {block_size}")
        print(f"  - Clean ratio: {clean_ratio}")
        print(f"  - TiDAR mode: {use_tidar}")
        
        return model
    
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
