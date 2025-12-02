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
        draft_len: int = None,
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
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
        
        # Get draft length from config if not specified
        if draft_len is None:
            draft_len = self.config.block_size
        
        # Prepare logits processor for sampling
        logits_processor = self._prepare_logits_processor(temperature, top_p, top_k)
        
        # Initialize variables
        batch_size = input_ids.shape[0]
        device = input_ids.device
        accepted_tokens = input_ids.clone()
        current_draft_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        past_key_values = None
        
        # Get mask token ID
        mask_token_id = getattr(self.config, 'mask_token_id', self.config.vocab_size - 1)
        
        # Get EOS token ID
        eos_token_id = self.config.eos_token_id
        
        # Track number of generated tokens
        num_generated = 0
        input_len = accepted_tokens.shape[1]
        
        # Prefill: Single forward pass with [prompt, MASK_BLOCK]
        # This simultaneously:
        # 1. Populates KV cache for the prompt
        # 2. Generates initial draft tokens from the mask block
        mask_block = torch.full(
            (batch_size, draft_len),
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
        
        # Forward pass
        prefill_outputs = self.model(
            input_ids=prefill_input,
            position_ids=prefill_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = prefill_outputs.past_key_values
        
        # Get logits
        prefill_logits = self.lm_head(prefill_outputs.last_hidden_state)
        
        # Extract draft logits from mask positions and sample initial draft
        draft_logits = prefill_logits[:, -draft_len:, :]
        current_draft_tokens = self._sample_from_logits(
            draft_logits,
            draft_len,
            logits_processor
        )
        
        # Main generation loop
        while num_generated < max_new_tokens:
            # Step 1: Forward pass with current draft + multiple mask blocks
            # This returns validation_logits and all conditional draft_logits
            validation_logits, all_draft_logits = self._forward_with_conditional_masks(
                current_draft_tokens,
                past_key_values,
                draft_len,
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
            
            # If rejection occurred, add resampled token
            if resampled_token is not None:
                accepted_tokens = torch.cat([accepted_tokens, resampled_token], dim=1)
                num_generated += 1
                actual_accept_length = accept_length  # For table lookup
            else:
                # All draft tokens accepted
                actual_accept_length = accept_length
            
            # Step 4: Table-lookup to select next draft tokens
            # Select from all_draft_logits based on actual_accept_length
            next_draft_logits = all_draft_logits[actual_accept_length]  # Shape: [batch, draft_len, vocab]
            current_draft_tokens = self._sample_from_logits(
                next_draft_logits,
                draft_len,
                logits_processor
            )
            
            # Step 5: Update KV cache
            # We need to update past_key_values to reflect the newly accepted tokens
            # This requires a forward pass with the accepted tokens
            if accept_length > 0 or resampled_token is not None:
                # Determine which tokens to add to KV cache
                if accept_length > 0 and resampled_token is not None:
                    new_tokens = torch.cat([
                        current_draft_tokens[:, :accept_length],
                        resampled_token
                    ], dim=1)
                elif accept_length > 0:
                    new_tokens = current_draft_tokens[:, :accept_length]
                else:
                    new_tokens = resampled_token
                
                # Update KV cache with accepted tokens
                update_outputs = self.model(
                    input_ids=new_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = update_outputs.past_key_values
            
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
        import random
        
        batch_size = draft_tokens.shape[0]
        draft_len = draft_tokens.shape[1]
        accept_length = 0
        resampled_token = None
        
        for i in range(draft_len):
            # Get the current draft token
            draft_token = draft_tokens[:, i:i+1]
            
            # Get validation logits for this position
            val_logits = validation_logits[:, i, :]
            
            # Apply logits processor if provided
            if logits_processor is not None:
                val_logits = logits_processor(None, val_logits)
            
            # Get probability distribution
            probs = torch.softmax(val_logits, dim=-1)
            
            # Greedy or sampling validation
            if logits_processor is None:
                # Greedy: check if draft token matches argmax
                predicted_token = val_logits.argmax(dim=-1, keepdim=True)
                if (draft_token == predicted_token).all():
                    accept_length += 1
                else:
                    # Rejection: use the predicted token
                    resampled_token = predicted_token
                    break
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
                    resampled_token = torch.multinomial(probs, 1)
                    break
        
        return accept_length, resampled_token
    
    def _generate_draft_tokens(self, draft_logits, draft_len, logits_processor, device):
        """
        Generate new draft tokens from draft logits.
        
        Args:
            draft_logits: Logits for draft generation [batch_size, num_masks, vocab_size]
            draft_len: Number of draft tokens to generate
            logits_processor: Logits processor for sampling
            device: Device to use
            
        Returns:
            new_draft_tokens: New draft tokens [batch_size, draft_len]
        """
        batch_size = draft_logits.shape[0]
        
        # Generate draft tokens from the first draft_len positions
        # Each position should generate one token
        new_draft_tokens = []
        
        for i in range(min(draft_len, draft_logits.shape[1])):
            logits = draft_logits[:, i, :]
            
            if logits_processor is not None:
                logits = logits_processor(None, logits)
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            
            new_draft_tokens.append(token)
        
        if len(new_draft_tokens) > 0:
            new_draft_tokens = torch.cat(new_draft_tokens, dim=1)
        else:
            new_draft_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        return new_draft_tokens
    
    def _truncate_kv_cache(self, past_key_values, keep_length):
        """
        Truncate KV cache to only keep the first keep_length positions.
        
        Args:
            past_key_values: Past key values cache
            keep_length: Number of positions to keep
            
        Returns:
            Truncated past_key_values
        """
        if past_key_values is None:
            return None
        
        # For DynamicCache or Cache objects
        if hasattr(past_key_values, 'crop'):
            # Use the crop method if available (transformers >= 4.36)
            past_key_values.crop(keep_length)
            return past_key_values
        
        # For tuple-based cache (legacy)
        truncated_cache = []
        for layer_cache in past_key_values:
            truncated_layer = []
            for cache_tensor in layer_cache:
                # Truncate along the sequence dimension (typically dim=2)
                truncated_layer.append(cache_tensor[:, :, :keep_length, :])
            truncated_cache.append(tuple(truncated_layer))
        
        return tuple(truncated_cache)
    
    def _forward_with_masks(
        self,
        current_draft_tokens: torch.LongTensor,
        past_key_values: Cache,
        draft_len: int,
        mask_token_id: int,
        accept_length: int
    ):
        """
        Forward pass with mask tokens to generate draft predictions.
        
        Args:
            current_draft_tokens: Current draft tokens being validated [batch, draft_len]
            past_key_values: KV cache containing accepted tokens
            draft_len: Number of draft tokens to generate
            mask_token_id: ID of the mask token
            accept_length: Number of accepted tokens from current draft (for positioning)
            
        Returns:
            draft_logits: Logits for the mask positions [batch, draft_len, vocab]
        """
        batch_size = current_draft_tokens.shape[0] if current_draft_tokens.numel() > 0 else past_key_values[0][0].shape[0]
        device = current_draft_tokens.device if current_draft_tokens.numel() > 0 else past_key_values[0][0].device
        
        # Create mask block
        mask_tokens = torch.full(
            (batch_size, draft_len),
            mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Combine current draft (if any) with mask tokens
        if current_draft_tokens.numel() > 0:
            input_ids = torch.cat([current_draft_tokens, mask_tokens], dim=1)
        else:
            input_ids = mask_tokens
        
        # Calculate position IDs
        # They should start from the KV cache length + accept_length
        kv_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else past_key_values[0][0].shape[2]
        start_pos = kv_len + accept_length
        position_ids = torch.arange(
            start_pos,
            start_pos + input_ids.shape[1],
            dtype=torch.long,
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,  # We don't need to update KV cache here
        )
        
        # Get logits
        logits = self.lm_head(outputs.last_hidden_state)
        
        # Extract draft logits (from mask positions)
        draft_start = current_draft_tokens.shape[1] if current_draft_tokens.numel() > 0 else 0
        draft_logits = logits[:, draft_start:, :]
        
        return draft_logits
    
    def _forward_with_conditional_masks(
        self,
        current_draft_tokens: torch.LongTensor,
        past_key_values: Cache,
        draft_len: int,
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
        batch_size = current_draft_tokens.shape[0]
        device = current_draft_tokens.device
        num_scenarios = draft_len + 1  # 0 to draft_len accepted tokens
        
        # Build input: [current_draft_tokens, mask_block_0, mask_block_1, ..., mask_block_K]
        # Each mask_block has draft_len MASK tokens
        mask_blocks = torch.full(
            (batch_size, num_scenarios * draft_len),
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
        draft_positions = torch.arange(kv_len, kv_len + draft_len, dtype=torch.long, device=device)
        position_ids.append(draft_positions)
        
        # Positions for each mask_block_j
        for j in range(num_scenarios):
            block_start = kv_len + j
            block_positions = torch.arange(
                block_start,
                block_start + draft_len,
                dtype=torch.long,
                device=device
            )
            position_ids.append(block_positions)
        
        position_ids = torch.cat(position_ids, dim=0).unsqueeze(0).expand(batch_size, -1)
        
        # Build complex attention mask
        # This is the crucial part that implements TiDAR's parallel drafting
        attention_mask = self._create_tidar_attention_mask(
            current_draft_len=draft_len,
            num_mask_blocks=num_scenarios,
            draft_len=draft_len,
            kv_len=kv_len,
            batch_size=batch_size,
            device=device
        )
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=False,  # We don't update KV cache here
        )
        
        # Get logits
        logits = self.lm_head(outputs.last_hidden_state)
        
        # Extract validation logits (for current_draft_tokens)
        validation_logits = logits[:, :draft_len, :]
        
        # Extract draft logits for each scenario
        all_draft_logits = []
        for j in range(num_scenarios):
            start_idx = draft_len + j * draft_len
            end_idx = start_idx + draft_len
            scenario_logits = logits[:, start_idx:end_idx, :]
            all_draft_logits.append(scenario_logits)
        
        return validation_logits, all_draft_logits
    
    def _create_tidar_attention_mask(
        self,
        current_draft_len: int,
        num_mask_blocks: int,
        draft_len: int,
        kv_len: int,
        batch_size: int,
        device: torch.device
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
        total_len = current_draft_len + num_mask_blocks * draft_len
        total_seq_len = kv_len + total_len
        
        # Create mask matrix
        # We'll use 0 for allowed attention, large negative for blocked
        min_dtype = torch.finfo(torch.float32).min
        mask = torch.full(
            (total_len, total_seq_len),
            min_dtype,
            dtype=torch.float32,
            device=device
        )
        
        # 1. All positions can attend to KV cache
        mask[:, :kv_len] = 0
        
        # 2. current_draft_tokens: causal attention to themselves
        for i in range(current_draft_len):
            # Can attend to KV cache (already set above)
            # Can attend to previous draft tokens
            mask[i, kv_len:kv_len + i + 1] = 0
        
        # 3. Each mask_block_j
        for j in range(num_mask_blocks):
            block_start = current_draft_len + j * draft_len
            block_end = block_start + draft_len
            
            # Can attend to KV cache (already set)
            # Can attend to first j tokens of current_draft
            if j > 0:
                mask[block_start:block_end, kv_len:kv_len + j] = 0
            
            # Block-wise bidirectional attention within the block
            for i in range(block_start, block_end):
                mask[i, kv_len + block_start:kv_len + block_end] = 0
        
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
