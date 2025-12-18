# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
SpAF Inference Module

This module implements the speculative decoding logic for SpAF:
- Draft generation using adapters
- Parallel verification using alignment head
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def prepare_logits_processor(temperature=1.0, top_p=0.0, top_k=0.0):
    """
    Prepare logits processor for sampling.
    
    Args:
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling parameter.
        top_k: Top-k sampling parameter.
        
    Returns:
        Logits processor function.
    """
    def logits_processor(input_ids, logits):
        if temperature > 0:
            logits = logits / temperature
            
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, int(top_k))[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        if top_p > 0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
        return logits
    
    return logits_processor


def spaf_draft(
    model,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Tuple] = None,
    num_draft_tokens: int = 5,
    cutoff_layer: Optional[int] = None,
    serial_parallel_transition: int = 2,
    logits_processor=None,
) -> Tuple[torch.LongTensor, List[torch.Tensor]]:
    """
    Generate draft tokens using SpAF adapters.
    
    Strategy:
    1. LLM computes up to cutoff layer k
    2. Use Adapter_k to predict first draft token
    3. Continue serially for 'serial_parallel_transition' tokens
    4. Switch to parallel mode using adapters at higher layers
    
    Args:
        model: SpAFModel instance.
        input_ids: Current input token IDs.
        past_key_values: Cached KV states.
        num_draft_tokens: Number of draft tokens to generate.
        cutoff_layer: Layer to start adapter-based drafting.
        serial_parallel_transition: Number of serial steps before parallel.
        logits_processor: Function to process logits for sampling.
        
    Returns:
        Tuple of (draft_tokens, draft_hidden_states).
    """
    if cutoff_layer is None:
        cutoff_layer = model.cutoff_layer
    
    device = input_ids.device
    draft_tokens = []
    draft_hidden_states = []
    
    # Step 1: Initial forward pass up to cutoff layer
    with torch.no_grad():
        outputs = model.base_model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        
        all_hidden_states = outputs.hidden_states
        current_hidden = all_hidden_states[cutoff_layer + 1]  # +1 for embedding layer
        
        # Step 2: Serial drafting phase
        for i in range(min(serial_parallel_transition, num_draft_tokens)):
            # Use adapter at current layer to predict token
            layer_idx = min(cutoff_layer + i, model.num_layers - 1)
            layer = model.base_model.model.layers[layer_idx]
            
            if layer.adapter is not None:
                # Get prediction from adapter
                adapter_logits = layer.adapter(current_hidden[:, -1:, :])
                
                # Sample token
                if logits_processor is not None:
                    adapter_logits = logits_processor(None, adapter_logits[:, -1, :])
                    probs = F.softmax(adapter_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = adapter_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                draft_tokens.append(next_token)
                draft_hidden_states.append(current_hidden[:, -1:, :])
                
                # Continue forward pass to next layer if needed
                if i < serial_parallel_transition - 1 and layer_idx < model.num_layers - 1:
                    # Get embedding for next token
                    next_token_emb = model.base_model.model.embed_tokens(next_token)
                    
                    # Forward through remaining layers up to next adapter layer
                    for j in range(layer_idx + 1, min(layer_idx + 2, model.num_layers)):
                        layer_output = model.base_model.model.layers[j](
                            next_token_emb,
                            use_cache=False,
                        )
                        current_hidden = layer_output[0]
                        next_token_emb = current_hidden
            else:
                break
        
        # Step 3: Parallel drafting phase (if more tokens needed)
        if len(draft_tokens) < num_draft_tokens:
            # Use adapters at higher layers to predict remaining tokens in parallel
            remaining_tokens = num_draft_tokens - len(draft_tokens)
            
            for i in range(remaining_tokens):
                layer_idx = min(
                    cutoff_layer + serial_parallel_transition + i,
                    model.num_layers - 1
                )
                layer = model.base_model.model.layers[layer_idx]
                
                if layer.adapter is not None and layer_idx < len(all_hidden_states) - 1:
                    # Use hidden state from this layer
                    layer_hidden = all_hidden_states[layer_idx + 1]
                    adapter_logits = layer.adapter(layer_hidden[:, -1:, :])
                    
                    # Sample token
                    if logits_processor is not None:
                        adapter_logits = logits_processor(None, adapter_logits[:, -1, :])
                        probs = F.softmax(adapter_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = adapter_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    
                    draft_tokens.append(next_token)
                    draft_hidden_states.append(layer_hidden[:, -1:, :])
    
    # Concatenate draft tokens
    if draft_tokens:
        draft_sequence = torch.cat(draft_tokens, dim=1)
    else:
        draft_sequence = torch.empty((input_ids.size(0), 0), dtype=torch.long, device=device)
    
    return draft_sequence, draft_hidden_states


def spaf_verify(
    model,
    input_ids: torch.LongTensor,
    draft_tokens: torch.LongTensor,
    past_key_values: Optional[Tuple] = None,
    cutoff_layer: Optional[int] = None,
    logits_processor=None,
) -> Tuple[torch.LongTensor, int, torch.Tensor]:
    """
    Verify draft tokens using AlignmentHead and parallel partial forward passes.
    
    Strategy:
    1. For each draft token, use AlignmentHead to generate pseudo hidden state
    2. Run partial forward pass from cutoff layer to end in parallel
    3. Apply standard speculative decoding rejection sampling
    
    Args:
        model: SpAFModel instance.
        input_ids: Current input token IDs.
        draft_tokens: Draft tokens to verify.
        past_key_values: Cached KV states.
        cutoff_layer: Layer to start verification from.
        logits_processor: Function to process logits for sampling.
        
    Returns:
        Tuple of (accepted_tokens, num_accepted, sample_prob).
    """
    if cutoff_layer is None:
        cutoff_layer = model.cutoff_layer
    
    device = input_ids.device
    batch_size = input_ids.size(0)
    num_drafts = draft_tokens.size(1)
    
    if num_drafts == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device), 0, None
    
    with torch.no_grad():
        # Step 1: Generate pseudo hidden states using AlignmentHead
        pseudo_hidden_states = model.alignment_head(token_ids=draft_tokens)
        
        # Step 2: Parallel partial forward passes
        # Prepare input for partial forward (batch all draft tokens)
        all_verification_logits = []
        
        for i in range(num_drafts):
            # Get pseudo hidden state for this draft token
            pseudo_h = pseudo_hidden_states[:, i:i+1, :]
            
            # Run through layers from cutoff_layer to end
            hidden = pseudo_h
            for layer_idx in range(cutoff_layer, model.num_layers):
                layer = model.base_model.model.layers[layer_idx]
                layer_output = layer(
                    hidden,
                    use_cache=False,
                )
                hidden = layer_output[0]
            
            # Apply final norm and LM head
            hidden = model.base_model.model.norm(hidden)
            logits = model.base_model.lm_head(hidden)
            all_verification_logits.append(logits)
        
        # Stack all logits
        verification_logits = torch.cat(all_verification_logits, dim=1)  # (batch, num_drafts, vocab)
        
        # Step 3: Rejection sampling
        # Also need the probability from original model for the first draft position
        # Get logits for the current position
        base_outputs = model.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        base_logits = base_outputs.logits[:, -1:, :]  # Last position
        
        # Apply logits processor if provided
        if logits_processor is not None:
            base_logits = logits_processor(None, base_logits[:, -1, :]).unsqueeze(1)
            for i in range(num_drafts):
                verification_logits[:, i, :] = logits_processor(
                    None, verification_logits[:, i, :]
                )
        
        # Compute probabilities
        base_probs = F.softmax(base_logits, dim=-1)
        verification_probs = F.softmax(verification_logits, dim=-1)
        
        # Rejection sampling
        accepted_tokens = []
        num_accepted = 0
        
        # Check first draft token against base model
        draft_token_0 = draft_tokens[:, 0]
        p_draft_0 = base_probs[0, 0, draft_token_0].item()
        
        # Accept/reject based on probability threshold
        if torch.rand(1).item() < min(1.0, p_draft_0):
            accepted_tokens.append(draft_token_0.unsqueeze(1))
            num_accepted = 1
            
            # Check subsequent tokens
            for i in range(1, num_drafts):
                draft_token_i = draft_tokens[:, i]
                p_model = verification_probs[:, i-1, draft_token_i].item()
                p_draft = verification_probs[:, i, draft_token_i].item()
                
                # Rejection sampling criterion
                if torch.rand(1).item() < min(1.0, p_model / (p_draft + 1e-10)):
                    accepted_tokens.append(draft_token_i.unsqueeze(1))
                    num_accepted += 1
                else:
                    break
        
        # Concatenate accepted tokens
        if accepted_tokens:
            accepted_sequence = torch.cat(accepted_tokens, dim=1)
        else:
            # If no tokens accepted, sample from base model
            if logits_processor is not None:
                probs = F.softmax(base_logits[:, -1, :], dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1)
            else:
                sampled_token = base_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            accepted_sequence = sampled_token
            num_accepted = 1
        
        sample_prob = base_probs
    
    return accepted_sequence, num_accepted, sample_prob


@torch.no_grad()
def spaf_generate(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 512,
    num_draft_tokens: int = 5,
    temperature: float = 0.0,
    top_p: float = 0.0,
    top_k: float = 0.0,
    serial_parallel_transition: int = 2,
) -> torch.LongTensor:
    """
    Generate text using SpAF speculative decoding.
    
    Args:
        model: SpAFModel instance.
        input_ids: Input token IDs.
        max_new_tokens: Maximum number of new tokens to generate.
        num_draft_tokens: Number of draft tokens per iteration.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        serial_parallel_transition: Transition point from serial to parallel.
        
    Returns:
        Generated token IDs.
    """
    device = input_ids.device
    
    # Prepare logits processor
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature, top_p, top_k)
    else:
        logits_processor = None
    
    # Initialize KV cache
    from ..model.kv_cache import initialize_past_key_values
    past_key_values, past_key_values_data, current_length_data = \
        initialize_past_key_values(model.base_model)
    
    generated_tokens = 0
    current_ids = input_ids.clone()
    
    while generated_tokens < max_new_tokens:
        # Draft phase
        draft_tokens, draft_hidden = spaf_draft(
            model=model,
            input_ids=current_ids,
            past_key_values=past_key_values,
            num_draft_tokens=num_draft_tokens,
            cutoff_layer=model.cutoff_layer,
            serial_parallel_transition=serial_parallel_transition,
            logits_processor=logits_processor,
        )
        
        # Verify phase
        accepted_tokens, num_accepted, sample_prob = spaf_verify(
            model=model,
            input_ids=current_ids,
            draft_tokens=draft_tokens,
            past_key_values=past_key_values,
            cutoff_layer=model.cutoff_layer,
            logits_processor=logits_processor,
        )
        
        # Update current sequence
        current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
        generated_tokens += num_accepted
        
        # Check for EOS token
        if model.tokenizer.eos_token_id in current_ids[0, input_ids.size(1):].tolist():
            break
    
    return current_ids
