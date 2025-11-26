# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
SpAF (Speculative Adapter Fusion) Model

This module contains the main SpAF model architecture that combines:
- Base LLM with integrated Adapters
- AlignmentHead for generating pseudo hidden states
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoTokenizer

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .spaf_modules import Adapter, AlignmentHead
from .kv_cache import initialize_past_key_values


class SpAFModel(nn.Module):
    """
    SpAF Model: Self-Speculative Decoding with Adapter Fusion.
    
    This model implements the SpAF architecture where:
    - Adapters predict the final token at each layer
    - AlignmentHead generates pseudo hidden states for parallel verification
    - No independent draft model required
    
    Args:
        base_model (KVLlamaForCausalLM): The base LLM model.
        base_model_name_or_path (str): Path to the base model for tokenizer loading.
        spaf_config (dict): Configuration for SpAF components.
    """
    
    def __init__(
        self,
        base_model: KVLlamaForCausalLM,
        base_model_name_or_path: str,
        spaf_config: dict,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.num_layers = self.config.num_hidden_layers
        
        # SpAF specific configurations
        self.cutoff_layer = spaf_config.get('cutoff_layer', self.num_layers // 2)
        self.adapter_dim_ratio = spaf_config.get('adapter_dim_ratio', 0.25)
        self.alignment_head_config = spaf_config.get('alignment_head', {})
        self.enable_spaf = spaf_config.get('enable_spaf', True)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path, use_fast=False
        )
        
        # Initialize Adapters for each layer (starting from cutoff_layer)
        if self.enable_spaf:
            self._initialize_adapters()
            self._initialize_alignment_head()
        
    def _initialize_adapters(self):
        """Initialize Adapter modules for each decoder layer."""
        for idx, layer in enumerate(self.base_model.model.layers):
            if idx >= self.cutoff_layer:
                adapter = Adapter(
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    adapter_dim_ratio=self.adapter_dim_ratio,
                    hidden_act=self.config.hidden_act
                )
                layer.adapter = adapter
                
    def _initialize_alignment_head(self):
        """Initialize the AlignmentHead module."""
        num_layers = self.alignment_head_config.get('num_layers', 2)
        hidden_dim_ratio = self.alignment_head_config.get('hidden_dim_ratio', 2.0)
        hidden_act = self.alignment_head_config.get('hidden_act', self.config.hidden_act)
        
        self.alignment_head = AlignmentHead(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            hidden_dim_ratio=hidden_dim_ratio,
            hidden_act=hidden_act
        )
        
        # Initialize alignment head embeddings from base model
        self.alignment_head.load_embedding_from_model(
            self.base_model.model.embed_tokens.weight
        )
        
    def freeze_base_model(self):
        """Freeze all parameters of the base LLM."""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def get_trainable_parameters(self):
        """Get all trainable parameters (Adapters + AlignmentHead)."""
        trainable_params = []
        
        # Adapter parameters
        for layer in self.base_model.model.layers:
            if layer.adapter is not None:
                trainable_params.extend(layer.adapter.parameters())
        
        # AlignmentHead parameters
        if hasattr(self, 'alignment_head'):
            trainable_params.extend(self.alignment_head.parameters())
            
        return trainable_params
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        output_hidden_states: bool = True,
        output_adapter_logits: bool = False,
        use_cache: bool = True,
    ):
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            past_key_values: Cached key-value states.
            output_hidden_states: Whether to output all hidden states.
            output_adapter_logits: Whether to output adapter predictions.
            use_cache: Whether to use KV cache.
            
        Returns:
            Dict containing outputs including hidden states and optionally adapter logits.
        """
        # Forward through base model with all hidden states
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states if output_hidden_states else None
        
        # Get final logits from LM head
        lm_logits = self.base_model.lm_head(hidden_states)
        
        result = {
            'logits': lm_logits,
            'hidden_states': all_hidden_states,
            'past_key_values': outputs.past_key_values,
        }
        
        # Optionally compute adapter logits for all layers
        if output_adapter_logits and all_hidden_states is not None:
            adapter_logits_list = []
            for idx, layer in enumerate(self.base_model.model.layers):
                if layer.adapter is not None and idx < len(all_hidden_states) - 1:
                    # Use hidden states from this layer
                    layer_hidden = all_hidden_states[idx + 1]  # +1 because all_hidden_states includes embedding
                    adapter_logits = layer.adapter(layer_hidden)
                    adapter_logits_list.append(adapter_logits)
                else:
                    adapter_logits_list.append(None)
            result['adapter_logits'] = adapter_logits_list
            
        return result
    
    def get_tokenizer(self):
        """Get the tokenizer of the base model."""
        return self.tokenizer
    
    @torch.no_grad()
    def generate_with_spaf(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 512,
        num_draft_tokens: int = 5,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: float = 0.0,
    ):
        """
        Generate text using SpAF speculative decoding.
        
        This is a placeholder for the generation logic that will be implemented
        in the inference module.
        
        Args:
            input_ids: Input token IDs.
            max_new_tokens: Maximum number of new tokens to generate.
            num_draft_tokens: Number of draft tokens to generate at each step.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            
        Returns:
            Generated token IDs.
        """
        # This will be implemented in the inference module
        raise NotImplementedError(
            "SpAF generation logic should be implemented in the inference module"
        )
    
    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        spaf_config: Optional[dict] = None,
        **kwargs
    ):
        """
        Load a SpAF model from pretrained base model.
        
        Args:
            base_model_path: Path to the base model.
            spaf_config: Configuration for SpAF components.
            **kwargs: Additional arguments for model loading.
            
        Returns:
            SpAFModel instance.
        """
        # Default SpAF configuration
        if spaf_config is None:
            spaf_config = {
                'enable_spaf': True,
                'cutoff_layer': None,  # Will be set based on model size
                'adapter_dim_ratio': 0.25,
                'alignment_head': {
                    'num_layers': 2,
                    'hidden_dim_ratio': 2.0,
                },
            }
        
        # Load base model
        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )
        
        # Set default cutoff layer if not specified
        if spaf_config['cutoff_layer'] is None:
            num_layers = base_model.config.num_hidden_layers
            spaf_config['cutoff_layer'] = num_layers // 2
        
        # Create SpAF model
        model = cls(
            base_model=base_model,
            base_model_name_or_path=base_model_path,
            spaf_config=spaf_config,
        )
        
        return model
    
    def save_spaf_components(self, save_path: str):
        """
        Save only the SpAF components (Adapters + AlignmentHead).
        
        Args:
            save_path: Path to save the SpAF components.
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        spaf_state_dict = {}
        
        # Save adapter weights
        for idx, layer in enumerate(self.base_model.model.layers):
            if layer.adapter is not None:
                adapter_state = layer.adapter.state_dict()
                for key, value in adapter_state.items():
                    spaf_state_dict[f'adapter.{idx}.{key}'] = value
        
        # Save alignment head weights
        if hasattr(self, 'alignment_head'):
            alignment_state = self.alignment_head.state_dict()
            for key, value in alignment_state.items():
                spaf_state_dict[f'alignment_head.{key}'] = value
        
        # Save to file
        save_file = os.path.join(save_path, 'spaf_components.pt')
        torch.save(spaf_state_dict, save_file)
        
        # Save configuration
        import json
        config = {
            'cutoff_layer': self.cutoff_layer,
            'adapter_dim_ratio': self.adapter_dim_ratio,
            'alignment_head_config': self.alignment_head_config,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
        }
        config_file = os.path.join(save_path, 'spaf_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_spaf_components(self, load_path: str):
        """
        Load SpAF components from saved checkpoint.
        
        Args:
            load_path: Path to load the SpAF components from.
        """
        import os
        
        # Load state dict
        load_file = os.path.join(load_path, 'spaf_components.pt')
        spaf_state_dict = torch.load(load_file, map_location=self.base_model.device)
        
        # Load adapter weights
        for idx, layer in enumerate(self.base_model.model.layers):
            if layer.adapter is not None:
                adapter_state = {}
                prefix = f'adapter.{idx}.'
                for key, value in spaf_state_dict.items():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        adapter_state[new_key] = value
                if adapter_state:
                    layer.adapter.load_state_dict(adapter_state)
        
        # Load alignment head weights
        if hasattr(self, 'alignment_head'):
            alignment_state = {}
            prefix = 'alignment_head.'
            for key, value in spaf_state_dict.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    alignment_state[new_key] = value
            if alignment_state:
                self.alignment_head.load_state_dict(alignment_state)
