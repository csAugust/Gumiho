# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import os
from typing import Dict, Any


class EvalConfigLoader:
    """Configuration loader for evaluation parameters."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default to the config file in scripts directory
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'scripts', 'eval_config.json'
            )
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_gumiho_params(self) -> Dict[str, Any]:
        """Get Gumiho model parameters."""
        return self.config.get('gumiho_params', {})
    
    def get_distributed_config(self) -> Dict[str, Any]:
        """Get distributed training configuration."""
        return self.config.get('distributed', {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self.config.get('models', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_model_paths(self, model_name: str) -> Dict[str, str]:
        """
        Get model paths for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'l3_8b', 'l3_70b')
        
        Returns:
            Dictionary with gumiho_path and base_model_path
        """
        models_config = self.get_models_config()
        return models_config.get(model_name, {})
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., 'evaluation.temperature')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates in dot notation
        """
        for key, value in updates.items():
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
    
    def save_config(self, output_path: str = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def create_default_config(cls, output_path: str = 'scripts/eval_config.json'):
        """
        Create a default configuration file.
        
        Args:
            output_path: Path to save the default configuration
        """
        default_config = {
            "evaluation": {
                "model_name": "l3_8b",
                "bench_name": "mt_bench",
                "temperature": 0,
                "num_trials": 2,
                "max_new_token": 1024
            },
            "gumiho_params": {
                "mlptopk": 35,
                "total_tokens": 64,
                "depth": 6,
                "mlp_num": 5,
                "pruning": 35,
                "topk": 14,
                "complete_mask": 1
            },
            "distributed": {
                "num_gpus_per_model": 1,
                "num_gpus_total": 1,
                "num_nodes": 1,
                "use_deepspeed": false,
                "use_ray": false
            },
            "models": {
                "l3_8b": {
                    "gumiho_path": "/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/gumiho/state_199",
                    "base_model_path": "/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct"
                },
                "l3_70b": {
                    "gumiho_path": "l3_70b_ckpt",
                    "base_model_path": "meta-llama/Meta-Llama-3-70B-Instruct"
                }
            },
            "logging": {
                "log_dir": "log/evaluation",
                "tensorboard_log_path": "./runs/evaluation"
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default evaluation configuration saved to: {output_path}")
