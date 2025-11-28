# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import os
from typing import Dict, Any


class ConfigLoader:
    """Configuration loader for training parameters."""
    
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
                'scripts', 'train_config.json'
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
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_loss_weights(self) -> Dict[str, Any]:
        """Get loss weights configuration."""
        return self.config.get('loss_weights', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_paths(self) -> Dict[str, Any]:
        """Get path configuration."""
        return self.config.get('paths', {})
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return self.config.get('optimizer', {})
    
    def get_spaf_training_config(self) -> Dict[str, Any]:
        """Get SpAF training configuration."""
        return self.config.get('spaf_training', {})
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., 'training.num_epochs')
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
    def create_default_config(cls, output_path: str = 'scripts/train_config.json'):
        """
        Create a default configuration file.
        
        Args:
            output_path: Path to save the default configuration
        """
        default_config = {
            "training": {
                "num_epochs": 200,
                "max_len": 2048,
                "test_freq": 100,
                "start_epoch": 0,
                "run_mode": "train",
                "data_noise": 1,
                "noise_type": "uniform",
                "noise_mean": 0.0,
                "noise_std": 0.2
            },
            "model": {
                "model_name": "l3_8b",
                "dropout_rate": 0.0,
                "mlp_num": 5,
                "train_mlp_input": "decoder_output",
                "serial_head_num": 2
            },
            "loss_weights": {
                "p_w": 0.1,
                "v_w": 1.0,
                "mlp_v_w": 1.0,
                "mlp_p_w": 0.1,
                "mlp_loss_weight": 9.0,
                "mlp_loss_decay_coefficient": 0.8,
                "topk_loss_num": 0
            },
            "data": {
                "train_split_ratio": 0.95,
                "test_split_ratio": 0.05,
                "num_workers": 2,
                "batch_size": 4
            },
            "logging": {
                "logger_name": "wandb",
                "tensorboard_log_path": "./runs"
            },
            "paths": {
                "tmpdir": "/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/train_data",
                "cpdir": "./ckpts-cloudml",
                "configpath": "/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/gumiho/train/Gumiho-LLaMA3-Instruct-8B.json",
                "basepath": "/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct",
                "tokenizer_path": "/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct"
            },
            "optimizer": {
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.95,
                "grad_clip": 0.5
            },
            "spaf_training": {
                "num_epochs": 3,
                "max_seq_length": 512,
                "batch_size": 8,
                "adapter_dim_ratio": 0.25,
                "alignment_weight": 1.0,
                "adapter_weight": 1.0,
                "alignment_loss_type": "mse",
                "logging_steps": 100,
                "save_steps": 1000,
                "eval_steps": 500
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default configuration saved to: {output_path}")
