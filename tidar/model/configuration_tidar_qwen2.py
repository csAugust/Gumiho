"""
TiDAR-Qwen2 Configuration
"""

from transformers import Qwen2Config


class TiDARQwen2Config(Qwen2Config):
    """
    TiDAR-Qwen2 configuration class.
    
    This configuration extends Qwen2Config with TiDAR-specific parameters.
    
    TiDAR-specific parameters:
        block_size (int, optional): Block size for block-wise bidirectional attention. 
            Defaults to 8.
        use_tidar (bool, optional): Whether to use TiDAR mode. Defaults to True.
        clean_ratio (float, optional): Ratio of clean tokens in the input sequence.
            Should be between 0 and 1. Defaults to 0.5.
        tidar_config (dict, optional): Additional TiDAR configuration parameters.
    """
    
    model_type = "tidar_qwen2"
    
    def __init__(
        self,
        block_size: int = 8,
        use_tidar: bool = True,
        clean_ratio: float = 0.5,
        tidar_config: dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.block_size = block_size
        self.use_tidar = use_tidar
        self.clean_ratio = clean_ratio
        self.tidar_config = tidar_config or {
            'block_size': block_size,
            'use_tidar': use_tidar,
            'clean_ratio': clean_ratio
        }
        
        # Validate parameters
        if not 0 < clean_ratio < 1:
            raise ValueError(f"clean_ratio must be between 0 and 1, got {clean_ratio}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
