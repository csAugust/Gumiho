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
    
    @classmethod
    def from_qwen2_config(
        cls,
        qwen_config: Qwen2Config,
        block_size: int = 8,
        clean_ratio: float = 0.5,
        use_tidar: bool = True,
        **kwargs
    ) -> "TiDARQwen2Config":
        """
        Create TiDAR configuration from Qwen2 configuration.
        
        Args:
            qwen_config: Qwen2 configuration to convert
            block_size: Block size for block-wise bidirectional attention
            clean_ratio: Ratio of clean tokens in the input sequence
            use_tidar: Whether to enable TiDAR mode
            **kwargs: Additional configuration parameters
            
        Returns:
            TiDARQwen2Config instance with Qwen2 parameters and TiDAR extensions
        """
        # Extract all Qwen2 config attributes
        qwen_dict = qwen_config.to_dict()
        
        # Remove model_type to avoid conflicts
        qwen_dict.pop('model_type', None)
        
        # Add TiDAR-specific parameters
        qwen_dict.update({
            'block_size': block_size,
            'clean_ratio': clean_ratio,
            'use_tidar': use_tidar,
            **kwargs
        })
        
        return cls(**qwen_dict)
