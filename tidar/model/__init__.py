"""
TiDAR: Token-level Diffusion Autoregressive Reasoning
A speculative decoding framework that combines autoregressive and diffusion models.
"""

from .configuration_tidar_qwen2 import TiDARQwen2Config
from .modeling_tidar_qwen2 import TiDARQwen2ForCausalLM

__version__ = "0.1.0"
__all__ = ["TiDARQwen2Config", "TiDARQwen2ForCausalLM"]
