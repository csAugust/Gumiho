# TiDAR-Qwen2: Token-level Diffusion Autoregressive Reasoning

TiDAR (Token-level Diffusion Autoregressive Reasoning) is a novel speculative decoding framework that combines autoregressive and diffusion models to achieve efficient text generation. This implementation is based on Qwen2.5 architecture.

## Overview

TiDAR introduces a hybrid attention mechanism that allows a single model to simultaneously:
1. **Validate** draft tokens using autoregressive attention (Talk)
2. **Generate** new draft tokens using diffusion attention (Think)

This is achieved through a carefully designed hybrid attention mask that enables parallel validation and drafting in a single forward pass.

## Key Features

- **Hybrid Attention Mask**: Combines causal attention for clean tokens with block-wise bidirectional attention for masked tokens
- **Single Forward Pass**: Both validation and drafting happen in parallel
- **KV Cache Management**: Precise KV cache management for efficient inference
- **Configurable Parameters**: Adjustable block size and clean ratio for different use cases

## Architecture

The hybrid attention mask has the following structure for a sequence of length 2S:

```
[ Causal Mask (S×S)    | No Attention (S×S)  ]
[ Full Attention (S×S) | Block-wise Bi (S×S) ]
```

Where:
- **Causal Mask (S×S)**: Standard autoregressive mask for clean tokens
- **Block-wise Bi (S×S)**: Block-wise bidirectional mask for masked tokens
- **Full Attention (S×S)**: Masked tokens can attend to all clean tokens
- **No Attention (S×S)**: Clean tokens cannot attend to masked tokens

## Installation

```bash
# Clone the repository
cd /mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho

# The tidar package is already available in the Gumiho project
```

## Quick Start

### Basic Usage

```python
import torch
from tidar import TiDARQwen2Config, TiDARQwen2ForCausalLM

# Create configuration
config = TiDARQwen2Config(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=512,
    max_position_embeddings=512,
    use_tidar=True,
    block_size=4,
    clean_ratio=0.5
)

# Create model
model = TiDARQwen2ForCausalLM(config)

# Standard input
input_ids = torch.randint(0, 1000, (2, 16))
outputs = model(input_ids)
print(f"Logits shape: {outputs.logits.shape}")
```

### TiDAR Input Format

```python
# Create TiDAR input: [clean_tokens, masked_tokens]
batch_size = 2
seq_length = 16
clean_length = 8
MASK_TOKEN_ID = 500

clean_tokens = torch.randint(0, MASK_TOKEN_ID, (batch_size, clean_length))
masked_tokens = torch.full((batch_size, seq_length - clean_length), MASK_TOKEN_ID)

input_ids = torch.cat([clean_tokens, masked_tokens], dim=1)
outputs = model(input_ids)
```

### Load from Qwen2.5 1.5B

```python
from tidar.model.modeling_tidar_qwen2 import TiDARQwen2ForCausalLM

# Load Qwen2.5 1.5B and convert to TiDAR
model = TiDARQwen2ForCausalLM.from_pretrained(
    pretrained_model_name_or_path="/path/to/Qwen2.5-1.5B-Instruct",
    block_size=8,
    clean_ratio=0.5,
    use_tidar=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Generation

```python
# Standard generation (TiDAR generation coming soon)
input_ids = torch.randint(0, 1000, (1, 16))
generated = model.generate(input_ids, max_new_tokens=128)
```

## Configuration

### TiDARQwen2Config Parameters

- `vocab_size` (int): Vocabulary size
- `hidden_size` (int): Hidden dimension size
- `num_hidden_layers` (int): Number of transformer layers
- `num_attention_heads` (int): Number of attention heads
- `num_key_value_heads` (int): Number of key-value heads (for GQA)
- `intermediate_size` (int): MLP intermediate size
- `max_position_embeddings` (int): Maximum sequence length
- `block_size` (int): Block size for block-wise bidirectional attention (default: 8)
- `use_tidar` (bool): Whether to use TiDAR mode (default: True)
- `clean_ratio` (float): Ratio of clean tokens in the input sequence (default: 0.5)

## Testing

Run the test suite to verify the implementation:

```bash
python tidar/test_tidar_model.py
```

The test suite includes:
1. Basic model creation
2. Standard input format
3. TiDAR input format
4. Hybrid attention mask structure
5. Model modes (TiDAR on/off)
6. Position encoding

## Implementation Details

### Hybrid Attention Mask

The `_create_hybrid_attention_mask` method in `TiDARModel` creates the special attention mask:

```python
def _create_hybrid_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
) -> torch.Tensor:
    # Creates the 4-quadrant hybrid mask
    # 1. Top-left: Causal mask for clean tokens
    # 2. Bottom-right: Block-wise bidirectional for masked tokens
    # 3. Bottom-left: Full attention from masked to clean
    # 4. Top-right: No attention from clean to masked
```

### Position Encoding

For TiDAR format, both clean and masked tokens use positions 0, 1, 2, ..., S-1 to ensure proper RoPE application.

## Future Work

- [ ] Implement TiDAR-specific generation with parallel validation and drafting
- [ ] Add training support for TiDAR models
- [ ] Optimize KV cache management for TiDAR
- [ ] Add support for larger models (Qwen2.5-7B, Qwen2.5-14B)
- [ ] Implement advanced sampling strategies

## References

- TiDAR Paper: [Token-level Diffusion Autoregressive Reasoning](https://arxiv.org/abs/2511.08923)
- Qwen2.5: [Qwen2.5 Technical Report](https://qwenlm.github.io/blog/qwen2.5/)

## License

This implementation is part of the Gumiho project and follows the same license terms.

## Citation

```bibtex
@article{tidar2024,
  title={Token-level Diffusion Autoregressive Reasoning},
  author={...},
  journal={arXiv preprint arXiv:2511.08923},
  year={2024}
}
