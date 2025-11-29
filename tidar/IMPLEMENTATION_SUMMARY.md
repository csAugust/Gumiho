# TiDAR Implementation Summary

This document provides a comprehensive overview of the TiDAR (Think in Diffusion, Talk in Autoregression) implementation for training on the Qwen2.5 model.

## What Was Implemented

### 1. Model Architecture (`tidar/model/modeling_tidar_qwen2.py`)

#### TiDARModel
- **Hybrid Attention Mask**: Implements the core TiDAR attention pattern
  - Causal attention for clean tokens (autoregressive verification)
  - Block-wise bidirectional attention for masked tokens (diffusion drafting)
  - Full attention from masked to clean tokens (conditioning)
  - No attention from clean to masked tokens (prevents information leakage)

#### TiDARQwen2ForCausalLM
- Based on Qwen2.5 architecture
- Uses a shared `lm_head` for both AR and diffusion tasks (parameter efficient)
- Includes `from_pretrained` method to load Qwen2.5 weights and convert to TiDAR format
- Supports both standard and TiDAR-formatted inputs

### 2. Training Script (`tidar/train_tidar_deepspeed.py`)

#### CustomDataset
- Loads pre-processed data (compatible with Gumiho data format)
- Converts sequences to TiDAR format:
  - First `clean_ratio * seq_length` tokens remain as-is (clean tokens)
  - Remaining tokens replaced with MASK tokens (masked tokens)
- Labels preserve original tokens for both regions

#### Loss Computation
```python
def compute_tidar_loss(logits, labels, loss_mask, clean_lengths, args):
    # AR Loss: Next token prediction in clean region
    # Diffusion Loss: Predict original tokens in masked region
    # Total Loss: Weighted combination
```

**Key Features:**
- Separate calculation of AR loss and diffusion loss
- Configurable loss weights
- Accurate tracking of tokens in each region

#### Training Loop
- DeepSpeed integration for distributed training
- FP16 mixed precision training
- ZeRO Stage 2 optimization
- TensorBoard logging
- Checkpoint saving

### 3. Configuration Files

#### `train_config.json`
Main training parameters:
- Data and checkpoint directories
- Model hyperparameters (block_size, clean_ratio)
- Training settings (epochs, batch size)
- Loss weights

#### `ds_config.json`
DeepSpeed configuration:
- Optimizer settings (AdamW)
- Learning rate scheduler
- FP16 settings
- ZeRO optimization stage 2

### 4. Documentation

#### `README.md`
Comprehensive guide covering:
- Quick start instructions
- Training process explanation
- Architecture details
- Hyperparameter reference
- Troubleshooting tips

#### `train.sh`
Convenient shell script for launching training with customizable options.

## Key Design Decisions

### 1. Shared LM Head
Rather than using separate validation and draft heads, we use a single shared `lm_head`:
- **Pros**: Parameter efficient, simpler architecture
- **Rationale**: The paper's core insight is about the attention pattern, not separate prediction heads

### 2. Loss Calculation Strategy
Split loss into two components based on sequence regions:
- **AR Loss**: Applied to clean tokens (positions 0 to `clean_length-1`)
  - Standard next-token prediction
  - Uses original `loss_mask` to respect data-specific masking
- **Diffusion Loss**: Applied to masked tokens (positions `clean_length` to end)
  - Predicts original tokens that were masked
  - All masked positions contribute to loss

### 3. Data Format Compatibility
The implementation uses the same data format as Gumiho:
- `input_ids`: Token sequences
- `loss_mask`: Which tokens to include in loss
- No need to regenerate data, just reuse existing preprocessed data

### 4. Position Encoding
Both clean and masked tokens use sequential position IDs (0, 1, 2, ..., seq_len-1):
- Maintains positional information
- Enables the model to learn position-dependent patterns

## Training Flow

```
1. Load Data
   ↓
2. Format to TiDAR
   [clean_tokens, MASK, MASK, ...]
   ↓
3. Forward Pass
   - Hybrid attention mask applied automatically
   - Single forward pass through model
   ↓
4. Compute Loss
   - AR loss on clean region
   - Diffusion loss on masked region
   - Combine with weights
   ↓
5. Backward & Update
   ↓
6. Log & Save
```

## Differences from Original Paper

### Simplified Training
The paper discusses a more complex training procedure with:
- Masking strategies during training
- Progressive training schedules

Our implementation uses a simpler approach:
- Fixed `clean_ratio` throughout training
- Direct end-to-end training of both objectives

### Shared Head vs Dual Heads
The paper suggests separate heads for validation and drafting. Our implementation:
- Uses a single shared `lm_head`
- Relies on the hybrid attention mask to differentiate behavior
- Simpler and more parameter-efficient

## Usage Example

### 1. Generate Training Data
```bash
bash scripts/gendata_llama3.sh \
    --outdir ./train_data \
    --model-path Qwen/Qwen2.5-1.5B \
    --end 10000
```

### 2. Configure Training
Edit `tidar/train_config.json`:
```json
{
  "data_dir": "./train_data",
  "qwen_model_path": "Qwen/Qwen2.5-1.5B",
  "tidar_clean_ratio": 0.5,
  "tidar_block_size": 8,
  "num_epochs": 3
}
```

### 3. Launch Training
```bash
cd tidar
bash train.sh --num-gpus 8
```

### 4. Monitor Training
```bash
tensorboard --logdir runs
```

## Expected Behavior

### During Training
- AR loss should decrease as the model learns next-token prediction on clean tokens
- Diffusion loss should decrease as the model learns to predict masked tokens
- Both losses should converge, but may not reach the same value (depends on task difficulty)

### Convergence
- AR loss typically converges faster (standard task)
- Diffusion loss may take longer (harder denoising task)
- Adjust `ar_loss_weight` and `diffusion_loss_weight` to balance training

## Future Enhancements

### Potential Improvements
1. **Inference Implementation**: Add `tidar_generate()` method for actual speculative decoding
2. **Dynamic Masking**: Vary `clean_ratio` during training
3. **Advanced Masking Strategies**: Block masking, random masking patterns
4. **Curriculum Learning**: Start with easier tasks, gradually increase difficulty
5. **Knowledge Distillation**: Use a teacher model to guide training

### Performance Optimization
1. **Flash Attention**: Enable flash attention for faster training
2. **Gradient Checkpointing**: Reduce memory usage for larger models
3. **DDP vs DeepSpeed**: Compare performance with PyTorch DDP

## Validation

To verify the implementation is working correctly:

1. **Check Loss Values**: Both AR and diffusion losses should decrease
2. **Inspect Outputs**: Model should predict reasonable tokens in both regions
3. **Attention Visualization**: Verify hybrid attention mask is applied correctly
4. **Gradients**: Ensure gradients flow to all parameters

## Comparison with Gumiho

| Aspect | Gumiho | TiDAR |
|--------|--------|-------|
| **Architecture** | Additional MLP layers | Hybrid attention mask |
| **Training** | Hidden state matching | Dual objective (AR + Diffusion) |
| **Data Format** | Teacher hidden states | Same format, different usage |
| **Inference** | Sequential prediction | Parallel drafting (to be implemented) |

## Summary

This implementation provides a complete training pipeline for TiDAR models:
- ✅ Model architecture with hybrid attention
- ✅ Training script with dual loss objectives
- ✅ Configuration files for easy setup
- ✅ Documentation and usage examples
- ⏳ Inference implementation (future work)

The implementation is production-ready for training TiDAR models on the Qwen2.5 architecture using the same data format as Gumiho.
