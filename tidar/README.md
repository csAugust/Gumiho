# TiDAR Training Implementation

This directory contains the implementation for training TiDAR (Think in Diffusion, Talk in Autoregression) models based on the paper.

## Overview

TiDAR is a hybrid architecture that combines:
- **Autoregressive (AR) mode**: For high-quality token generation (validation)
- **Diffusion mode**: For parallel draft generation (think)
- **Single forward pass**: Both tasks executed simultaneously

## Files

- `model/modeling_tidar_qwen2.py`: TiDAR model implementation based on Qwen2.5
- `model/configuration_tidar_qwen2.py`: TiDAR configuration
- `train_tidar_deepspeed.py`: Training script with DeepSpeed support
- `train_config.json`: Training configuration
- `ds_config.json`: DeepSpeed configuration
- `train.sh`: Shell script to launch training

## Quick Start

### 1. Prepare Data

Use the same data format as Gumiho. Generate training data using:

```bash
cd ..
bash scripts/gendata_llama3.sh --outdir ./train_data --model-path Qwen/Qwen2.5-1.5B
```

### 2. Configure Training

Edit `train_config.json` to set:
- `data_dir`: Path to your training data
- `ckpt_dir`: Where to save checkpoints
- `qwen_model_path`: Path to Qwen2.5 base model
- `tidar_block_size`: Block size for bidirectional attention (default: 8)
- `tidar_clean_ratio`: Ratio of clean vs masked tokens (default: 0.5)
- `ar_loss_weight`: Weight for autoregressive loss (default: 1.0)
- `diffusion_loss_weight`: Weight for diffusion loss (default: 1.0)

### 3. Launch Training

```bash
bash train.sh
```

Or with custom arguments:

```bash
deepspeed --num_gpus=8 train_tidar_deepspeed.py \
    --config_path train_config.json \
    --deepspeed_config ds_config.json
```

## Training Process

### Data Format

The training script expects data files with:
- `input_ids`: Token IDs
- `loss_mask`: Mask indicating which tokens to include in loss

### Data Processing

For each training sample, the script:
1. Takes the first `clean_ratio * seq_length` tokens as "clean tokens"
2. Replaces the rest with MASK tokens
3. Computes AR loss on clean region (next token prediction)
4. Computes diffusion loss on masked region (denoising)

### Loss Calculation

- **AR Loss**: Standard next-token prediction loss on clean tokens
- **Diffusion Loss**: Reconstruction loss on masked tokens
- **Total Loss**: Weighted combination of both losses

## Model Architecture

### TiDAR Attention Mask

The hybrid attention mask has the following structure:

```
[ Causal (Clean)     | No Attention        ]
[ Full Attention     | Block-wise Bidir    ]
```

- **Top-left**: Causal attention for clean tokens (autoregressive)
- **Bottom-left**: Masked tokens attend to all clean tokens (conditioning)
- **Bottom-right**: Block-wise bidirectional attention for masked tokens
- **Top-right**: Clean tokens cannot see masked tokens (no leakage)

### Key Components

1. **TiDARModel**: Base model with hybrid attention mask
2. **TiDARQwen2ForCausalLM**: Causal LM with shared lm_head for both tasks
3. **CustomDataset**: Formats data into TiDAR input format
4. **compute_tidar_loss**: Calculates combined AR + diffusion loss

## Hyperparameters

### Model Parameters
- `tidar_block_size`: Block size for bidirectional attention (default: 8)
- `tidar_clean_ratio`: Ratio of clean tokens (default: 0.5)

### Training Parameters
- Learning rate: 2e-5
- Batch size: 32 (can be adjusted via `train_micro_batch_size_per_gpu`)
- Optimizer: AdamW
- FP16 training enabled
- ZeRO Stage 2 optimization

### Loss Weights
- `ar_loss_weight`: Weight for AR loss (default: 1.0)
- `diffusion_loss_weight`: Weight for diffusion loss (default: 1.0)

## Monitoring

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir runs
```

Metrics include:
- `train/batch_loss`: Total loss per batch
- `train/batch_ar_loss`: AR loss per batch
- `train/batch_diffusion_loss`: Diffusion loss per batch
- `train/batch_accuracy`: Token prediction accuracy
- `train/epoch_loss`: Average loss per epoch

## Checkpointing

Checkpoints are saved to `ckpt_dir` at intervals specified by `save_interval`:

- `epoch_{n}/`: Model checkpoint for epoch n
  - `pytorch_model.bin`: Model state dict
  - `config.json`: Model configuration

To resume training from a checkpoint:

```bash
deepspeed --num_gpus=8 train_tidar_deepspeed.py \
    --config_path train_config.json \
    --existing_model_path ./tidar_checkpoints/epoch_2/pytorch_model.bin \
    --start_epoch 3
```

## Troubleshooting

### Out of Memory
- Reduce `train_micro_batch_size_per_gpu` in `ds_config.json`
- Increase `gradient_accumulation_steps`
- Use ZeRO Stage 3 instead of Stage 2

### Slow Training
- Increase `train_micro_batch_size_per_gpu`
- Reduce `num_workers` if CPU is bottleneck
- Enable `offload_optimizer` in DeepSpeed config for large models

### Loss Not Decreasing
- Check if `ar_loss_weight` and `diffusion_loss_weight` are balanced
- Verify data format is correct
- Adjust learning rate
- Check if clean_ratio is appropriate for your data

## Citation

If you use this implementation, please cite the original TiDAR paper:

```bibtex
@article{tidar2024,
  title={TiDAR: Think in Diffusion, Talk in Autoregression},
  author={...},
  journal={arXiv preprint arXiv:2511.08923},
  year={2024}
}
```

## License

This implementation is based on the Gumiho project and follows the same license.
