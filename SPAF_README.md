# SpAF (Speculative Adapter Fusion) Implementation

## Overview

SpAF is an efficient edge-side speculative decoding approach that eliminates the need for an independent draft model by using:
- **Adapters**: Lightweight modules inserted into each Transformer layer to predict tokens
- **AlignmentHead**: A module that generates pseudo hidden states for parallel verification

This implementation upgrades the Gumiho architecture to SpAF, significantly reducing memory footprint while maintaining high speedup ratios.

## Architecture

### Key Components

1. **Adapter Modules** (`gumiho/model/spaf_modules.py`)
   - Lightweight bottleneck architecture (down-projection → activation → up-projection)
   - Inserted into Transformer layers starting from cutoff layer `k`
   - Predicts final output tokens based on intermediate hidden states

2. **AlignmentHead** (`gumiho/model/spaf_modules.py`)
   - MLP that generates pseudo hidden states from token embeddings
   - Trained to mimic hidden states at layer `k-1`
   - Enables parallel verification of draft tokens

3. **SpAFModel** (`gumiho/model/spaf_model.py`)
   - Top-level model encapsulating base LLM and AlignmentHead
   - Manages adapter initialization and training
   - Provides save/load functionality for SpAF components

4. **Training Logic** (`gumiho/train/train_spaf.py`)
   - Multi-task joint loss: adapter losses + alignment loss
   - Freezes base LLM parameters
   - Only trains adapters and alignment head

5. **Inference Logic** (`gumiho/inference/spaf_generate.py`)
   - `spaf_draft()`: Hierarchical draft generation (serial → parallel)
   - `spaf_verify()`: Parallel verification using AlignmentHead
   - `spaf_generate()`: Complete generation pipeline

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure transformers, torch, and other dependencies are installed
pip install transformers>=4.31.0 torch>=2.0.0
```

## Quick Start

### 1. Training SpAF Components (Single GPU)

```python
import json
import torch
from torch.utils.data import DataLoader
from gumiho.model.spaf_model import SpAFModel
from gumiho.train.train_spaf import (
    SpAFDataset, 
    setup_spaf_training, 
    train_spaf_epoch,
    evaluate_spaf
)

# Load configuration
with open('gumiho/train/spaf_config.json', 'r') as f:
    config = json.load(f)

# Initialize SpAF model
model = SpAFModel.from_pretrained(
    base_model_path=config['base_model_path'],
    spaf_config=config['spaf_settings'],
)

# Setup training
optimizer, loss_fn = setup_spaf_training(
    model=model,
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    loss_config=config['training']['loss_weights'],
)

# Prepare dataset
train_dataset = SpAFDataset(
    tokenized_data=your_tokenized_data,
    max_length=config['training']['max_seq_length'],
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

for epoch in range(config['training']['num_epochs']):
    avg_losses = train_spaf_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        log_interval=config['training']['logging_steps'],
    )
    print(f"Epoch {epoch+1}, Avg Loss: {avg_losses['total']:.4f}")
    
    # Save checkpoint
    model.save_spaf_components(
        f"{config['output']['output_dir']}/epoch_{epoch+1}"
    )
```

### 2. Training SpAF Components (Multi-GPU with DeepSpeed)

For efficient multi-GPU training, use the DeepSpeed-enabled training script:

```bash
# Prepare data first
cd examples
python train_spaf_deepspeed_example.py

# Launch multi-GPU training (e.g., 2 GPUs)
bash scripts/train_spaf_deepspeed.sh

# Or use the auto-generated script
bash launch_spaf_training.sh
```

**Manual launch with DeepSpeed:**

```bash
# Single machine with 2 GPUs
deepspeed --num_gpus=2 gumiho/train/train_spaf_deepspeed.py \
    --deepspeed_config gumiho/train/ds_config_spaf.json \
    --base_model_path meta-llama/Llama-2-7b-hf \
    --train_data_path ./train_data \
    --checkpoint_dir ./spaf_checkpoints \
    --num_epochs 3 \
    --max_seq_length 512 \
    --cutoff_layer 16

# Multiple machines (distributed training)
deepspeed --hostfile=hostfile.txt gumiho/train/train_spaf_deepspeed.py \
    --deepspeed_config gumiho/train/ds_config_spaf.json \
    --base_model_path meta-llama/Llama-2-7b-hf \
    --train_data_path ./train_data \
    --checkpoint_dir ./spaf_checkpoints
```

**DeepSpeed Configuration:**

The `ds_config_spaf.json` file controls DeepSpeed optimization:
- **ZeRO Stage 2**: Shards optimizer states and gradients across GPUs
- **FP16 Training**: Mixed precision for faster training
- **Optimizer Offloading**: Offloads optimizer states to CPU to save GPU memory
- **Gradient Accumulation**: Accumulates gradients for effective larger batch sizes

Adjust the configuration based on your hardware:

```json
{
  "train_batch_size": 32,           # Total batch size across all GPUs
  "train_micro_batch_size_per_gpu": 8,  # Batch size per GPU
  "gradient_accumulation_steps": 1,
  "zero_optimization": {
    "stage": 2,                      # Use ZeRO-2 for balanced performance
    "offload_optimizer": {
      "device": "cpu"                # Offload to CPU to save GPU memory
    }
  }
}
```

### 3. Inference with SpAF

```python
import torch
from gumiho.model.spaf_model import SpAFModel
from gumiho.inference.spaf_generate import spaf_generate

# Load trained model
model = SpAFModel.from_pretrained(
    base_model_path='path/to/base/model',
    spaf_config={
        'enable_spaf': True,
        'cutoff_layer': 16,  # For 7B models
        'adapter_dim_ratio': 0.25,
    }
)

# Load trained SpAF components
model.load_spaf_components('path/to/spaf/checkpoint')
model.eval()
model = model.to('cuda')

# Prepare input
tokenizer = model.get_tokenizer()
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

# Generate with SpAF
output_ids = spaf_generate(
    model=model,
    input_ids=input_ids,
    max_new_tokens=100,
    num_draft_tokens=5,
    temperature=0.7,
    top_p=0.9,
)

# Decode output
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
```

## Configuration

Edit `gumiho/train/spaf_config.json` to customize:

### SpAF Settings
- `cutoff_layer`: Layer index to split LLM (typically `num_layers // 2`)
- `adapter_dim_ratio`: Size of adapter relative to hidden size (default: 0.25)
- `alignment_head.num_layers`: Number of MLP layers in AlignmentHead (default: 2)
- `alignment_head.hidden_dim_ratio`: Intermediate dimension ratio (default: 2.0)

### Training Settings
- `learning_rate`: Learning rate for optimizer (default: 1e-4)
- `loss_weights`: Weights for different loss components
  - `alignment`: Weight for alignment loss (default: 1.0)
  - `adapter_base`: Base weight for adapter losses (default: 1.0)
- `alignment_loss_type`: Choose 'mse' or 'cosine' (default: 'mse')

### Inference Settings
- `num_draft_tokens`: Number of draft tokens per iteration (default: 5)
- `serial_parallel_transition`: Serial steps before parallel (default: 2)
- `temperature`, `top_p`, `top_k`: Sampling parameters

## File Structure

```
gumiho/
├── model/
│   ├── spaf_modules.py              # Adapter & AlignmentHead definitions
│   ├── spaf_model.py                # SpAFModel class
│   └── modeling_llama_kv.py         # Modified LlamaDecoderLayer with adapter support
├── train/
│   ├── train_spaf.py                # Single-GPU training logic
│   ├── train_spaf_deepspeed.py      # Multi-GPU training with DeepSpeed
│   ├── spaf_config.json             # Configuration file
│   └── ds_config_spaf.json          # DeepSpeed configuration
├── inference/
│   └── spaf_generate.py             # Inference functions (draft, verify, generate)
├── examples/
│   ├── train_spaf_example.py        # Single-GPU training example
│   ├── train_spaf_deepspeed_example.py  # Multi-GPU setup example
│   └── inference_spaf_example.py    # Inference example
└── scripts/
    └── train_spaf_deepspeed.sh      # Multi-GPU training launch script
```

## Key Features

### Memory Efficiency
- **No Independent Draft Model**: Adapters are lightweight (typically 1/4 parameters of a layer)
- **Frozen Base Model**: Only adapters and alignment head are trained
- **Shared Backbone**: Uses the same LLM for both drafting and verification

### Inference Speed
- **Hierarchical Drafting**: Serial steps for accuracy, parallel for speed
- **Batch Verification**: All draft tokens verified in parallel
- **Adaptive Acceptance**: Standard speculative decoding rejection sampling

### Training Efficiency
- **Multi-Task Learning**: Joint training of adapters and alignment head
- **On-the-Fly Hidden States**: No need to pre-compute and store hidden states
- **Minimal Parameters**: Only ~5-10% of total model parameters are trainable

## Performance Tips

1. **Cutoff Layer Selection**:
   - Earlier layers (k < num_layers/2): Better alignment, slower drafting
   - Later layers (k > num_layers/2): Faster drafting, harder alignment
   - Recommended: k = num_layers // 2

2. **Draft Token Count**:
   - More tokens (5-10): Higher speedup potential, more rejections
   - Fewer tokens (2-4): Lower speedup, higher acceptance rate
   - Recommended: Start with 5, tune based on acceptance rate

3. **Serial-Parallel Transition**:
   - More serial steps: Better draft quality, less parallelism
   - More parallel steps: More parallelism, potential quality drop
   - Recommended: 2-3 serial steps

4. **Adapter Size**:
   - Larger ratio (0.5): Better predictions, more memory
   - Smaller ratio (0.125): Less memory, potential accuracy drop
   - Recommended: 0.25 for balance

## Benchmarking

To benchmark SpAF against baseline:

```python
import time
from gumiho.inference.spaf_generate import spaf_generate

# Baseline generation
start = time.time()
baseline_output = model.base_model.generate(input_ids, max_new_tokens=100)
baseline_time = time.time() - start

# SpAF generation
start = time.time()
spaf_output = spaf_generate(model, input_ids, max_new_tokens=100)
spaf_time = time.time() - start

speedup = baseline_time / spaf_time
print(f"Speedup: {speedup:.2f}x")
```

## Multi-GPU Training Benefits

### Memory Efficiency
- **ZeRO Optimization**: Reduces memory redundancy across GPUs
- **Optimizer Offloading**: Moves optimizer states to CPU RAM
- **Gradient Checkpointing**: Trades computation for memory

### Speed Improvements
- **Data Parallelism**: Each GPU processes different batches
- **Gradient Accumulation**: Simulates larger batch sizes
- **Communication Overlap**: Overlaps communication with computation

### Scalability
- **Single Machine**: 2-8 GPUs with DeepSpeed
- **Multi-Node**: Scales to multiple machines with `--hostfile`
- **Efficient Memory**: Train larger models with limited GPU memory

## Troubleshooting

### Issue: Low acceptance rate
- **Solution**: Increase serial drafting steps, decrease num_draft_tokens, or train longer

### Issue: High memory usage (Single GPU)
- **Solution**: Reduce adapter_dim_ratio, use gradient checkpointing, or reduce batch size

### Issue: High memory usage (Multi-GPU)
- **Solution**: Enable ZeRO Stage 3, enable optimizer offloading, or reduce micro batch size per GPU

### Issue: Slow convergence
- **Solution**: Increase learning rate, adjust loss weights, or use warmup schedule

### Issue: DeepSpeed OOM errors
- **Solution**: 
  1. Reduce `train_micro_batch_size_per_gpu` in DeepSpeed config
  2. Enable optimizer offloading: `"offload_optimizer": {"device": "cpu"}`
  3. Use ZeRO Stage 3 instead of Stage 2
  4. Enable activation checkpointing

### Issue: Slow multi-GPU training
- **Solution**:
  1. Check GPU communication bandwidth
  2. Increase `train_micro_batch_size_per_gpu` if memory allows
  3. Disable `offload_optimizer` if CPU is bottleneck
  4. Use faster interconnect (NVLink, InfiniBand)

## Citation

If you use this implementation, please cite:

```bibtex
@article{spaf2025,
  title={SpAF: Speculative Adapter Fusion for Edge-Side Large Language Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This implementation is based on Gumiho and follows the same license terms.
