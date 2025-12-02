"""
TiDAR Training Script with DeepSpeed
Adapted from Gumiho's main_deepspeed.py for TiDAR model training
"""

import argparse
import deepspeed
from loguru import logger
from transformers import AutoTokenizer
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import TiDAR model
from tidar.model.modeling_tidar_qwen2 import TiDARQwen2ForCausalLM
from tidar.model.configuration_tidar_qwen2 import TiDARQwen2Config

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description='TiDAR Training')
parser.add_argument('--config_path', type=str, default='tidar/train_config.json',
                    help='Path to the training configuration file')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument('--existing_model_path', type=str, help='Path to existing model checkpoint to resume from')

# TiDAR-specific arguments
# parser.add_argument('--qwen_model_path', type=str, default="Qwen/Qwen2.5-1.5B",
#                     help='Path to the Qwen2.5 base model')
# parser.add_argument('--tidar_block_size', type=int, default=8,
#                     help='Block size for block-wise bidirectional attention')
# parser.add_argument('--tidar_clean_ratio', type=float, default=0.5,
#                     help='Ratio of clean tokens in the sequence')

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# Load configuration if config file exists
if os.path.exists(args.config_path):
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)

# Set default values for required arguments
if not hasattr(args, 'data_dir'):
    args.data_dir = './train_data'
if not hasattr(args, 'ckpt_dir'):
    args.ckpt_dir = './tidar_checkpoints'
if not hasattr(args, 'num_epochs'):
    args.num_epochs = 3
if not hasattr(args, 'max_len'):
    args.max_len = 512
if not hasattr(args, 'num_workers'):
    args.num_workers = 4
if not hasattr(args, 'start_epoch'):
    args.start_epoch = 0
if not hasattr(args, 'save_interval'):
    args.save_interval = 1
if not hasattr(args, 'run_mode'):
    args.run_mode = 'train'
if not hasattr(args, 'model_name'):
    args.model_name = 'tidar'
if not hasattr(args, 'logger_file'):
    args.logger_file = 'training'
if not hasattr(args, 'ar_loss_weight'):
    args.ar_loss_weight = 1.0
if not hasattr(args, 'diffusion_loss_weight'):
    args.diffusion_loss_weight = 1.0

logger.remove()

deepspeed.init_distributed()
rank = torch.distributed.get_rank()

if rank == 0:
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.getenv('TENSORBOARD_LOG_PATH', './runs')
    writer = SummaryWriter(log_dir=log_dir)
    logger.add(f"./{args.model_name}/{args.logger_file}.log", level="DEBUG", mode="w")
    logger.info(f"Arguments: {args}")


def list_files(path):
    """Recursively list all files in a directory"""
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class CustomDataset(Dataset):
    """
    Dataset for TiDAR training.
    
    Generates TiDAR-style data from input_ids:
    - input_ids: [clean_tokens, MASK_tokens] (2S length)
    - labels: [shifted_clean_tokens + ignore, original_tokens] (2S length)
    - attention_mask: special 2S x 2S structure
    - position_ids: [0..S-1, 0..S-1] (restarting for masked region)
    """
    
    def __init__(self, datapath, tokenizer, max_len=512):
        self.data = datapath
        self.tokenizer = tokenizer
        # Since we'll double the sequence, actual limit is max_len // 2
        self.max_len = max_len // 2
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id else 151643

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        
        # Load original sequence (limit to S tokens)
        original_input_ids = data['input_ids'][:self.max_len]
        original_loss_mask = data['loss_mask'][:self.max_len]  # Load loss_mask
        seq_length = len(original_input_ids)  # S
        
        # 1. Construct input_ids: [clean_tokens, MASK_tokens] (2S length)
        clean_tokens = original_input_ids  # [t1, t2, ..., tS]
        masked_tokens = torch.full((seq_length,), self.mask_token_id, dtype=torch.long)  # [MASK, MASK, ..., MASK]
        input_ids = torch.cat([clean_tokens, masked_tokens])  # Length 2S
        
        # 2. Construct labels: [AR_labels, Diffusion_labels] (2S length)
        # AR labels: shifted by 1 (for next-token prediction), last position is ignored
        ar_labels = torch.cat([
            original_input_ids[1:],  # [t2, t3, ..., tS]
            torch.tensor([0], dtype=torch.long)  # ignore last position
        ])
        # Diffusion labels: original tokens (for denoising)
        diffusion_labels = original_input_ids  # [t1, t2, ..., tS]
        labels = torch.cat([ar_labels, diffusion_labels])  # Length 2S
        
        # 3. Construct loss_mask: [AR_loss_mask, Diffusion_loss_mask] (2S length)
        # AR loss_mask: shifted by 1 to match AR labels, last position is 0
        ar_loss_mask = torch.cat([
            original_loss_mask[1:],  # Shift mask to align with shifted labels
            torch.tensor([0], dtype=torch.long)  # No loss for last position
        ])
        # Diffusion loss_mask: same as original (all positions contribute to diffusion loss)
        diffusion_loss_mask = torch.cat([
            torch.tensor([0], dtype=torch.long),  # No loss for first position
            original_loss_mask[1:],  # Shift mask to align with shifted labels
        ])  # [m1, m2, ..., mS]
        loss_mask = torch.cat([ar_loss_mask, diffusion_loss_mask])  # Length 2S
        
        # 4. Construct position_ids: [0..S-1, 0..S-1] (restarting for masked region)
        position_ids = torch.cat([
            torch.arange(seq_length, dtype=torch.long),
            torch.arange(seq_length, dtype=torch.long)
        ])  # Length 2S
        
        # 5. Attention mask will be created in collator (needs to be 2S x 2S)
        # For now, just mark valid positions
        attention_mask = torch.ones(2 * seq_length, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "seq_length": seq_length,  # Store S for creating attention mask later
        }


class DataCollatorWithPadding:
    """Collate function for batching TiDAR data with special 2S x 2S attention mask"""
    
    def __init__(self, pad_token_id=0, block_size=8):
        self.pad_token_id = pad_token_id
        self.block_size = block_size
    
    def create_tidar_attention_mask(self, seq_length):
        """
        Create TiDAR attention mask with structure:
        - Clean -> Clean (top-left): Causal
        - Masked -> Clean (bottom-left): Full
        - Masked -> Masked (bottom-right): Block-wise Bidirectional
        - Clean -> Masked (top-right): Zero
        
        Args:
            seq_length: S (half of the total 2S sequence)
        
        Returns:
            attention_mask: [2S, 2S] tensor
        """
        total_length = 2 * seq_length
        mask = torch.zeros(total_length, total_length, dtype=torch.long)
        
        # Top-left: Clean -> Clean (Causal mask)
        # Position i can attend to position j if j <= i
        for i in range(seq_length):
            mask[i, :i+1] = 1
        
        # Bottom-left: Masked -> Clean (Full mask)
        # All masked positions can attend to all clean positions
        mask[seq_length:, :seq_length] = 1
        
        # Bottom-right: Masked -> Masked (Block-wise bidirectional)
        # For simplicity in training, use full bidirectional within masked region
        # (In inference, this would be block-wise)
        mask[seq_length:, seq_length:] = 1
        
        # Top-right: Clean -> Masked (Zero mask) - already initialized to 0
        
        return mask
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Find max sequence length (note: this is 2S)
        max_length = max(len(item['input_ids']) for item in features)
        max_seq_length = max_length // 2  # This is S
        
        batch_input_ids = []
        batch_labels = []
        batch_loss_mask = []
        batch_position_ids = []
        batch_attention_mask = []
        batch_seq_lengths = []
        
        for item in features:
            length = len(item['input_ids'])  # Current 2S
            seq_length = item['seq_length']  # Current S
            pad_length = max_length - length
            
            # Pad input_ids
            input_ids = torch.cat([
                item['input_ids'],
                torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
            ])
            batch_input_ids.append(input_ids)
            
            # Pad labels
            labels = torch.cat([
                item['labels'],
                torch.full((pad_length,), -100, dtype=torch.long)  # -100 is ignore index
            ])
            batch_labels.append(labels)
            
            # Pad loss_mask
            loss_mask = torch.cat([
                item['loss_mask'],
                torch.zeros(pad_length, dtype=torch.long)  # Padded positions don't contribute to loss
            ])
            batch_loss_mask.append(loss_mask)
            
            # Pad position_ids
            position_ids = torch.cat([
                item['position_ids'],
                torch.zeros(pad_length, dtype=torch.long)
            ])
            batch_position_ids.append(position_ids)
            
            # Create 2D attention mask for this sequence
            # Start with the special TiDAR structure
            attn_mask = self.create_tidar_attention_mask(seq_length)
            
            # Pad to max_length x max_length
            if length < max_length:
                padded_mask = torch.zeros(max_length, max_length, dtype=torch.long)
                padded_mask[:length, :length] = attn_mask
                attn_mask = padded_mask
            
            batch_attention_mask.append(attn_mask)
            batch_seq_lengths.append(seq_length)
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "loss_mask": torch.stack(batch_loss_mask),
            "position_ids": torch.stack(batch_position_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "seq_lengths": torch.tensor(batch_seq_lengths),
        }


def compute_tidar_loss(logits, labels, loss_mask, seq_lengths, args):
    """
    Compute TiDAR loss combining AR loss and diffusion loss.
    
    The input sequence is 2S tokens: [clean_tokens, MASK_tokens]
    The labels are 2S tokens: [AR_labels (shifted), Diffusion_labels (unshifted)]
    The loss_mask indicates which positions to include in loss calculation (0=exclude, 1=include)
    
    - AR loss: Computed on first S positions (next-token prediction)
    - Diffusion loss: Computed on second S positions (denoising)
    
    Args:
        logits: Model output logits [batch_size, 2S, vocab_size]
        labels: Ground truth tokens [batch_size, 2S]
        loss_mask: Binary mask for loss calculation [batch_size, 2S]
        seq_lengths: Actual sequence length S for each sample [batch_size]
        args: Training arguments
    
    Returns:
        total_loss, ar_loss, diffusion_loss, metrics
    """
    batch_size, total_seq_length, vocab_size = logits.shape
    
    # Compute cross-entropy loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    
    device = logits.device
    ar_loss_total = torch.tensor(0.0, device=device)
    diffusion_loss_total = torch.tensor(0.0, device=device)
    ar_count = 0
    diffusion_count = 0
    diff_correct_count = 0
    
    for i in range(batch_size):
        seq_len = seq_lengths[i].item()  # S
        
        # 1. AR Loss (first S positions)
        # Predicting positions 0 to S-1 from positions 0 to S-1
        ar_logits = logits[i, :seq_len, :]  # [S-1, vocab_size]
        ar_labels = labels[i, :seq_len]  # [S-1] - shifted labels
        ar_mask = loss_mask[i, :seq_len]  # [S-1] - shifted loss_mask
        
        # Compute AR loss
        ar_loss = loss_fct(ar_logits, ar_labels)
        # Apply both label validity check and loss_mask
        valid_ar = (ar_mask > 0)
        if valid_ar.sum() > 0:
            ar_loss_total += ar_loss[valid_ar].sum()
            ar_count += valid_ar.sum().item()
        
        # 2. Diffusion Loss (second S positions)
        # Predicting all S positions in the masked region
        diff_logits = logits[i, seq_len:seq_len*2, :]  # [S, vocab_size]
        diff_labels = labels[i, seq_len:seq_len*2]  # [S] - original tokens
        diff_mask = loss_mask[i, seq_len:seq_len*2]  # [S] - loss_mask for diffusion
        
        # Compute Diffusion loss
        diff_loss = loss_fct(diff_logits, diff_labels)
        # Apply both label validity check and loss_mask
        valid_diff = (diff_mask > 0)
        if valid_diff.sum() > 0:
            diffusion_loss_total += diff_loss[valid_diff].sum()
            diffusion_count += valid_diff.sum().item()

        
        with torch.no_grad():
            diff_predictions = torch.argmax(diff_logits, dim=-1)
            diff_valid_labels = (diff_labels != -100) & (diff_mask > 0)
            diff_correct = (diff_predictions == diff_labels) & diff_valid_labels
            diff_correct_count += diff_correct.sum().float()
    
    # Average losses
    ar_loss = ar_loss_total / (ar_count + 1e-8)
    diffusion_loss = diffusion_loss_total / (diffusion_count + 1e-8)
    
    # Combine losses with weights
    total_loss = (args.ar_loss_weight * ar_loss + 
                  args.diffusion_loss_weight * diffusion_loss)
    
    # Calculate accuracy (only on positions with loss_mask > 0)
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        valid_labels = (labels != -100) & (loss_mask > 0)
        correct = (predictions == labels) & valid_labels
        accuracy = correct.sum().float() / (valid_labels.sum().float() + 1e-8)

        diff_accuracy = diff_correct_count / diffusion_count
    
    metrics = {
        "ar_loss": ar_loss.detach().item() if isinstance(ar_loss, torch.Tensor) else ar_loss,
        "diffusion_loss": diffusion_loss.detach().item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss,
        "accuracy": accuracy.item(),
        "diff_accuracy": diff_accuracy.item(),
        "ar_tokens": ar_count,
        "diffusion_tokens": diffusion_count,
    }
    
    return total_loss, ar_loss, diffusion_loss, metrics


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load data
datapath = list_files(args.data_dir)
traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(
    traindatapath, 
    tokenizer, 
    max_len=args.max_len
)
testdataset = CustomDataset(
    testdatapath,
    tokenizer,
    max_len=args.max_len
)

# Create checkpoint directory
if rank == 0:
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.model_name):
        os.makedirs(args.model_name)

# Initialize TiDAR model
if rank == 0:
    logger.info("Initializing TiDAR model...")

# Check if we should use an existing TiDAR checkpoint or convert from Qwen
tidar_init_checkpoint = os.path.join(args.ckpt_dir, "tidar_init")
use_existing_tidar_init = os.path.exists(os.path.join(tidar_init_checkpoint, "config.json"))

if use_existing_tidar_init:
    # Load from existing TiDAR initialization checkpoint (fast path)
    if rank == 0:
        logger.info(f"Loading from existing TiDAR initialization: {tidar_init_checkpoint}")
    
    model = TiDARQwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=tidar_init_checkpoint,
        torch_dtype=torch.float16,
        is_training=True,
        device_map=None  # Will be handled by DeepSpeed
    )
else:
    # Convert from Qwen2.5 (slow path, only happens once)
    if rank == 0:
        logger.info(f"Converting from Qwen2.5: {args.qwen_model_path}")
        logger.info("This is a one-time conversion. A TiDAR checkpoint will be saved for future use.")
    
    model = TiDARQwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.qwen_model_path,
        block_size=args.tidar_block_size,
        clean_ratio=args.tidar_clean_ratio,
        use_tidar=True,
        is_training=True,
        torch_dtype=torch.float16,
        device_map=None  # Will be handled by DeepSpeed
    )
    
    # Save the initial TiDAR checkpoint for future use
    if rank == 0:
        logger.info(f"Saving initial TiDAR checkpoint to: {tidar_init_checkpoint}")
        model.save_pretrained(tidar_init_checkpoint)
        logger.info("✓ Initial TiDAR checkpoint saved. Future runs will load this directly.")

# Load existing training checkpoint if specified
if args.existing_model_path is not None:
    if rank == 0:
        logger.info(f"Loading training checkpoint from {args.existing_model_path}")

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == -1:
        local_rank = rank 
    map_location = f"cuda:{local_rank}"
    checkpoint = torch.load(args.existing_model_path, map_location=map_location)
    model.load_state_dict(checkpoint, strict=True)

# Initialize DeepSpeed
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    training_data=traindataset,
    collate_fn=DataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        block_size=args.tidar_block_size
    )
)

if rank == 0:
    logger.info(f"Training started with {len(traindataset)} samples")
    logger.info(f"Batch size: {model_engine.train_batch_size()}")
    logger.info(f"Clean ratio: {args.tidar_clean_ratio}")
    logger.info(f"Block size: {args.tidar_block_size}")

# Training loop
for epoch in range(args.start_epoch, args.num_epochs):
    model_engine.train()
    epoch_loss = 0.0
    epoch_ar_loss = 0.0
    epoch_diffusion_loss = 0.0
    num_batches = 0
    total_correct = 0
    total_tokens = 0
    
    if rank == 0:
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        epoch_iterator = train_loader
    
    if args.run_mode == "debug" and epoch == 1:
        raise ValueError("Debug mode: Successful at epoch == 1")
    
    for batch_idx, batch in enumerate(epoch_iterator):
        if args.run_mode == "debug" and batch_idx == 2:
            if rank == 0:
                logger.info("Debug mode: Break at batch_idx == 2")
            break
        
        # Move batch to device
        batch = {k: v.to(model_engine.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"]
        position_ids = batch["position_ids"]
        attention_mask = batch["attention_mask"]
        seq_lengths = batch["seq_lengths"]
        
        # Forward pass
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        logits = outputs.logits
        
        # Compute loss
        total_loss, ar_loss, diffusion_loss, metrics = compute_tidar_loss(
            logits, labels, loss_mask, seq_lengths, args
        )
        
        # Backward pass
        model_engine.backward(total_loss)
        model_engine.step()
        
        # Accumulate metrics
        epoch_loss += total_loss.detach().item()
        epoch_ar_loss += metrics["ar_loss"]
        epoch_diffusion_loss += metrics["diffusion_loss"]
        num_batches += 1
        
        # Log batch metrics
        if rank == 0:
            writer.add_scalar("train/batch_loss", total_loss.detach().item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("train/batch_ar_loss", metrics["ar_loss"], epoch * len(train_loader) + batch_idx)
            writer.add_scalar("train/batch_diffusion_loss", metrics["diffusion_loss"], epoch * len(train_loader) + batch_idx)
            writer.add_scalar("train/batch_accuracy", metrics["accuracy"], epoch * len(train_loader) + batch_idx)
            writer.add_scalar("train/batch_accuracy_diff", metrics["diff_accuracy"], epoch * len(train_loader) + batch_idx)
            writer.add_scalar("train/lr", optimizer.optimizer.param_groups[0]["lr"], epoch * len(train_loader) + batch_idx)
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx}: "
                    f"Loss={total_loss.detach().item():.4f}, "
                    f"AR Loss={metrics['ar_loss']:.4f}, "
                    f"Diff Loss={metrics['diffusion_loss']:.4f}, "
                    f"Acc={metrics['accuracy']:.4f}"
                    f"Acc_diff={metrics['diff_accuracy']:.4f}"
                )
    
    # Epoch summary
    epoch_loss /= num_batches
    epoch_ar_loss /= num_batches
    epoch_diffusion_loss /= num_batches
    
    if rank == 0:
        logger.info(
            f"Epoch {epoch} Summary: "
            f"Loss={epoch_loss:.4f}, "
            f"AR Loss={epoch_ar_loss:.4f}, "
            f"Diffusion Loss={epoch_diffusion_loss:.4f}"
        )
        
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("train/epoch_ar_loss", epoch_ar_loss, epoch)
        writer.add_scalar("train/epoch_diffusion_loss", epoch_diffusion_loss, epoch)
    
    # Save checkpoint
    if args.run_mode == "train" and (epoch % args.save_interval == 0 or epoch == args.num_epochs - 1):
        if rank == 0:
            logger.info(f"Saving checkpoint for epoch {epoch}")
        model_engine.save_16bit_model(f"{args.ckpt_dir}/epoch_{epoch}")

if rank == 0:
    logger.info("Training completed!")
    writer.close()
