# -*- coding: utf-8 -*-
"""
Hybrid-AgileDrafter DeepSpeed 训练脚本
基于 Gumiho 和 TiDAR 的 DeepSpeed 训练流程
"""

import argparse
import deepspeed
from loguru import logger
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynaspec.hybrid_model import HybridAgileDrafterModel
from dynaspec.loss import compute_hybrid_loss, compute_accuracy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List
from tqdm import tqdm
from safetensors import safe_open

torch.backends.cuda.matmul.allow_tf32 = True

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description='Hybrid-AgileDrafter DeepSpeed Training')
parser.add_argument('--config_path', type=str, default='dynaspec/train_config.json',
                    help='Path to the training configuration file')
parser.add_argument("--local_rank", type=int, default=-1, 
                    help="local_rank for distributed training on gpus")
parser.add_argument('--existing_model_path', type=str, default=None,
                    help='Path to existing model checkpoint to resume from')

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# ===== Load Configuration =====
if os.path.exists(args.config_path):
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        # Flatten nested config and add to args
        for category, config_dict in config.items():
            for key, value in config_dict.items():
                setattr(args, key, value)
else:
    logger.error(f"Config file not found: {args.config_path}")
    sys.exit(1)

logger.info(f"Arguments: {args}")
logger.remove()

# ===== Initialize DeepSpeed Distributed =====
deepspeed.init_distributed()
rank = torch.distributed.get_rank()

# ===== Setup Logging =====
if rank == 0:
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.getenv('TENSORBOARD_LOG_PATH', './runs/hybrid_agile')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create log directory
    os.makedirs(os.path.dirname(f"./{args.model_name}/{args.logger_file}.log"), exist_ok=True)
    logger.add(f"./{args.model_name}/{args.logger_file}.log", level="INFO", mode="w")
    logger.info(f"Config loaded: {args}")


# ===== Data Loading Utilities =====
def list_files(path):
    """Recursively list all files in a directory"""
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None, max_len=2048):
        self.data = datapath
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:self.max_len][None, :]
        input_ids = data['input_ids'][:self.max_len][None, :]
        loss_mask = data["loss_mask"][:self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


# ===== Load LLM Head =====
if rank == 0:
    logger.info("Loading LLM head...")

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
head.weight.data = tensor

for param in head.parameters():
    param.requires_grad = False


# ===== Prepare Data =====
if rank == 0:
    logger.info("Loading data...")

# Data augmentation
if args.data_noise:
    if args.noise_type == "uniform":
        aug = AddUniformNoise(std=args.noise_std)
    else:
        aug = AddGaussianNoise(mean=args.noise_mean, std=args.noise_std)
else:
    aug = None

datapath = list_files(args.data_dir)
traindatapath = datapath[:int(len(datapath) * 1.0)]  # Use all data for training

traindataset = CustomDataset(traindatapath, transform=aug, max_len=args.max_len)

if rank == 0:
    logger.info(f"Training data size: {len(traindataset)}")

# ===== Create Checkpoint Directory =====
if rank == 0:
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)


# ===== Initialize Model =====
if rank == 0:
    logger.info("Initializing Hybrid-AgileDrafter model...")

# Load config
if os.path.exists(args.configpath):
    with open(args.configpath, 'r') as f:
        model_config_dict = json.load(f)
    model_config = type('Config', (), model_config_dict)()
else:
    # Use default config from args
    model_config = type('Config', (), {
        'num_mlp_blocks': args.num_mlp_blocks,
        'num_transformer_blocks': args.num_transformer_blocks,
        'gumbel_temperature': args.gumbel_temperature
    })()

# Initialize HybridAgileDrafterModel
model = HybridAgileDrafterModel.from_pretrained(
    base_model_path=args.basepath,
    config_path=args.configpath if os.path.exists(args.configpath) else None,
    torch_dtype=torch.float16,
    device_map=None  # Will be handled by DeepSpeed
)

# Freeze base_model parameters
for param in model.base_model.parameters():
    param.requires_grad = False

# Only train draft_model
for param in model.draft_model.parameters():
    param.requires_grad = True

if rank == 0:
    total_params = sum(p.numel() for p in model.draft_model.parameters())
    trainable_params = sum(p.numel() for p in model.draft_model.parameters() if p.requires_grad)
    logger.info(f"Draft model - Total params: {total_params:,}, Trainable: {trainable_params:,}")


# ===== Load Existing Checkpoint (if specified) =====
if args.existing_model_path is not None:
    if rank == 0:
        logger.info(f"Loading checkpoint from {args.existing_model_path}")
    
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == -1:
        local_rank = rank
    map_location = f"cuda:{local_rank}"
    
    checkpoint = torch.load(args.existing_model_path, map_location=map_location)
    model.load_state_dict(checkpoint, strict=True)


# ===== Initialize DeepSpeed =====
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    args=args,
    model=model.draft_model,  # Only train draft_model
    model_parameters=model.draft_model.parameters(),
    training_data=traindataset,
    collate_fn=DataCollatorWithPadding()
)

# Move head to device
head_engine = head.half().to(rank)
head_engine.eval()

# Move base_model to device
model.base_model = model.base_model.to(rank)
model.base_model.eval()

if rank == 0:
    logger.info(f"Training started with batch size: {model_engine.train_batch_size()}")
    logger.info(f"Gradient accumulation steps: {model_engine.gradient_accumulation_steps()}")


# ===== Training Loop =====
for epoch in range(args.start_epoch, args.num_epochs):
    model_engine.train()
    
    epoch_loss = 0.0
    epoch_metrics = {
        "draft_loss": 0.0,
        "expected_cost": 0.0,
        "top1_acc": 0.0,
        "top3_acc": 0.0,
    }
    num_batches = 0
    
    if rank == 0:
        tqdm_desc = f"Epoch {epoch+1}/{args.num_epochs}"
        epoch_iterator = tqdm(train_loader, desc=tqdm_desc)
    else:
        epoch_iterator = train_loader
    
    if args.run_mode == "debug" and epoch == 1:
        raise ValueError("Debug mode: Successful at epoch == 1")
    
    for batch_idx, data in enumerate(epoch_iterator):
        if args.run_mode == "debug" and batch_idx == 2:
            if rank == 0:
                logger.info("Debug mode: Break at batch_idx == 2")
            break
        
        model_engine.zero_grad()
        
        # Move data to device
        hidden_states = data["hidden_states"].to(rank).half()
        input_ids = data["input_ids"].to(rank)
        attention_mask = data["attention_mask"].to(rank)
        loss_mask = data["loss_mask"].to(rank)
        
        # Get draft model output
        # Note: We need to get inputs_embeds from base_model
        with torch.no_grad():
            inputs_embeds = model.base_model.model.embed_tokens(input_ids)
        
        # Draft forward pass
        draft_logits, gating_probs = model_engine(
            inputs=inputs_embeds.half(),
            llm_hidden_state=hidden_states,
            attention_mask=attention_mask
        )
        
        # Compute loss
        loss, loss_dict = compute_hybrid_loss(
            draft_logits=draft_logits,
            target_labels=input_ids,
            gating_probs=gating_probs,
            num_mlp_blocks=args.num_mlp_blocks,
            num_transformer_blocks=args.num_transformer_blocks,
            cost_lambda=args.cost_lambda,
            mlp_cost=args.mlp_cost,
            transformer_cost=args.transformer_cost,
            loss_mask=loss_mask
        )
        
        # Backward pass
        model_engine.backward(loss)
        model_engine.step()
        
        # Compute accuracy
        with torch.no_grad():
            acc_dict = compute_accuracy(
                draft_logits=draft_logits,
                target_labels=input_ids,
                loss_mask=loss_mask,
                topk=(1, 3, 5)
            )
        
        # Accumulate metrics
        epoch_loss += loss.detach().item()
        epoch_metrics["draft_loss"] += loss_dict["draft_loss"].item()
        epoch_metrics["expected_cost"] += loss_dict["expected_cost"].item()
        epoch_metrics["top1_acc"] += acc_dict["top1_acc"]
        epoch_metrics["top3_acc"] += acc_dict["top3_acc"]
        num_batches += 1
        
        # Log to TensorBoard
        if rank == 0:
            global_step = epoch * len(train_loader) + batch_idx
            
            writer.add_scalar("train/batch_loss", loss.detach().item(), global_step)
            writer.add_scalar("train/batch_draft_loss", loss_dict["draft_loss"].item(), global_step)
            writer.add_scalar("train/batch_expected_cost", loss_dict["expected_cost"].item(), global_step)
            writer.add_scalar("train/batch_cost_loss", loss_dict["cost_loss"].item(), global_step)
            writer.add_scalar("train/batch_top1_acc", acc_dict["top1_acc"], global_step)
            writer.add_scalar("train/batch_top3_acc", acc_dict["top3_acc"], global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            
            # Log gating statistics
            for key, value in loss_dict.items():
                if key.startswith("gating/"):
                    writer.add_scalar(f"train/{key}", 
                                    value.item() if torch.is_tensor(value) else value, 
                                    global_step)
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx}: "
                    f"Loss={loss.detach().item():.4f}, "
                    f"Draft Loss={loss_dict['draft_loss'].item():.4f}, "
                    f"Cost={loss_dict['expected_cost'].item():.4f}, "
                    f"Top1 Acc={acc_dict['top1_acc']:.4f}"
                )
                
                # Log gating probabilities
                gating_info = ", ".join([
                    f"{k.replace('gating/', '')}={v.item() if torch.is_tensor(v) else v:.3f}"
                    for k, v in loss_dict.items() if k.startswith("gating/")
                ])
                if gating_info:
                    logger.info(f"  Gating: {gating_info}")
    
    # Epoch summary
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    if rank == 0:
        logger.info(
            f"Epoch {epoch+1} Summary: "
            f"Loss={epoch_loss:.4f}, "
            f"Draft Loss={epoch_metrics['draft_loss']:.4f}, "
            f"Cost={epoch_metrics['expected_cost']:.4f}, "
            f"Top1 Acc={epoch_metrics['top1_acc']:.4f}"
        )
        
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("train/epoch_draft_loss", epoch_metrics["draft_loss"], epoch)
        writer.add_scalar("train/epoch_cost", epoch_metrics["expected_cost"], epoch)
        writer.add_scalar("train/epoch_top1_acc", epoch_metrics["top1_acc"], epoch)
        writer.add_scalar("epoch", epoch, epoch)
    
    # Save checkpoint
    if args.run_mode == "train" and (epoch % args.save_interval == 0 or epoch == args.num_epochs - 1):
        if rank == 0:
            logger.info(f"Saving checkpoint for epoch {epoch}")
        model_engine.save_16bit_model(f"{args.ckpt_dir}/epoch_{epoch}")

if rank == 0:
    logger.info("Training completed!")
    writer.close()
