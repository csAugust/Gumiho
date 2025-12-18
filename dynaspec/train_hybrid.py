# -*- coding: utf-8 -*-
"""
Hybrid-AgileDrafter 训练脚本
基于 Gumiho 的训练流程，使用新的 HybridAgileDrafter 模型和损失函数
"""

import os
os.environ['MIOPEN_DISABLE_CACHE'] = '1'
import argparse
from loguru import logger
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynaspec.hybrid_model import HybridAgileDrafterModel
from dynaspec.loss import compute_hybrid_loss, compute_accuracy
from gumiho.train.main import CustomDataset, DataCollatorWithPadding, AddUniformNoise, AddGaussianNoise, list_files


# ===== 参数解析 =====
parser = argparse.ArgumentParser(description='Hybrid-AgileDrafter Training')
parser.add_argument('--basepath', type=str, required=True, help='Base model path')
parser.add_argument('--configpath', type=str, default="dynaspec/config.json", help='Config file path')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, required=True, help='Pre-generated data directory')
parser.add_argument('--cpdir', type=str, default='./checkpoints/hybrid_agile', help='Checkpoint directory')
parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'debug'])
parser.add_argument('--logger_file', type=str, default='hybrid_agile_train')
parser.add_argument('--resume_from', type=int, default=0, help='Resume from epoch')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--warmup_step', type=int, default=6000, help='Warmup steps')
parser.add_argument('--total_step', type=float, default=8.0, help='Total steps (in 100k)')
parser.add_argument('--cost_lambda', type=float, default=0.01, help='Cost loss weight')
parser.add_argument('--mlp_cost', type=float, default=1.0, help='MLP block cost')
parser.add_argument('--transformer_cost', type=float, default=5.0, help='Transformer block cost')
parser.add_argument('--max_len', type=int, default=2048, help='Max sequence length')
parser.add_argument('--num_workers', type=int, default=2, help='Number of data workers')
parser.add_argument('--data_noise', action='store_true', help='Add noise to data')
parser.add_argument('--noise_type', type=str, default='uniform', choices=['uniform', 'gaussian'])
parser.add_argument('--noise_std', type=float, default=0.2, help='Noise std')
parser.add_argument('--save_freq', type=int, default=5, help='Save frequency (epochs)')
parser.add_argument('--test_freq', type=int, default=1, help='Test frequency (epochs)')
parser.add_argument('--grad_clip', type=float, default=0.5, help='Gradient clipping value')

args = parser.parse_args()

# ===== 日志配置 =====
if args.run_mode == "train":
    logger_level = "INFO"
elif args.run_mode == "debug":
    logger_level = "DEBUG"
    os.environ['WANDB_MODE'] = 'disabled'

logger.remove()
logger.add(f"./{args.cpdir}/{args.logger_file}.log", level=logger_level, mode="w")

# ===== 训练配置 =====
train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": args.tmpdir,
    "num_epochs": args.num_epochs,
    "num_warmup_steps": args.warmup_step,
    "total_steps": args.total_step * 100000,
    "cost_lambda": args.cost_lambda,
    "mlp_cost": args.mlp_cost,
    "transformer_cost": args.transformer_cost,
    "max_len": args.max_len,
    "num_workers": args.num_workers,
    "data_noise": args.data_noise,
    "noise_type": args.noise_type,
    "noise_std": args.noise_std,
    "grad_clip": args.grad_clip,
    "save_freq": args.save_freq,
    "b1": 0.9,
    "b2": 0.95,
}

logger.info(f"Training config: {train_config}")
logger.info(f"Arguments: {args}")

# ===== Accelerator 初始化 =====
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(
    mixed_precision='bf16',
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"]
)

# ===== WandB 初始化 =====
if accelerator.is_main_process:
    import wandb
    wandb.init(
        project="hybrid-agile-drafter",
        name=args.logger_file,
        config=train_config
    )

# ===== 数据准备 =====
logger.info("Loading data...")

# 数据增强
if train_config["data_noise"]:
    if train_config["noise_type"] == "uniform":
        aug = AddUniformNoise(std=train_config["noise_std"])
    else:
        aug = AddGaussianNoise(mean=0.0, std=train_config["noise_std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])
traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

logger.info(f"Train samples: {len(traindatapath)}, Test samples: {len(testdatapath)}")

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)

train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    shuffle=True,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"],
    pin_memory=True
)

test_loader = DataLoader(
    testdataset,
    batch_size=train_config["bs"],
    shuffle=False,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"],
    pin_memory=True
)

# ===== 模型初始化 =====
logger.info("Initializing model...")
model = HybridAgileDrafterModel.from_pretrained(
    base_model_path=args.basepath,
    config_path=args.configpath if os.path.exists(args.configpath) else None,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 只训练 draft_model，冻结 base_model
for param in model.base_model.parameters():
    param.requires_grad = False

logger.info(f"Model initialized. Draft model has {sum(p.numel() for p in model.draft_model.parameters())} parameters")

# ===== 优化器和调度器 =====
optimizer = optim.AdamW(
    model.draft_model.parameters(),
    lr=train_config["lr"],
    betas=(train_config["b1"], train_config["b2"])
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=train_config["num_warmup_steps"],
    num_training_steps=train_config["total_steps"]
)

# ===== Accelerator 准备 =====
model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)

# ===== 从检查点恢复 =====
start_epoch = args.resume_from
if start_epoch > 0:
    logger.info(f"Resuming from epoch {start_epoch}")
    checkpoint_path = f"{args.cpdir}/epoch{start_epoch}"
    if os.path.exists(checkpoint_path):
        accelerator.load_state(checkpoint_path)
    else:
        logger.warning(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        start_epoch = 0

# 创建检查点目录
if accelerator.is_main_process:
    os.makedirs(args.cpdir, exist_ok=True)

# ===== 训练循环 =====
logger.info("Starting training...")

for epoch in range(start_epoch, train_config["num_epochs"]):
    # ===== 训练阶段 =====
    model.train()
    model.draft_model.train()
    
    epoch_loss = 0
    num_batches = 0
    epoch_metrics = {
        "draft_loss": 0,
        "expected_cost": 0,
        "top1_acc": 0,
        "top3_acc": 0,
        "top5_acc": 0
    }
    
    if accelerator.is_main_process:
        tqdm_desc = f"Epoch {epoch+1}/{train_config['num_epochs']}"
        logger.info(tqdm_desc)
        epoch_iterator = tqdm(train_loader, desc=tqdm_desc)
    else:
        epoch_iterator = train_loader
    
    for batch_idx, data in enumerate(epoch_iterator):
        if args.run_mode == "debug" and batch_idx > 3:
            logger.debug("Debug mode: breaking after 3 batches")
            break
        
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            
            # 获取 LLM 隐藏状态
            hidden_states = data["hidden_states"]
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            loss_mask = data["loss_mask"]
            
            # 调用草稿模型
            draft_logits, gating_probs = model.draft_forward(
                input_ids=input_ids,
                llm_hidden_states=hidden_states,
                attention_mask=attention_mask
            )
            
            # 计算损失
            loss, loss_dict = compute_hybrid_loss(
                draft_logits=draft_logits,
                target_labels=input_ids,
                gating_probs=gating_probs,
                num_mlp_blocks=model.draft_model.num_mlp_blocks,
                num_transformer_blocks=model.draft_model.num_transformer_blocks,
                cost_lambda=train_config["cost_lambda"],
                mlp_cost=train_config["mlp_cost"],
                transformer_cost=train_config["transformer_cost"],
                loss_mask=loss_mask
            )
            
            # 反向传播
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.draft_model.parameters(), train_config["grad_clip"])
            optimizer.step()
            scheduler.step()
            
            # 计算准确率
            acc_dict = compute_accuracy(
                draft_logits=draft_logits,
                target_labels=input_ids,
                loss_mask=loss_mask,
                topk=(1, 3, 5)
            )
            
            # 累积指标
            epoch_loss += loss.item()
            epoch_metrics["draft_loss"] += loss_dict["draft_loss"].item()
            epoch_metrics["expected_cost"] += loss_dict["expected_cost"].item()
            epoch_metrics["top1_acc"] += acc_dict["top1_acc"]
            epoch_metrics["top3_acc"] += acc_dict["top3_acc"]
            epoch_metrics["top5_acc"] += acc_dict["top5_acc"]
            num_batches += 1
            
            # 记录到 WandB
            if accelerator.is_main_process and num_batches % 10 == 0:
                log_dict = {
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/loss": loss.item(),
                    "train/draft_loss": loss_dict["draft_loss"].item(),
                    "train/expected_cost": loss_dict["expected_cost"].item(),
                    "train/cost_loss": loss_dict["cost_loss"].item(),
                    "train/top1_acc": acc_dict["top1_acc"],
                    "train/top3_acc": acc_dict["top3_acc"],
                    "train/top5_acc": acc_dict["top5_acc"],
                }
                
                # 添加门控统计
                for key, value in loss_dict.items():
                    if key.startswith("gating/"):
                        log_dict[f"train/{key}"] = value.item() if torch.is_tensor(value) else value
                
                wandb.log(log_dict)
    
    # 计算平均指标
    avg_epoch_loss = epoch_loss / num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    if accelerator.is_main_process:
        logger.info(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, "
                   f"Draft Loss: {epoch_metrics['draft_loss']:.4f}, "
                   f"Cost: {epoch_metrics['expected_cost']:.4f}, "
                   f"Top1 Acc: {epoch_metrics['top1_acc']:.2%}")
        
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_draft_loss": epoch_metrics["draft_loss"],
            "train/epoch_cost": epoch_metrics["expected_cost"],
            "train/epoch_top1_acc": epoch_metrics["top1_acc"],
            "epoch": epoch + 1
        })
    
    # ===== 测试阶段 =====
    if (epoch + 1) % args.test_freq == 0:
        model.eval()
        model.draft_model.eval()
        
        test_loss = 0
        test_num_batches = 0
        test_metrics = {
            "draft_loss": 0,
            "expected_cost": 0,
            "top1_acc": 0,
            "top3_acc": 0,
            "top5_acc": 0
        }
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing")):
                if args.run_mode == "debug" and batch_idx > 2:
                    break
                
                hidden_states = data["hidden_states"]
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                loss_mask = data["loss_mask"]
                
                draft_logits, gating_probs = model.draft_forward(
                    input_ids=input_ids,
                    llm_hidden_states=hidden_states,
                    attention_mask=attention_mask
                )
                
                loss, loss_dict = compute_hybrid_loss(
                    draft_logits=draft_logits,
                    target_labels=input_ids,
                    gating_probs=gating_probs,
                    num_mlp_blocks=model.draft_model.num_mlp_blocks,
                    num_transformer_blocks=model.draft_model.num_transformer_blocks,
                    cost_lambda=train_config["cost_lambda"],
                    mlp_cost=train_config["mlp_cost"],
                    transformer_cost=train_config["transformer_cost"],
                    loss_mask=loss_mask
                )
                
                acc_dict = compute_accuracy(
                    draft_logits=draft_logits,
                    target_labels=input_ids,
                    loss_mask=loss_mask,
                    topk=(1, 3, 5)
                )
                
                test_loss += loss.item()
                test_metrics["draft_loss"] += loss_dict["draft_loss"].item()
                test_metrics["expected_cost"] += loss_dict["expected_cost"].item()
                test_metrics["top1_acc"] += acc_dict["top1_acc"]
                test_metrics["top3_acc"] += acc_dict["top3_acc"]
                test_metrics["top5_acc"] += acc_dict["top5_acc"]
                test_num_batches += 1
        
        # 计算平均测试指标
        avg_test_loss = test_loss / test_num_batches
        for key in test_metrics:
            test_metrics[key] /= test_num_batches
        
        if accelerator.is_main_process:
            logger.info(f"Test - Loss: {avg_test_loss:.4f}, "
                       f"Draft Loss: {test_metrics['draft_loss']:.4f}, "
                       f"Cost: {test_metrics['expected_cost']:.4f}, "
                       f"Top1 Acc: {test_metrics['top1_acc']:.2%}")
            
            wandb.log({
                "test/loss": avg_test_loss,
                "test/draft_loss": test_metrics["draft_loss"],
                "test/expected_cost": test_metrics["expected_cost"],
                "test/top1_acc": test_metrics["top1_acc"],
                "test/top3_acc": test_metrics["top3_acc"],
                "test/top5_acc": test_metrics["top5_acc"],
                "epoch": epoch + 1
            })
    
    # ===== 保存检查点 =====
    if accelerator.is_main_process and (epoch + 1) % args.save_freq == 0:
        checkpoint_path = f"{args.cpdir}/epoch{epoch+1}"
        accelerator.save_state(output_dir=checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

logger.info("Training completed!")
