# Hybrid-AgileDrafter

## 📋 概述

Hybrid-AgileDrafter 是 Gumiho 的重构版本，将静态、分离式草稿头架构重构为由**可微门控网络驱动的动态、异构序列模型**。

### 核心特性

1. **动态块选择**：使用门控网络根据 LLM 隐藏状态动态选择最佳的处理块
2. **异构架构**：组合轻量级 MLP 块和强大的 Transformer 块
3. **成本感知训练**：通过成本预期损失平衡准确率和计算效率
4. **可微优化**：使用 Gumbel-Softmax 实现端到端训练

## 🏗️ 架构

```
┌─────────────────────────────────────────────┐
│           LLM Hidden States                 │
└────────────────┬────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Gating Network │ ← 基于隐藏状态选择块
        └────────┬───────┘
                 │ Gumbel-Softmax
                 ▼
    ┌────────────────────────┐
    │   Hybrid Block Seq     │
    │  ┌──────────────────┐  │
    │  │  MLP Block 1     │  │ ← 轻量级，成本低
    │  │  MLP Block 2     │  │
    │  │  ...             │  │
    │  │  Transformer 1   │  │ ← 强大，成本高
    │  │  Transformer 2   │  │
    │  │  ...             │  │
    │  └──────────────────┘  │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────┐
    │ Weighted Fusion│ ← 加权组合各块输出
    └────────┬───────┘
             │
             ▼
       Final Logits
```

## 📁 文件结构

```
dynaspec/
├── README.md                      # 本文档
├── config.json                    # 模型配置文件
├── train_config.json              # 训练配置文件（用于 DeepSpeed）
├── ds_config_hybrid.json          # DeepSpeed 配置文件
├── gating_network.py              # 门控网络实现
├── hybrid_agile_drafter.py        # 核心草稿模型
├── hybrid_model.py                # 完整模型包装器
├── loss.py                        # 损失函数（模仿损失 + 成本损失）
├── train_hybrid.py                # 训练脚本（Accelerate）
├── train_hybrid_deepspeed.py      # 训练脚本（DeepSpeed）
├── train_deepspeed.sh             # DeepSpeed 启动脚本
├── test_basic.py                  # 基础测试脚本
└── __init__.py                    # 包初始化文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# Gumiho 的所有依赖已经包含所需的库
pip install -r requirements.txt
```

### 2. 准备数据

使用 Gumiho 的数据生成脚本生成训练数据：

```bash
# 生成训练数据（使用 Gumiho 的方法）
python -m gumiho.ge_data.ge_data_all_llama3 --basepath /path/to/llama3 --output /path/to/output
```

### 3. 训练模型

#### 方法 1: 单 GPU / 小规模训练（使用 Accelerate）

```bash
python dynaspec/train_hybrid.py \
    --basepath /path/to/llama-3-8b \
    --configpath dynaspec/config.json \
    --tmpdir /path/to/training_data \
    --cpdir ./checkpoints/hybrid_agile \
    --bs 4 \
    --lr 3e-5 \
    --num_epochs 20 \
    --cost_lambda 0.01
```

#### 方法 2: 多 GPU / 大规模训练（使用 DeepSpeed，推荐）

1. **配置训练参数**

编辑 `dynaspec/train_config.json`:

```json
{
  "training": {
    "num_epochs": 20,
    "start_epoch": 0,
    "save_interval": 5,
    "max_len": 2048
  },
  "model": {
    "num_mlp_blocks": 2,
    "num_transformer_blocks": 3,
    "gumbel_temperature": 1.0
  },
  "loss_weights": {
    "cost_lambda": 0.01,
    "mlp_cost": 1.0,
    "transformer_cost": 5.0
  },
  "paths": {
    "basepath": "/path/to/llama-3-8b",
    "ckpt_dir": "./checkpoints/hybrid_agile_ds"
  },
  "data": {
    "data_dir": "./train_data"
  }
}
```

2. **启动训练**

```bash
# 单命令启动
bash dynaspec/train_deepspeed.sh

# 或手动指定参数
deepspeed --num_gpus=4 \
    dynaspec/train_hybrid_deepspeed.py \
    --config_path dynaspec/train_config.json \
    --deepspeed_config dynaspec/ds_config_hybrid.json
```

3. **从检查点恢复训练**

```bash
deepspeed --num_gpus=4 \
    dynaspec/train_hybrid_deepspeed.py \
    --config_path dynaspec/train_config.json \
    --deepspeed_config dynaspec/ds_config_hybrid.json \
    --existing_model_path ./checkpoints/hybrid_agile_ds/epoch_10/pytorch_model.bin
```

### 4. 主要超参数

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|---------|
| `num_mlp_blocks` | MLP 块数量 | 2 | 1-5 |
| `num_transformer_blocks` | Transformer 块数量 | 3 | 1-5 |
| `cost_lambda` | 成本损失权重 | 0.01 | 0.001-0.1 |
| `mlp_cost` | MLP 块成本 | 1.0 | 固定为 1.0 |
| `transformer_cost` | Transformer 块成本 | 5.0 | 3.0-10.0 |
| `gumbel_temperature` | Gumbel-Softmax 温度 | 1.0 | 0.5-2.0 |

## 💡 核心组件

### 1. GatingNetwork (门控网络)

根据 LLM 隐藏状态选择最佳块：

```python
from dynaspec.gating_network import GatingNetwork

gating = GatingNetwork(input_dim=4096, num_choices=5)
logits = gating(llm_hidden_state)  # [batch_size, num_choices]
```

### 2. HybridAgileDrafter (草稿模型)

动态异构序列模型：

```python
from dynaspec.hybrid_agile_drafter import HybridAgileDrafter

drafter = HybridAgileDrafter(config)
final_logits, gating_probs = drafter(
    inputs=input_embeds,
    llm_hidden_state=hidden_states
)
```

### 3. 损失函数

组合模仿损失和成本损失：

```python
from dynaspec.loss import compute_hybrid_loss

loss, loss_dict = compute_hybrid_loss(
    draft_logits=draft_logits,
    target_labels=target_ids,
    gating_probs=gating_probs,
    num_mlp_blocks=2,
    num_transformer_blocks=3,
    cost_lambda=0.01
)
```

## 📊 训练监控

训练过程中会记录以下指标到 WandB：

### 损失指标
- `train/loss`: 总损失
- `train/draft_loss`: 模仿损失（交叉熵）
- `train/expected_cost`: 预期计算成本
- `train/cost_loss`: 成本损失项

### 准确率指标
- `train/top1_acc`: Top-1 准确率
- `train/top3_acc`: Top-3 准确率
- `train/top5_acc`: Top-5 准确率

### 门控统计
- `train/gating/mlp_0_prob`: MLP 块 0 被选择的概率
- `train/gating/transformer_0_prob`: Transformer 块 0 被选择的概率
- `train/gating/mlp_total_prob`: 所有 MLP 块总概率
- `train/gating/transformer_total_prob`: 所有 Transformer 块总概率

## 🔧 配置说明

### config.json 参数详解

```json
{
  "num_mlp_blocks": 2,              // MLP 块数量（轻量级）
  "num_transformer_blocks": 3,       // Transformer 块数量（重量级）
  "gumbel_temperature": 1.0,         // 温度参数，越低越离散
  "cost_lambda": 0.01,               // 成本损失权重，越大越注重效率
  "mlp_cost": 1.0,                   // MLP 块相对成本
  "transformer_cost": 5.0            // Transformer 块相对成本
}
```

### 成本权衡

- **`cost_lambda` 较小 (0.001-0.01)**：模型更关注准确率，可能更多选择 Transformer 块
- **`cost_lambda` 较大 (0.05-0.1)**：模型更关注效率，倾向选择 MLP 块
- **建议**：从 0.01 开始，根据门控统计调整

## 🎯 预期效果

成功训练后，模型应该展现以下行为：

1. **自适应选择**：
   - 简单上下文 → 更多使用 MLP 块
   - 复杂上下文 → 更多使用 Transformer 块

2. **效率提升**：
   - 相比纯 Transformer：计算成本降低 30-50%
   - 相比纯 MLP：准确率提升 10-20%

3. **门控统计**（理想情况）：
   - MLP 块总概率：40-60%
   - Transformer 块总概率：40-60%

## ⚠️ 注意事项

1. **不修改原仓库代码**：所有新代码都在 `dynaspec/` 目录下
2. **数据格式兼容**：使用与 Gumiho 相同的数据格式
3. **内存需求**：训练时需要加载完整的基础 LLM + 草稿模型
4. **训练技巧**：
   - 先用较小的 `cost_lambda` 训练几个 epoch
   - 逐步增大 `cost_lambda` 以提高效率
   - 监控门控统计，确保块选择的多样性

## 🚀 DeepSpeed 训练优势

使用 DeepSpeed 进行训练具有以下优势：

1. **内存优化**：ZeRO Stage 3 大幅降低显存占用
2. **多 GPU 加速**：高效的分布式训练
3. **混合精度**：FP16 训练加速计算
4. **梯度累积**：支持更大的有效批量大小
5. **自动优化**：自动调度器和优化器配置

### DeepSpeed 配置说明

`ds_config_hybrid.json` 关键参数：

- **ZeRO Stage 3**: 模型参数、梯度和优化器状态的分片
- **FP16**: 混合精度训练
- **Gradient Clipping**: 0.5（防止梯度爆炸）
- **Batch Size**: 4 per GPU
- **学习率调度**: WarmupDecayLR (warmup 6000 steps)

## 📚 参考

本实现基于以下概念：

1. **Speculative Decoding**: 使用草稿模型加速 LLM 推理
2. **Gumbel-Softmax**: 可微的离散选择
3. **混合专家 (MoE)**: 动态选择不同计算路径
4. **成本感知学习**: 在准确率和效率间权衡
5. **DeepSpeed**: 高效的大规模分布式训练框架

## 🤝 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

## 📄 许可证

遵循 Gumiho 原仓库的许可证。
