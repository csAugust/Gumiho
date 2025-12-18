# -*- coding: utf-8 -*-
"""
Hybrid-AgileDrafter: 动态异构序列草稿模型
将 Gumiho 重构为可微门控网络驱动的自适应推理系统
"""

__version__ = "1.0.0"
__author__ = "Hybrid-AgileDrafter Team"

from dynaspec.gating_network import GatingNetwork
from dynaspec.hybrid_agile_drafter import HybridAgileDrafter
from dynaspec.hybrid_model import HybridAgileDrafterModel
from dynaspec.loss import compute_hybrid_loss, compute_accuracy

__all__ = [
    "GatingNetwork",
    "HybridAgileDrafter",
    "HybridAgileDrafterModel",
    "compute_hybrid_loss",
    "compute_accuracy",
]
