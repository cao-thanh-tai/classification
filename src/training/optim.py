"""Optimizer construction."""

from torch import nn
from torch.optim import AdamW, Optimizer


def build_optimizer(model: nn.Module, name: str, learning_rate: float, weight_decay: float) -> Optimizer:
    if name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

