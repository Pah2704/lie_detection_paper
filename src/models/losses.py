from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def class_alpha_from_counts(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    inv = 1.0 / counts.clamp_min(1.0)
    return inv / inv.sum() * num_classes


def build_loss(name: str, labels: torch.Tensor | None = None, gamma: float = 2.0) -> nn.Module:
    normalized = name.lower()
    if normalized in {"ce", "cross_entropy", "crossentropy"}:
        return nn.CrossEntropyLoss()
    if normalized == "focal":
        alpha = class_alpha_from_counts(labels) if labels is not None else None
        return FocalLoss(gamma=gamma, alpha=alpha)
    raise ValueError(f"Unknown loss: {name}")
