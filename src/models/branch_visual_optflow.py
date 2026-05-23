from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class TemporalFlowBranch(nn.Module):
    def __init__(self, projection_dim: int = 256, pretrained: bool = True, flow_scale: float = 20.0) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            2,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained and old_conv.weight is not None:
            with torch.no_grad():
                averaged = old_conv.weight.mean(dim=1, keepdim=True)
                model.conv1.weight.copy_(averaged.repeat(1, 2, 1, 1))
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.flow_scale = float(flow_scale)
        self.projection = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    def forward(self, optflow_frames: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps = optflow_frames.shape[:2]
        flat = optflow_frames.reshape(batch_size * timesteps, *optflow_frames.shape[2:])
        flat = flat / self.flow_scale
        features = self.backbone(flat)
        projected = self.projection(features)
        return projected.reshape(batch_size, timesteps, -1)
