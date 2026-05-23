from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class SpatialViTBranch(nn.Module):
    def __init__(
        self,
        model_name: str = "LaurenGurgiolo/vit-micro-facial-expressions",
        projection_dim: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.freeze = freeze
        hidden_size = int(self.backbone.config.hidden_size)
        if freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    def forward(self, rgb_frames: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps = rgb_frames.shape[:2]
        flat = rgb_frames.reshape(batch_size * timesteps, *rgb_frames.shape[2:])
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.backbone(pixel_values=flat)
            cls = outputs.last_hidden_state[:, 0]
        projected = self.projection(cls)
        return projected.reshape(batch_size, timesteps, -1)
