from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import AutoModel


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int = 768, t_out: int = 16, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(t_out, dim) * 0.02)
        self.register_buffer("pe", sinusoidal_encoding(t_out, dim), persistent=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = (self.query + self.pe).unsqueeze(0).expand(x.size(0), -1, -1)
        output, _ = self.attn(query, x, x, need_weights=False)
        return output


class AudioBranch(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        projection_dim: int = 256,
        t_out: int = 16,
        num_heads: int = 4,
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
        self.pool = TemporalAttentionPool(hidden_size, t_out, num_heads)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.backbone(audio_waveform)
            hidden = outputs.last_hidden_state
        pooled = self.pool(hidden)
        return self.projection(pooled)
