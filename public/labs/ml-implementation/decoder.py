import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return causal self-attention output with shape [batch, time, d_model]."""
        raise NotImplementedError("implement projections, heads, mask, stable softmax, and output")


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden
