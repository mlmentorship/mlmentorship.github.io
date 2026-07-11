import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Add a trainable low-rank update to a frozen Linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("return frozen base output plus the scaled low-rank update")
