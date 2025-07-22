from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFLinear(nn.Module):
    """Linear layer adapted for forward-forward training."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.linear(x))

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=1)


class FFConv2d(nn.Module):
    """Convolutional layer adapted for forward-forward training."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.conv(x))

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=(1, 2, 3))
