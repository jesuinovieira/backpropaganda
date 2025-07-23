from typing import Callable

import torch
import torch.nn as nn


class FFLinear(nn.Linear):
    """Linear layer adapted for forward-forward training."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x if self.act_fn is None else self.act_fn(x)

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=1)


class FFConv2d(nn.Conv2d):
    """Convolutional layer adapted for forward-forward training."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, int],
        act_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs
    ):
        super().__init__(in_features, out_features, kernel_size, **kwargs)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x if self.act_fn is None else self.act_fn(x)

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=(1, 2, 3))
