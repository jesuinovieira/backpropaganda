from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5 style convolutional neural network for digit classification.

    "Gradient-based learning applied to document recognition."
    LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
    http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

    Architecture:
        - C1: Conv2d(1, 6, kernel=5x5) → Activation → AvgPool(2x2)
        - C3: Conv2d(6, 16, kernel=5x5) → Activation → AvgPool(2x2)
        - C5: Conv2d(16, 120, kernel=5x5) → Activation
        - F6: Linear(120, latent_dim) → Activation
        - Output: Linear(latent_dim, n_classes)
    """

    def __init__(
        self, n_classes: int = 10, latent_dim: int = 84, activation: str = "relu"
    ):
        """Initializes the LeNet-5 model.

        Args:
            n_classes: Number of output classes.
            latent_dim: Number of units in the penultimate fully connected layer.
            activation: Name of the activation function to use.
        """
        super().__init__()
        self.act_fn = self._set_activation(activation)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # C1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  # C5

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=120, out_features=latent_dim)  # F6
        self.fc2 = nn.Linear(in_features=latent_dim, out_features=n_classes)  # Output

    def _set_activation(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Returns the activation function specified by name."""
        name = name.lower()

        if name == "relu":
            return F.relu
        if name == "tanh":
            return torch.tanh
        if name == "sigmoid":
            return torch.sigmoid

        raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the LeNet-5 model.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Logits tensor of shape (batch_size, n_classes).
        """
        x = self.pool1(self.act_fn(self.conv1(x)))  # C1 → S2
        x = self.pool2(self.act_fn(self.conv2(x)))  # C3 → S4
        x = self.act_fn(self.conv3(x))  # C5

        x = torch.flatten(x, 1)  # Flatten (N, 120)

        x = self.act_fn(self.fc1(x))  # F6
        x = self.fc2(x)  # Output layer

        return x


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


class FFLeNet5(nn.Module):
    """LeNet-5 architecture adapted for forward-forward training.

    Each layer is trained independently using local losses based on 'goodness'.

    Differences from standard LeNet5 with backprop:

    - The first conv layer receives class information injected into the input
    (1 + n_classes channels)
    - No softmax or cross-entropy: predictions come from the highest total goodness
    across class-conditional inputs
    """

    def __init__(self, n_classes: int = 10, latent_dim: int = 84):
        super().__init__()

        # Convolutional layers
        # self.conv1 = FFConv2d(1, 6, kernel_size=5, act_fn=F.relu)  # C1
        self.conv1 = FFConv2d(1 + n_classes, 6, kernel_size=5, act_fn=F.relu)  # C1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        self.conv2 = FFConv2d(6, 16, kernel_size=5, act_fn=F.relu)  # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        self.conv3 = FFConv2d(16, 120, kernel_size=5, act_fn=F.relu)  # C5

        # Fully connected layers
        self.fc1 = FFLinear(in_features=120, out_features=latent_dim, act_fn=F.relu)  # F6 # fmt: skip # noqa: E501
        self.fc2 = FFLinear(in_features=latent_dim, out_features=n_classes, act_fn=F.relu)  # Output # fmt: skip # noqa: E501

        self.layers = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]

    def _normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalizes each sample to unit L2 norm.

        Motivation:

        - Encourages distributed representations
        - Prevents large activations from dominating learning
        - Makes comparison of goodness values more meaningful
        """
        return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

    def post_layer_transform(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        x = self._normalize(x)

        if idx == 0:
            return self.pool1(x)
        if idx == 1:
            return self.pool2(x)
        if idx == 2:
            return torch.flatten(x, 1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass used for inference (not training).

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Activation tensor of shape (batch_size, n_classes).
        """
        x = self.pool1(self.conv1(x))  # C1 → S2
        x = self.pool2(self.conv2(x))  # C3 → S4
        x = self.conv3(x)  # C5

        x = torch.flatten(x, 1)  # Flatten (N, 120)

        x = self.fc1(x)  # F6
        x = self.fc2(x)  # Output layer

        return x

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the total goodness across all layers."""
        x1 = self.conv1(x)
        g1 = self.conv1.goodness(x1)
        x = self.pool1(self._normalize(x1))

        x2 = self.conv2(x)
        g2 = self.conv2.goodness(x2)
        x = self.pool2(self._normalize(x2))

        x = self.conv3(x)  # C5
        g3 = self.conv3.goodness(x)

        x = torch.flatten(self._normalize(x), 1)  # Flatten (N, 120)

        x = self.fc1(x)  # F6
        g4 = self.fc1.goodness(x)
        x = self._normalize(x)

        x = self.fc2(x)  # Output layer
        g5 = self.fc2.goodness(x)

        return g1 + g2 + g3 + g4 + g5
