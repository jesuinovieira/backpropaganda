from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers


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

    def __init__(self, n_classes: int = 10, latent_dim: int = 84, act_fn: str = "relu"):
        """Initializes the LeNet-5 model.

        Args:
            n_classes: Number of output classes.
            latent_dim: Number of units in the penultimate fully connected layer.
            act_fn: Name of the activation function to use.
        """
        super().__init__()
        self.act_fn = self._set_activation(act_fn)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # C1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  # C5

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=120, out_features=latent_dim)  # F6
        self.fc2 = nn.Linear(in_features=latent_dim, out_features=n_classes)  # Output

        _initialize_weights(self, act_fn_name=act_fn)

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
        self.conv1 = layers.FFConv2d(1 + n_classes, 6, kernel_size=5, act_fn=F.relu)  # C1 # fmt: skip # noqa: E501
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        self.conv2 = layers.FFConv2d(6, 16, kernel_size=5, act_fn=F.relu)  # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        self.conv3 = layers.FFConv2d(16, 120, kernel_size=5, act_fn=F.relu)  # C5

        # Fully connected layers
        self.fc1 = layers.FFLinear(in_features=120, out_features=latent_dim, act_fn=F.relu)  # F6 # fmt: skip # noqa: E501
        self.fc2 = layers.FFLinear(in_features=latent_dim, out_features=n_classes, act_fn=None)  # Output # fmt: skip # noqa: E501

        self.layers = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]
        _initialize_weights(self, act_fn_name="relu")

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


def PCLeNet5(n_classes: int = 10, latent_dim: int = 84) -> nn.Sequential:
    # TODO: implement PCLeNet5 class
    model = nn.Sequential(
        nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)),
        nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2)),
        nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU()),
        nn.Sequential(nn.Linear(120, latent_dim), nn.ReLU()),
        nn.Sequential(nn.Linear(latent_dim, n_classes)),
    )

    _initialize_weights(model, act_fn_name="relu")
    return model


def _initialize_weights(model, act_fn_name: str) -> None:
    for m in model.modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue

        if act_fn_name == "relu":
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif act_fn_name in ["tanh", "sigmoid"]:
            nn.init.xavier_normal_(m.weight)
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
