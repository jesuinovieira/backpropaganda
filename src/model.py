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

    def _set_activation(self, name: str):
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
