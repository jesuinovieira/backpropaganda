import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration parameters
LATENT_DIM = 84
NUM_CLASSES = 10


class ConvNeuralNet(nn.Module):
    """Convolutional Neural Network for MNIST digit classification.

    Architecture:
    - Conv2d(1, 6, 5) -> ReLU -> MaxPool2d(2, 2)
    - Conv2d(6, 16, 5) -> ReLU -> MaxPool2d(2, 2)
    - Linear(16*5*5, 120) -> ReLU
    - Linear(120, 84) -> ReLU
    - Linear(84, 10) -> Output (logits)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(ConvNeuralNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)      # 1 input channel, 6 output, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)     # 6 input channels, 16 output, 5x5 kernel

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # Flattened conv output -> 120 features
        self.fc2 = nn.Linear(120, LATENT_DIM)   # 120 -> 84 features (latent)
        self.fc3 = nn.Linear(LATENT_DIM, num_classes)  # 84 -> 10 classes

    def forward(self, x):
        # First conv block: Conv -> ReLU -> MaxPool
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # Second conv block: Conv -> ReLU -> MaxPool
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output logits (no activation)

        return x
