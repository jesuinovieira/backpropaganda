import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def load_mnist_data(batch_size, path="./data"):
    """Load and preprocess the MNIST dataset."""
    # Define transformations
    mnist_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Resize to 32x32 for LeNet compatibility
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # MNIST normalization
        ]
    )

    # Load datasets
    train_val_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=mnist_transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=mnist_transform, download=True
    )

    # Split training into train/validation (90%/10%)
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
