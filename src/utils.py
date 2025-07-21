import os

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def load_mnist_data(
    batch_size: int, path: str = "./data"
) -> tuple[DataLoader, DataLoader, DataLoader]:
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


def save_metrics(metrics: dict, path: str, model_name: str) -> None:
    """Save or update test metrics for a specific model in a CSV file.

    Args:
        metrics: Dictionary of metric values (e.g., accuracy, loss, etc.).
        path: Path to the CSV file.
        model_name: Identifier for the model (e.g., 'backprop', 'ff').
    """
    metrics = {"model": model_name, **metrics}
    new_row = pd.DataFrame([metrics])

    # If the file does not exist, create it with the new row
    if not os.path.exists(path):
        new_row.to_csv(path, index=False)
        return

    # If the file exists, read it and append the new row
    df = pd.read_csv(path)

    # Ensure all columns exist in both dataframes
    for col in new_row.columns:
        if col not in df.columns:
            df[col] = pd.NA
    for col in df.columns:
        if col not in new_row.columns:
            new_row[col] = pd.NA

    # Reorder new_row columns to match df
    new_row = new_row[df.columns]

    # Remove old row with the same model name
    df = df[df["model"] != model_name]

    # Concatenate and save
    df = pd.concat([df, new_row.astype(df.dtypes)], ignore_index=True)
    df.to_csv(path, index=False)
