import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def backprop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    n_epochs: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Trains a model using backpropagation algorithm.

    "Learning representations by back-propagating errors"
    Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986)
    https://www.nature.com/articles/323533a0

    Args:
        model: PyTorch neural network model.
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        optimizer: Optimization algorithm (e.g., torch.optim.SGD).
        n_epochs: Number of training epochs.
        device: torch.device ('cuda' or 'cpu') for computation.

    Returns:
        Dictionary with history of training and validation metrics.
    """
    model.to(device)

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    print(f"Starting backpropagation training for {n_epochs} epochs")
    print("-" * 60)

    for epoch in range(n_epochs):
        # Training phase: forward → loss → backward → update
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation phase: no gradient tracking
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch + 1}/{n_epochs}] "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%"
        )

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Runs one training epoch using backpropagation.

    This function performs the full training loop for one epoch:

        - Forward pass to compute predictions and loss.
        - Backward pass to compute gradients.
        - Optimizer step to update model weights.

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader for training samples.
        criterion: Loss function.
        optimizer: Optimizer (e.g., SGD, Adam).
        device: Device to run training on.

    Returns:
        Tuple of (average loss, accuracy for the epoch).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Forward pass: compute predicted outputs
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass: compute gradients of loss w.r.t. model params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss: float = total_loss / len(dataloader)
    accuracy: float = 100.0 * correct / total_samples
    return avg_loss, accuracy


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Evaluates the model on validation data.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader with validation samples.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == y).sum().item()
            total_samples += y.size(0)

    avg_loss: float = total_loss / len(dataloader)
    accuracy: float = 100.0 * correct / total_samples
    return avg_loss, accuracy
