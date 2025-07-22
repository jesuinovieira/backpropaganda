import time
from typing import Sequence

import sklearn.metrics
import torch
import torch.nn.functional as F
from torch import nn


def forward_forward(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    device: torch.device,
    n_classes: int,
    n_epochs: int,
    threshold: float = 2.0,
) -> dict[str, list[float]]:
    """Trains a model using forward-forward algorithm.

    "The Forward-Forward Algorithm: Some Preliminary Investigations"
    Geoffrey Hinton (2022)
    https://arxiv.org/abs/2212.13345

    Core idea:

    - Each layer is trained independently using a local loss based on "goodness"
    - No global error signal or backpropagation
    - Labels are not predicted via cross-entropy, but injected directly into inputs

    Args:
        model: An FF-compatible model (e.g., FFLeNet5).
        train_loader: Training DataLoader.
        device: Target device ('cuda' or 'cpu').
        n_classes: Number of classes (used for label overlay).
        threshold: Margin for FF loss.
        lr: Learning rate per layer.
        n_epochs: Number of passes per layer.
    """
    model.to(device)

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []
    epoch_times: list[float] = []

    print("Starting forward-forward training")
    print("-" * 60)

    for epoch in range(n_epochs):
        t1 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizers, device, n_classes, threshold
        )
        epoch_times.append(time.time() - t1)

        # Validation phase: no gradient tracking
        val_metrics = evaluate(model, val_loader, n_classes, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_metrics["accuracy"])

        # if (epoch + 1) % 5 == 0 or epoch == 0:
        tmp_acc = f"{train_acc:6.2f}%" if train_acc else "N/A"
        print(
            f"Epoch: [{epoch + 1:02d}/{n_epochs}]  "
            f"Train Loss: {train_loss:.4f},  "
            f"Train Acc: {tmp_acc},  "
            f"Val Acc: {val_metrics['accuracy']:6.2f},  "
            f"Time: {epoch_times[-1]:5.2f}s"
        )

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": [None] * n_epochs,  # No global loss in FF
        "val_accuracies": val_accuracies,
        "epoch_times": epoch_times,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    device: torch.device,
    n_classes: int,
    threshold: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_pred = []
    y_true = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Prepare positive and negative samples
        x_pos = overlay_label(x, y, n_classes, is_positive=True)
        x_neg = overlay_label(x, y, n_classes, is_positive=False)

        for i, (layer, optimizer) in enumerate(zip(model.layers, optimizers)):
            # Forward-forward pass through this layer
            x_pos = layer(x_pos)
            x_neg = layer(x_neg)

            # Compute goodness
            g_pos = layer.goodness(x_pos)
            g_neg = layer.goodness(x_neg)

            # Loss encourages g_pos > g_neg by margin `threshold`
            loss = F.relu(threshold - g_pos + g_neg).mean()

            # Forward-forward the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Prepare for next layer
            x_pos = model.post_layer_transform(x_pos, i).detach().requires_grad_()
            x_neg = model.post_layer_transform(x_neg, i).detach().requires_grad_()

        # FIXME: very slow to store accuracy for each batch, skipping
        with torch.no_grad():
            batch_size = x.size(0)
            g_scores = torch.zeros((batch_size, n_classes), device=device)

            for label in range(n_classes):
                y_label = torch.full_like(y, label)
                x_overlay = overlay_label(x, y_label, n_classes, is_positive=True)
                g_scores[:, label] = model.goodness(x_overlay)

            preds = g_scores.argmax(dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    avg_loss: float = total_loss / len(dataloader)
    accuracy: float = sklearn.metrics.accuracy_score(y_true, y_pred)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_classes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            batch_size = x.size(0)
            g_scores = torch.zeros((batch_size, n_classes), device=device)

            for label in range(n_classes):
                y_label = torch.full_like(y, label)
                x_overlay = overlay_label(x, y_label, n_classes, is_positive=True)
                g_scores[:, label] = model.goodness(x_overlay)

            preds = g_scores.argmax(dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    # confmat = sklearn.metrics.confusion_matrix(y_true, y_pred)

    return {
        "loss": None,  # No global loss in FF
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        # "confusion_matrix": confmat.tolist(),
    }


def overlay_label(
    x: torch.Tensor, y: torch.Tensor, num_classes: int, is_positive: bool = True
) -> torch.Tensor:
    """Overlays the label as additional channels in the input image.

    In the forward-forward algorithm, labels are not used in a traditional loss.
    Instead, we inject the label directly into the input by appending a one-hot
    encoding as extra channels (one channel per class).

    This allows each label to be interpreted as a different "hypothesis". During
    evaluation, we try each hypothesis and pick the one with highest goodness.

    For example:
        - Original shape: (B, 1, 28, 28)
        - After overlay:  (B, 1 + num_classes, 28, 28)

    Args:
        x: Input images of shape (B, C, H, W), typically (B, 1, 28, 28).
        y: Ground truth labels of shape (B,).
        num_classes: Total number of output classes (e.g., 10 for MNIST).
        is_positive: If True, use correct label; otherwise, use incorrect label.

    Returns:
        Tensor of shape (B, 1 + num_classes, H, W), combining image and label.
    """
    B, _, H, W = x.shape  # B: batch size, C: channels, H: height, W: width

    if is_positive:
        labels = y
    else:
        labels = get_random_wrong_labels(y, num_classes)

    one_hot = F.one_hot(labels, num_classes).float().to(x.device)
    label_map = one_hot.view(B, num_classes, 1, 1).expand(-1, -1, H, W)

    # Concatenate label representation to the image
    return torch.cat([x, label_map], dim=1)


def get_random_wrong_labels(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Generates wrong labels different from ground truth."""
    wrong = torch.randint_like(y, high=n_classes)
    mask = wrong == y
    while mask.any():
        wrong[mask] = torch.randint_like(wrong[mask], high=n_classes)
        mask = wrong == y
    return wrong
