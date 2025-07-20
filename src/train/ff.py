import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def forward_forward(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizers: list[Optimizer],
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

    print(f"Starting forward-forward training for {n_epochs} epochs")
    print("-" * 60)

    for epoch in range(n_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizers, device, n_classes, threshold
        )

        # Validation phase: no gradient tracking
        val_acc = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch + 1}/{n_epochs}] "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%  "
            f"Val Acc: {val_acc:.2f}%"
        )

    return {
        "train_accuracies": train_accuracies,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizers: list[Optimizer],
    device: torch.device,
    n_classes: int,
    threshold: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

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

        # FIXME: very slow to store accuracy for each batch
        # with torch.no_grad():
        #     goodness_scores = torch.zeros((x.size(0), n_classes), device=device)
        #     for label in range(n_classes):
        #         y_label = torch.full_like(y, label)
        #         x_overlay = overlay_label(x, y_label, n_classes, is_positive=True)
        #         g = model.goodness(x_overlay)
        #         goodness_scores[:, label] = g

        #     preds = goodness_scores.argmax(dim=1)
        #     correct += (preds == y).sum().item()
        #     total_samples += y.size(0)

    # FIXME: train accuracy is not computed correctly?
    avg_loss: float = total_loss / len(dataloader)
    accuracy: float = 100.0 * correct / total_samples
    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            scores = []

            for class_id in range(model.fc2.out_features):
                fake_y = torch.full_like(y, class_id)
                x_overlay = overlay_label(x, fake_y, model.fc2.out_features)
                g = model.goodness(x_overlay)
                scores.append(g)

            logits = torch.stack(scores, dim=1)  # (B, C)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total_samples += y.size(0)

    accuracy: float = 100.0 * correct / total_samples
    return accuracy


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
