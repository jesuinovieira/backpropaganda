import time

import sklearn.metrics
import torch
from torch import nn

from . import T2PC


def predictive_coding(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    n_epochs,
    device: torch.device,
    inference_lr: float,
    n_inference_steps: int,
):
    model.to(device)

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    epoch_times: list[float] = []

    print("Starting predictive coding training")
    print("-" * 60)

    for epoch in range(n_epochs):
        # Training phase
        t1 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            inference_lr,
            n_inference_steps,
        )
        epoch_times.append(time.time() - t1)

        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_metrics["loss"])
        val_accuracies.append(val_metrics["accuracy"])

        print(
            f"Epoch: [{epoch + 1:02d}/{n_epochs:02d}]  "
            f"Train Loss: {train_loss:.4f},  "
            f"Train Acc: {train_acc:6.2f},  "
            f"Val Loss: {val_metrics['loss']:.4f},  "
            f"Val Acc: {val_metrics['accuracy']:6.2f},  "
            f"Time: {epoch_times[-1]:5.2f}s"
        )

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "epoch_times": epoch_times,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    inference_lr: float,
    n_inference_steps: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_pred = []
    y_true = []

    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        vhat, loss, _, _, _ = T2PC.PCInfer(
            model,
            criterion,
            x,
            y,
            "Strict",
            eta=inference_lr,
            n=n_inference_steps,
        )

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        _, preds = torch.max(vhat[-1].data, 1)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(y.cpu().numpy())

    avg_loss: float = total_loss / len(dataloader)
    accuracy: float = sklearn.metrics.accuracy_score(y_true, y_pred)
    return avg_loss, accuracy


def evaluate(
    model,
    dataloader,
    criterion,
    device: torch.device,
):
    from . import backprop

    return backprop.evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )
