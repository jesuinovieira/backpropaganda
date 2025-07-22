import time

import torch
import torch.nn.functional as F

from . import T2PC


def predictive_coding(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    n_classes: int,
    n_epochs,
    device: torch.device,
    INFERENCE_LEARNING_RATE: float,
    N_INFERENCE_STEPS: int,
):
    """Train the CNN model using Predictive Coding (Strict)."""
    model.to(device)

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "epoch_times": [],
    }

    print("Starting predictive coding training")
    print("-" * 60)

    for epoch in range(n_epochs):
        t1 = time.time()
        model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            y_onehot = F.one_hot(y, num_classes=n_classes).float()
            vhat, loss, _, _, _ = T2PC.PCInfer(
                model,
                criterion,
                x,
                y_onehot,
                "Strict",
                eta=INFERENCE_LEARNING_RATE,
                n=N_INFERENCE_STEPS,
            )

            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            _, predicted_labels = torch.max(vhat[-1].data, 1)
            total_samples += y.size(0)
            correct_predictions += (predicted_labels == y).sum().item()

        history["train_losses"].append(running_loss / len(train_loader))
        history["train_accuracies"].append(correct_predictions / total_samples)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                y_onehot = F.one_hot(y, num_classes=n_classes).float()

                loss = criterion(outputs, y_onehot)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        history["val_losses"].append(val_loss / len(val_loader))
        history["val_accuracies"].append(val_correct / val_total)
        history["epoch_times"].append(time.time() - t1)

        print(
            f"Epoch: [{epoch + 1}/{n_epochs}]  "
            f"Train Loss: {history['train_losses'][-1]:.4f},  "
            f"Train Acc: {history['train_accuracies'][-1]:6.2f},  "
            f"Val Loss: {history['val_losses'][-1]:.4f},  "
            f"Val Acc: {history['val_accuracies'][-1]:6.2f},  "
            f"Time: { history['epoch_times'][-1]:5.2f}s"
        )

    return history
