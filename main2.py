import os

import torch
from src import models
from src import train
from src import utils
from src.train import ff


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
) -> tuple[float, list[int], list[int]]:
    """Evaluates a trained FF model on test data using goodness.

    For each input, the function tests every class label and selects the one with
    the highest goodness score.

    Args:
        model: Trained FF model.
        test_loader: DataLoader containing test dataset.
        num_classes: Number of output classes.

    Returns:
        Tuple containing:
            - Test accuracy as a percentage.
            - List of predicted labels.
            - List of true labels.
    """
    model.eval()
    correct = 0
    total = 0
    predictions: list[int] = []
    true_labels: list[int] = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            goodness_scores = torch.zeros((batch_size, num_classes), device=device)

            for label in range(num_classes):
                y_label = torch.full_like(y, label)
                x_overlay = ff.overlay_label(x, y_label, num_classes, is_positive=True)
                scores = model.goodness(x_overlay)  # shape: (batch_size,)
                goodness_scores[:, label] = scores

            predicted = goodness_scores.argmax(dim=1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(y.cpu().tolist())

    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}% | Error Rate: {100 - test_acc:.2f}%")

    return test_acc, predictions, true_labels


if __name__ == "__main__":
    # Config
    batch_size = 32
    n_classes = 10
    lr = 0.001
    n_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = utils.load_mnist_data(batch_size)
    class_names = [str(i) for i in range(n_classes)]

    # Initialize model, loss, optimizer
    model = models.FFLeNet5(n_classes=n_classes)
    optimizers = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in model.layers]

    total_params = sum(p.numel() for p in model.parameters())
    print(model, "\n")
    print(f"Total parameters: {total_params:,}")

    # Train model using forward-forward
    # - No criterion: FF does not use global loss
    # - No optimizer: FF uses per-layer optimizers
    history = train.forward_forward(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        n_epochs=n_epochs,
        device=device,
        n_classes=n_classes,
        threshold=2.0,
    )

    # Save trained model
    save_path = "results/forward-forward.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining completed. Model saved to '{save_path}'")

    # Evaluate model
    test_accuracy, predictions, true_labels = evaluate_model(
        model, test_loader, n_classes
    )

    # print("\nDetailed Classification Report:")
    # print(classification_report(true_labels, predictions, target_names=class_names))

    # Final summary
    print("\nForward-Forward – Training Summary")
    print("-" * 60)
    print(f"Train Accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Val Accuracy: {history['val_accuracies'][-1]:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Total Params: {total_params:,}")
    print(f"Epochs Trained: {n_epochs}")
