import os

import torch
import torch.nn as nn
import torch.optim as optim
from src import model
from src import train
from src import utils


def evaluate_model(
    model: nn.Module, test_loader: torch.utils.data.DataLoader
) -> tuple[float, list[int], list[int]]:
    """Evaluates a trained model on test data.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader containing test dataset.

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

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(targets.cpu().tolist())

    test_acc = 100.0 * correct / total

    print(f"Test Accuracy: {test_acc:.2f}% | Error Rate: {100 - test_acc:.2f}%")
    return test_acc, predictions, true_labels


if __name__ == "__main__":
    # Config
    batch_size = 32
    n_classes = 10
    learning_rate = 0.001
    n_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = utils.load_mnist_data(batch_size)
    class_names = [str(i) for i in range(n_classes)]

    # Initialize model, loss, optimizer
    model = model.LeNet5(n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in model.parameters())
    print(model, "\n")
    print(f"Total parameters: {total_params:,}")

    # Train model using backpropagation
    history = train.backprop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=n_epochs,
        device=device,
    )

    # Save trained model
    save_path = "results/backprop.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining completed. Model saved to '{save_path}'")

    # Evaluate model
    test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)

    # print("\nDetailed Classification Report:")
    # print(classification_report(true_labels, predictions, target_names=class_names))

    # Final summary
    print("\nBackpropagation – Training Summary")
    print("-" * 60)
    print(f"Train Accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Val Accuracy: {history['val_accuracies'][-1]:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Total Params: {total_params:,}")
    print(f"Epochs Trained: {n_epochs}")
