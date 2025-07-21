import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

sys.path.append(f"{os.path.dirname(__file__)}/src")  # noqa: E402

import models  # noqa: E402
import utils  # noqa: E402
import train.backprop  # noqa: E402


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
    model = models.LeNet5(n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    torchinfo.summary(model, input_size=(batch_size, 1, 32, 32))
    print("")

    # Train model using backpropagation
    history = train.backprop.backprop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
    )

    # Save trained model
    save_path = "results/backprop-model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining completed. Model saved to '{save_path}'")

    # Save training history
    dst = "results/backprop-history.csv"
    pd.DataFrame(history).to_csv(dst, index=False)

    # Evaluate model
    test_metrics = train.backprop.evaluate(model, test_loader, criterion, device)
    utils.save_metrics(test_metrics, "results/mnist-metrics.csv", "backprop")

    # Final summary
    print("\nBackpropagation – Training Summary")
    print("-" * 60)
    print(f"Train accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Val accuracy: {history['val_accuracies'][-1]:.2f}%")
    print(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Params: {n_params:,}")
    print(f"Epochs: {n_epochs}")
