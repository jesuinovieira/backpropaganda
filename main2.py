import os
import sys

import torch

sys.path.append(f"{os.path.dirname(__file__)}/src")

import models  # noqa: E402
import utils  # noqa: E402
import train.ff  # noqa: E402


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

    n_params = sum(p.numel() for p in model.parameters())
    print(model, "\n")

    # Train model using forward-forward
    # - No criterion: FF does not use global loss
    # - No optimizer: FF uses per-layer optimizers
    history = train.ff.forward_forward(
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
    test_acc = train.ff.evaluate(model, test_loader, n_classes, device)

    # print("\nDetailed Classification Report:")
    # print(classification_report(true_labels, predictions, target_names=class_names))

    # Final summary
    print("\nForward-Forward – Training Summary")
    print("-" * 60)
    print(f"Train accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Val accuracy: {history['val_accuracies'][-1]:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Params: {n_params:,}")
    print(f"Epochs: {n_epochs}")
