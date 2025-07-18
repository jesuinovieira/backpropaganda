from src import utils
import torch
from src.algorithms import backprop
from src.model import ConvNeuralNet
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report
import os


def evaluate_model(model, test_loader, class_names):
    """Evaluate the trained model on test data."""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())

    test_accuracy = 100.0 * correct / total

    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Error Rate: {100 - test_accuracy:.2f}%')
    print('\nDetailed Classification Report:')
    print(classification_report(true_labels, predictions, target_names=class_names))

    return test_accuracy, predictions, true_labels


if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5

    train_loader, val_loader, test_loader = utils.load_mnist_data(BATCH_SIZE)
    class_names = [str(i) for i in range(10)]

    # Initialize model, loss, and optimizer
    model = ConvNeuralNet(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    print(f"Model architecture:")
    print(model)

    # Train the model
    history = backprop.train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    print("\nTraining completed!")

    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the trained model
    torch.save(model.state_dict(), 'results/backprop.pth')
    print("Model saved as 'results/backprop.pth'")
 
    # Evaluate the model
    test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, class_names)

    # Final summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Final Training Accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_accuracies'][-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Epochs: {NUM_EPOCHS}")
    print("="*50)
