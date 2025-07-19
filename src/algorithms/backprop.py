import torch


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train the CNN model and track performance."""
    # Initialize tracking
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct_predictions / total_samples

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print progress every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }
