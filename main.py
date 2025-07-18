from src import utils
from src.backprop import MLPBackprop
from src.backprop import BasicOptimizer
import torch
import numpy as np
from src.train import train_model


if __name__ == "__main__":
    train, val, test = utils.download_mnist()

    # Model
    NUM_HIDDEN = 100
    ACTIVATION = "ReLU"  # output constrained between 0 and 1
    BIAS = True

    NUM_INPUTS = np.prod(train.dataset.data[0].shape)  # size of an MNIST image
    NUM_OUTPUTS = 10  # number of MNIST classes

    MLP = MLPBackprop(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        num_hidden=NUM_HIDDEN,
        activation_type=ACTIVATION,
        bias=BIAS,
    )

    # Dataloaders
    BATCH_SIZE = 32

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        val, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False
    )

    LR = 0.01
    weight_decay = 0.0
    backprop_optimizer = BasicOptimizer(
        MLP.parameters(), lr=LR, weight_decay=weight_decay
    )

    NUM_EPOCHS = 5

    MLP_results_dict = train_model(
        MLP,
        train_loader,
        valid_loader,
        optimizer=backprop_optimizer,
        num_epochs=NUM_EPOCHS,
    )
