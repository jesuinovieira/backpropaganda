import contextlib
import io
import torchvision
import torch


def download_mnist(train_prop=0.8, keep_prop=0.5, path="./data/"):
    # Create path if it does not exist
    if not torch.os.path.exists(path):
        torch.os.makedirs(path)

    # If the MNIST datasets already exist, load them from files
    if torch.os.path.exists(path + "MNIST"):
        print("MNIST datasets already exist, loading from files")

        # Load the datasets from the existing files
        train = torch.load(path + "train.pt", weights_only=False)
        val = torch.load(path + "val.pt", weights_only=False)
        test = torch.load(path + "test.pt", weights_only=False)

        return train, val, test

    # Otherwise, download the MNIST datasets
    print("Downloading MNIST datasets")
    valid_prop = 1 - train_prop
    discard_prop = 1 - keep_prop

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    with contextlib.redirect_stdout(io.StringIO()):  # to suppress output
        train = torchvision.datasets.MNIST(
            root=path, train=True, download=True, transform=transform
        )
        test = torchvision.datasets.MNIST(
            root=path, train=False, download=True, transform=transform
        )

    train, val, _ = torch.utils.data.random_split(
        train, [train_prop * keep_prop, valid_prop * keep_prop, discard_prop]
    )
    test, _ = torch.utils.data.random_split(test, [keep_prop, discard_prop])

    print("Number of examples retained:")
    print(f"  {len(train)} (training)")
    print(f"  {len(val)} (validation)")
    print(f"  {len(test)} (test)")

    # Save the datasets to the specified path
    torch.save(train, path + "train.pt")
    torch.save(val, path + "val.pt")
    torch.save(test, path + "test.pt")

    return train, val, test
