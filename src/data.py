from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(batch_size=128):
    """Download MNIST and create the training DataLoader."""

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


if __name__ == "__main__":
    train_loader = get_mnist_dataloader()

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Minimum pixel value:", images.min().item())
    print("Maximum pixel value:", images.max().item())