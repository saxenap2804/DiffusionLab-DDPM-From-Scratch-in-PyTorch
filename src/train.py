import os

import matplotlib.pyplot as plt
import torch
from torch import nn

from data import get_mnist_dataloader
from noise import corrupt
from unet import BasicUNet


def train_model(epochs=3, batch_size=128, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    dataloader = get_mnist_dataloader(batch_size=batch_size)

    model = BasicUNet().to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    losses = []

    for epoch in range(epochs):
        epoch_losses = []

        for images, _ in dataloader:
            images = images.to(device)

            noise_amount = torch.rand(
                images.shape[0],
                device=device
            )

            noisy_images = corrupt(
                images,
                noise_amount
            )

            predictions = model(noisy_images)

            loss = loss_fn(
                predictions,
                images
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())

        average_loss = sum(epoch_losses) / len(epoch_losses)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Average Loss: {average_loss:.6f}"
        )

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        model.state_dict(),
        "checkpoints/basic_unet.pth"
    )

    print("Model saved to checkpoints/basic_unet.pth")

    os.makedirs("outputs", exist_ok=True)

    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")

    plt.savefig(
        "outputs/training_loss.png",
        dpi=150
    )

    plt.close()

    print("Loss graph saved to outputs/training_loss.png")

    return model


if __name__ == "__main__":
    train_model()