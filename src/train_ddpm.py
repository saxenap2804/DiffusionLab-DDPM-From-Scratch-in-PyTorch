import os
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn

from data import get_mnist_dataloader
from ddpm_scheduler import DDPMScheduler
from unet import TimeConditionedUNet


def train_ddpm(
    epochs=3,
    batch_size=128,
    learning_rate=1e-3,
    num_timesteps=1000
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    dataloader = get_mnist_dataloader(
        batch_size=batch_size
    )

    model = TimeConditionedUNet().to(device)

    scheduler = DDPMScheduler(
        num_timesteps=num_timesteps
    )

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

            # DDPM commonly uses image range [-1, 1]
            images = images * 2.0 - 1.0

            batch_size_current = images.shape[0]

            # Random timestep for each image
            timesteps = torch.randint(
                0,
                num_timesteps,
                (batch_size_current,),
                device=device
            )

            # Gaussian noise
            noise = torch.randn_like(images)

            # Create x_t
            noisy_images = scheduler.add_noise(
                images,
                noise,
                timesteps
            )

            # Predict the noise epsilon
            predicted_noise = model(
                noisy_images,
                timesteps
            )

            # Compare predicted noise with actual noise
            loss = loss_fn(
                predicted_noise,
                noise
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())

        average_loss = (
            sum(epoch_losses) /
            len(epoch_losses)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Average Loss: {average_loss:.6f}"
        )

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "checkpoints/ddpm_unet.pth"
    )

    print(
        "Model saved to checkpoints/ddpm_unet.pth"
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.figure()

    plt.plot(losses)

    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.title("DDPM Noise Prediction Training Loss")

    plt.savefig(
        "outputs/ddpm_training_loss.png",
        dpi=150
    )

    plt.close()

    print(
        "Loss graph saved to outputs/ddpm_training_loss.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DDPM model on MNIST."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate"
    )

    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps"
    )

    args = parser.parse_args()

    train_ddpm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_timesteps=args.num_timesteps
    )