import os

import matplotlib.pyplot as plt
import torch

from data import get_mnist_dataloader
from noise import corrupt


def visualize_corruption():
    """Visualize an MNIST image at increasing noise levels."""

    # Load MNIST
    dataloader = get_mnist_dataloader(batch_size=1)
    images, _ = next(iter(dataloader))

    image = images[0:1]

    # Noise levels from clean image to pure noise
    noise_levels = torch.tensor([
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ])

    # Repeat the same image for every noise level
    repeated_images = image.repeat(len(noise_levels), 1, 1, 1)

    # Apply corruption
    noisy_images = corrupt(repeated_images, noise_levels)

    # Create visualization
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(15, 3))

    for i, amount in enumerate(noise_levels):
        axes[i].imshow(
            noisy_images[i].squeeze().numpy(),
            cmap="gray"
        )

        axes[i].set_title(f"{int(amount.item() * 100)}% Noise")
        axes[i].axis("off")

    plt.suptitle("Forward Diffusion: Increasing Image Corruption")
    plt.tight_layout()

    # Create output directory if necessary
    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/noise_progression.png"

    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    visualize_corruption()