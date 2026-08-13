import os

import matplotlib.pyplot as plt
import torch

from data import get_mnist_dataloader
from noise import corrupt
from unet import BasicUNet


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicUNet().to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/basic_unet.pth",
            map_location=device
        )
    )

    model.eval()

    dataloader = get_mnist_dataloader(batch_size=8)
    images, _ = next(iter(dataloader))

    images = images.to(device)

    noise_amount = torch.full(
        (images.shape[0],),
        0.5,
        device=device
    )

    noisy_images = corrupt(images, noise_amount)

    with torch.no_grad():
        predictions = model(noisy_images)

    images = images.cpu()
    noisy_images = noisy_images.cpu()
    predictions = predictions.cpu()

    fig, axes = plt.subplots(3, 8, figsize=(14, 6))

    for i in range(8):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")

        axes[1, i].imshow(noisy_images[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

        axes[2, i].imshow(
            predictions[i].squeeze().clip(0, 1),
            cmap="gray"
        )
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=12)
    axes[1, 0].set_ylabel("Noisy", fontsize=12)
    axes[2, 0].set_ylabel("Denoised", fontsize=12)

    plt.suptitle("U-Net Denoising Results")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    plt.savefig(
        "outputs/denoising_results.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":
    evaluate_model()