import os

import matplotlib.pyplot as plt
import torch
import torchvision

from unet import BasicUNet


def generate_samples(n_samples=64, n_steps=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicUNet().to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/basic_unet.pth",
            map_location=device
        )
    )

    model.eval()

    # Start from random noise
    x = torch.rand(
        n_samples,
        1,
        28,
        28,
        device=device
    )

    with torch.no_grad():
        for i in range(n_steps):
            prediction = model(x)

            mix_factor = 1 / (n_steps - i)

            x = (
                x * (1 - mix_factor)
                + prediction * mix_factor
            )

    samples = x.detach().cpu().clip(0, 1)

    grid = torchvision.utils.make_grid(
        samples,
        nrow=8
    )

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid[0], cmap="gray")
    plt.axis("off")
    plt.title("Generated Samples From Random Noise")

    plt.savefig(
        "outputs/generated_samples.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    generate_samples()