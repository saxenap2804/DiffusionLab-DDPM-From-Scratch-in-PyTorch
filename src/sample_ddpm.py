import os

import matplotlib.pyplot as plt
import torch
import torchvision

from ddpm_scheduler import DDPMScheduler
from unet import TimeConditionedUNet


def sample_ddpm(
    n_samples=64,
    num_timesteps=1000
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = TimeConditionedUNet().to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/ddpm_unet.pth",
            map_location=device
        )
    )

    model.eval()

    scheduler = DDPMScheduler(
        num_timesteps=num_timesteps
    )

    betas = scheduler.betas.to(device)
    alphas = scheduler.alphas.to(device)
    alpha_bars = scheduler.alpha_bars.to(device)

    # Start from pure Gaussian noise
    x = torch.randn(
        n_samples,
        1,
        28,
        28,
        device=device
    )

    with torch.no_grad():

        for t in reversed(range(num_timesteps)):

            timesteps = torch.full(
                (n_samples,),
                t,
                device=device,
                dtype=torch.long
            )

            # Predict epsilon
            predicted_noise = model(
                x,
                timesteps
            )

            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            beta_t = betas[t]

            # DDPM reverse-process mean
            model_mean = (
                1 / torch.sqrt(alpha_t)
            ) * (
                x
                -
                (
                    beta_t /
                    torch.sqrt(1 - alpha_bar_t)
                )
                * predicted_noise
            )

            # Add stochastic noise except at final step
            if t > 0:
                noise = torch.randn_like(x)

                x = (
                    model_mean
                    +
                    torch.sqrt(beta_t) * noise
                )

            else:
                x = model_mean

            # Optional progress feedback
            if t % 100 == 0:
                print(
                    f"Sampling timestep {t}"
                )

    # Convert [-1, 1] back to [0, 1]
    samples = (
        (x.clamp(-1, 1) + 1) / 2
    ).cpu()

    grid = torchvision.utils.make_grid(
        samples,
        nrow=8
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 10)
    )

    plt.imshow(
        grid[0],
        cmap="gray"
    )

    plt.axis("off")

    plt.title(
        "DDPM Generated MNIST Samples"
    )

    plt.savefig(
        "outputs/ddpm_generated_samples.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    sample_ddpm()