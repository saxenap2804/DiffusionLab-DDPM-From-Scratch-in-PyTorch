import os

import matplotlib.pyplot as plt
import torch

from ddpm_scheduler import DDPMScheduler


def visualize_schedule():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    alpha_bar = scheduler.alpha_bars

    signal = torch.sqrt(alpha_bar)
    noise = torch.sqrt(1 - alpha_bar)

    plt.figure(figsize=(10, 6))

    plt.plot(
        signal.numpy(),
        label="sqrt(alpha_bar_t)"
    )

    plt.plot(
        noise.numpy(),
        label="sqrt(1 - alpha_bar_t)"
    )

    plt.xlabel("Timestep")
    plt.ylabel("Contribution")
    plt.title("DDPM Signal and Noise Schedule")

    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.savefig(
        "outputs/ddpm_noise_schedule.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":
    visualize_schedule()