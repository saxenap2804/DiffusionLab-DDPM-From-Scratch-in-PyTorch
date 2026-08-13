import torch

from src.ddpm_scheduler import DDPMScheduler


def test_scheduler_length():
    scheduler = DDPMScheduler(
        num_timesteps=1000
    )

    assert len(scheduler.betas) == 1000
    assert len(scheduler.alphas) == 1000
    assert len(scheduler.alpha_bars) == 1000


def test_add_noise_preserves_shape():
    scheduler = DDPMScheduler()

    x = torch.rand(
        8,
        1,
        28,
        28
    )

    noise = torch.randn_like(x)

    timesteps = torch.randint(
        0,
        scheduler.num_timesteps,
        (8,)
    )

    noisy_x = scheduler.add_noise(
        x,
        noise,
        timesteps
    )

    assert noisy_x.shape == x.shape


def test_alpha_bar_decreases():
    scheduler = DDPMScheduler()

    assert scheduler.alpha_bars[0] > scheduler.alpha_bars[-1]