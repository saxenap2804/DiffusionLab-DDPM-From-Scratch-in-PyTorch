import torch


class DDPMScheduler:
    """
    Minimal DDPM noise scheduler implemented from scratch.
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps
        )

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )

    def add_noise(self, x0, noise, timesteps):
        """
        Add Gaussian noise to clean images using:

        x_t =
        sqrt(alpha_bar_t) * x_0
        +
        sqrt(1 - alpha_bar_t) * noise
        """

        alpha_bar_t = self.alpha_bars[
            timesteps
        ].to(x0.device)

        alpha_bar_t = alpha_bar_t.view(
            -1, 1, 1, 1
        )

        noisy_images = (
            torch.sqrt(alpha_bar_t) * x0
            +
            torch.sqrt(1 - alpha_bar_t) * noise
        )

        return noisy_images