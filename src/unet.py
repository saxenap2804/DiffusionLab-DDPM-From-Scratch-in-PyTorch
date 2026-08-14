import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Create sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2

        scale = math.log(10000) / (half_dim - 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device
            ) * -scale
        )

        embeddings = timesteps[:, None].float() * frequencies[None, :]

        embeddings = torch.cat(
            [
                torch.sin(embeddings),
                torch.cos(embeddings)
            ],
            dim=1
        )

        return embeddings


class TimeConditionedUNet(nn.Module):
    """Minimal U-Net with timestep conditioning."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_dim=64
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        self.time_to_down1 = nn.Linear(time_dim, 32)
        self.time_to_down2 = nn.Linear(time_dim, 64)
        self.time_to_bottleneck = nn.Linear(time_dim, 64)

        self.down1 = nn.Conv2d(
            in_channels,
            32,
            kernel_size=5,
            padding=2
        )

        self.down2 = nn.Conv2d(
            32,
            64,
            kernel_size=5,
            padding=2
        )

        self.bottleneck = nn.Conv2d(
            64,
            64,
            kernel_size=5,
            padding=2
        )

        self.up1 = nn.Conv2d(
            64,
            64,
            kernel_size=5,
            padding=2
        )

        self.up2 = nn.Conv2d(
            64,
            32,
            kernel_size=5,
            padding=2
        )

        self.output_layer = nn.Conv2d(
            32,
            out_channels,
            kernel_size=5,
            padding=2
        )

        self.activation = nn.SiLU()

        self.pool = nn.MaxPool2d(2)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

    def add_time_embedding(self, x, embedding):
        return x + embedding[:, :, None, None]

    def forward(self, x, timesteps):
        time_emb = self.time_embedding(timesteps)

        # Down path
        x1 = self.down1(x)

        x1 = self.add_time_embedding(
            x1,
            self.time_to_down1(time_emb)
        )

        x1 = self.activation(x1)

        x2 = self.pool(x1)

        x2 = self.down2(x2)

        x2 = self.add_time_embedding(
            x2,
            self.time_to_down2(time_emb)
        )

        x2 = self.activation(x2)

        x3 = self.pool(x2)

        x3 = self.bottleneck(x3)

        x3 = self.add_time_embedding(
            x3,
            self.time_to_bottleneck(time_emb)
        )

        x3 = self.activation(x3)

        # Up path
        x = self.up1(x3)
        x = self.activation(x)

        x = self.upsample(x)
        x = x + x2

        x = self.up2(x)
        x = self.activation(x)

        x = self.upsample(x)
        x = x + x1

        x = self.output_layer(x)

        return x


if __name__ == "__main__":
    model = TimeConditionedUNet()

    x = torch.rand(8, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (8,))

    output = model(x, timesteps)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Timesteps shape:", timesteps.shape)