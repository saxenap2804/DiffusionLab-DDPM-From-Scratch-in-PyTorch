import torch
from torch import nn


class BasicUNet(nn.Module):
    """A minimal U-Net for 28x28 grayscale images."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])

        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])

        self.activation = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        skip_connections = []

        # Downsampling path
        for i, layer in enumerate(self.down_layers):
            x = self.activation(layer(x))

            if i < 2:
                skip_connections.append(x)
                x = self.downscale(x)

        # Upsampling path
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x = x + skip_connections.pop()

            x = self.activation(layer(x))

        return x


if __name__ == "__main__":
    model = BasicUNet()

    sample = torch.rand(8, 1, 28, 28)
    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Number of parameters:", parameter_count)