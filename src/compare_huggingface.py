import torch
from diffusers import UNet2DModel

from unet import TimeConditionedUNet


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def compare_models():
    custom_model = TimeConditionedUNet()

    hf_model = UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    )

    custom_params = count_parameters(custom_model)
    hf_params = count_parameters(hf_model)

    print("Custom TimeConditionedUNet parameters:", custom_params)
    print("Hugging Face UNet2DModel parameters:", hf_params)

    x = torch.randn(8, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (8,))

    custom_output = custom_model(x, timesteps)
    hf_output = hf_model(x, timesteps).sample

    print("Custom output shape:", custom_output.shape)
    print("Hugging Face output shape:", hf_output.shape)


if __name__ == "__main__":
    compare_models()