import torch

from src.unet import TimeConditionedUNet


def test_unet_output_shape():
    model = TimeConditionedUNet()

    x = torch.rand(8, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (8,))

    output = model(x, timesteps)

    assert output.shape == x.shape


def test_unet_has_trainable_parameters():
    model = TimeConditionedUNet()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count > 0