import torch

from src.unet import BasicUNet


def test_unet_output_shape():
    model = BasicUNet()

    x = torch.rand(8, 1, 28, 28)

    output = model(x)

    assert output.shape == x.shape


def test_unet_has_trainable_parameters():
    model = BasicUNet()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count > 0