import torch

from src.noise import corrupt


def test_corrupt_preserves_shape():
    x = torch.rand(8, 1, 28, 28)
    amount = torch.rand(8)

    noisy_x = corrupt(x, amount)

    assert noisy_x.shape == x.shape


def test_zero_noise_returns_original():
    x = torch.rand(8, 1, 28, 28)
    amount = torch.zeros(8)

    noisy_x = corrupt(x, amount)

    assert torch.allclose(noisy_x, x)