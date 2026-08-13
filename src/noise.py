import torch


def corrupt(x, amount):
    """
    Corrupt an image tensor by mixing it with random noise.

    amount = 0.0 -> original image
    amount = 1.0 -> pure random noise
    """

    noise = torch.rand_like(x)

    amount = amount.view(-1, 1, 1, 1)

    noisy_x = x * (1 - amount) + noise * amount

    return noisy_x