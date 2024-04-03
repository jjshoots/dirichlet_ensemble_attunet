from typing import Literal

import torch
from torch import nn
from wingman import NeuralBlocks

activation_types = Literal["identity", "relu", "lrelu", "tanh"]


class MeanFilter(nn.Module):
    """MeanFilter."""

    def __init__(self, kernel_size: int = 3):
        """__init__.

        Args:
            kernel_size (int): kernel_size
        """
        super().__init__()
        # mean filter for uncertainty edges
        self.mean_filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            bias=False,
            padding_mode="reflect",
            groups=1,
        )
        self.mean_filter.weight = torch.nn.Parameter(
            torch.ones_like(self.mean_filter.weight) / (kernel_size**2)
        )
        self.mean_filter.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): x

        Returns:
            torch.Tensor:
        """
        return self.mean_filter(x)


def Plain(
    in_channel: int, out_channel: int, activation: activation_types = "identity"
) -> torch.nn.Module:
    """Just a plain ol' convolution layer.

    Args:
        in_channel:
        out_channel:

    Returns:
        torch.nn.Module: batchnorm -> conv
    """
    channels = [in_channel, out_channel]
    kernels = [3]
    pooling = [0]
    _activation = [activation] * len(kernels)

    return NeuralBlocks.generate_conv_stack(
        channels, kernels, pooling, _activation, norm="batch"
    )


def Down(
    in_channel: int, out_channel: int, activation: activation_types = "relu"
) -> torch.nn.Module:
    """Convolutional downscaler, downscales input by 2.

    Args:
        in_channel:
        out_channel:

    Returns:
        torch.nn.Module: batchnorm -> conv -> activation -> maxpool
    """
    channels = [in_channel, out_channel]
    kernels = [3]
    pooling = [2]
    _activation = [activation] * len(kernels)

    return NeuralBlocks.generate_conv_stack(
        channels, kernels, pooling, _activation, norm="batch"
    )


def Up(
    in_channel: int, out_channel: int, activation: activation_types = "relu"
) -> torch.nn.Module:
    """Convolutional upscaler, upscales input by 2.

    Args:
        in_channel:
        out_channel:

    Returns:
        torch.nn.Module: batchnorm -> deconv -> activation
    """
    channels = [in_channel, out_channel]
    kernels = [4]
    padding = [1]
    stride = [2]
    _activation = [activation] * len(kernels)

    return NeuralBlocks.generate_deconv_stack(
        channels, kernels, padding, stride, _activation, norm="batch"
    )
