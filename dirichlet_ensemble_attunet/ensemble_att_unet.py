from __future__ import annotations

import torch
import torch.nn as nn

from dirichlet_ensemble_attunet.att_unet import AttUNet
from dirichlet_ensemble_attunet.basic_blocks import activation_types


class EnsembleAttUNet(nn.Module):
    """EnsembleAttUNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inner_channels: list[int],
        att_num_heads: int,
        att_num_layers: int,
        activation: activation_types,
        residual: bool,
        num_ensemble: int,
    ):
        """A simple Attention UNet.

        Args:
            in_channels (int): number of channels at the input
            out_channels (int): number of channels at the output
            inner_channels (list[int]): channel descriptions for the downsampling conv net
            att_num_heads (int): number of attention heads per attention module
            att_num_layers (int): number of attention layers in the attention module
            activation (activation_types): type of activation to use in the downscaling and upscaling layers
            residual (bool): whether to have residual connections
            num_ensemble (int): number of networks in the ensemble
        """
        super().__init__()

        self.models = nn.ModuleList(
            [
                AttUNet(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    inner_channels=inner_channels,
                    att_num_heads=att_num_heads,
                    att_num_layers=att_num_layers,
                    activation=activation,
                    residual=residual,
                )
                for _ in range(num_ensemble)
            ]
        )
        self.quantize_size = self.models[0].quantize_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function.

        Args:
            x (torch.Tensor): input of (B, C, H, W).

        Returns:
            torch.Tensor: output of (num_ensemble, B, C, H, W) in [-inf, +inf].
        """
        # output here is [num_ensemble, B, C, H, W]
        return torch.stack([f(x) for f in self.models], dim=0)
