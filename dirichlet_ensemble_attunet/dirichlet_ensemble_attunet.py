from __future__ import annotations

import torch
import torch.nn.functional as F

from dirichlet_ensemble_attunet.basic_blocks import activation_types
from dirichlet_ensemble_attunet.ensemble_att_unet import EnsembleAttUNet


class DirichletEnsembleAttUNet(EnsembleAttUNet):
    """DirichletEnsembleAttUNet."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        inner_channels: list[int],
        att_num_heads: int,
        att_num_layers: int,
        activation: activation_types,
        residual: bool,
        num_ensemble: int,
    ):
        """Dirichlet Ensemble Attention U Net.

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
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            inner_channels=inner_channels,
            att_num_heads=att_num_heads,
            att_num_layers=att_num_layers,
            activation=activation,
            residual=residual,
            num_ensemble=num_ensemble,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function.

        Args:
            x (torch.Tensor): input of (B, C, H, W).

        Returns:
            torch.Tensor: output of (pos_neg, num_ensemble, B, C, H, W) in [0, +inf].
        """
        # x here is [num_ensemble, B, C, H, W]
        x = super().forward(x)

        # x here is [pos_neg, num_ensemble, B, C, H, W] in [0, +inf]
        x = torch.stack(
            [x[..., : self.out_channels, :, :], x[..., self.out_channels :, :, :]],
            dim=0,
        )

        # use evidence
        x = F.softplus(x) + 1.0

        # x here is [pos_neg, num_ensemble, B, C, H, W] in [0, +inf]
        return x

    def binarize(
        self, x: torch.Tensor, prediction_threshold: float = 1.0
    ) -> torch.Tensor:
        """Converts an input of (pos_neg, num_ensemble, B, C, H, W) into (B, C, H, W) in {0, 1}.

        Args:
            x (torch.Tensor): input of (pos_neg, num_ensemble, B, C, H, W) in [0, +inf].
            prediction_threshold (float): proportion of `num_ensemble` that predicts True for the resulting output pixel to be True.

        Returns:
            torch.Tensor: output of (B, C, H, W) in {0, 1}, the dtype is Boolean.
        """
        # input x should be [pos_neg, num_ensemble, B, C, H, W]
        # the output is [B, C, H, W] in {0, 1}
        return (x[0] > x[1]).float().mean(dim=0) >= prediction_threshold

    def compute_pixelwise_loss(
        self, x: torch.Tensor, target: torch.Tensor, peak_distance: float = 16.0
    ) -> torch.Tensor:
        """Computes a pixelwise loss against a target.

        Args:
            x (torch.Tensor): input of shape (B, C, H, W).
            target (torch.Tensor): boolean target of shape (B, C, H, W).
            peak_distance (float): the maximum dirichlet distance between positive and negative.

        Returns:
            torch.Tensor: pixelwise loss of shape (B, C, H, W).
        """
        assert (
            target.shape[-2:] == x.shape[-2:]
        ), f"The target's spatial shape {target.shape[-2:]} should be the same as the input's {x.shape[-2:]}."
        assert (
            target.dtype == torch.bool
        ), f"The target should be a boolean tensor, got {target.dtype}."

        # the output of this is [pos_neg, num_ensemble, B, C, H, W] in [0, +inf]
        y = self(x)

        # the ensemble loss is [num_ensemble, B, C, H, W]
        ensemble_loss = (
            # this serves to keep evidence low
            torch.log(y[0] + y[1])
            # this increases positive evidence if we need it to be positive, but only if there is distance available
            - torch.log(y[0]) * target * (y[0] < y[1] + peak_distance)
            # this increases negative evidence if we need it to be negative, but only if there is distance available
            - torch.log(y[1]) * ~target * (y[1] < y[0] + peak_distance)
        )

        # take the mean over the ensemble dimension, the resulting output is [B, C, H, W]
        return torch.mean(ensemble_loss, dim=0)
