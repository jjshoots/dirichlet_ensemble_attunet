from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as func

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

    def binarize(
        self, x: torch.Tensor, prediction_threshold: float = 1.0
    ) -> torch.Tensor:
        """Converts an input of (num_ensemble, B, C, H, W) into (B, C, H, W) in {0, 1}.

        Args:
            x (torch.Tensor): input of (num_ensemble, B, C, H, W) in [-inf, +inf].
            prediction_threshold (float): proportion of `num_ensemble` that predicts True for the resulting output pixel to be True.

        Returns:
            torch.Tensor: output of (B, C, H, W) in {0, 1}, the dtype is Boolean.
        """

    def compute_uncertainty_map(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the uncertainty map given the output of the model.

        Args:
            y (torch.Tensor): y of shape (num_ensemble, B, C, H, W) in [-inf, +inf].

        Returns:
            torch.Tensor: pixelwise uncertainty of shape (B, C, H, W) in [0, 1].
        """
        # uncertainty is defined as
        # (sum(p_i ln p_i) / log(num_ensemble)) for i \in num_ensemble
        probs = func.softmax(y, dim=0)
        return (probs * probs.log()).sum(dim=0) / math.log(probs.shape[0])

    def compute_pixelwise_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        peak_distance: float = 0.0,
    ) -> torch.Tensor:
        """Computes a pixelwise BCE loss against a target.

        Args:
            x (torch.Tensor): input of shape (B, C, H, W).
            target (torch.Tensor): boolean target of shape (B, C, H, W).
            peak_distance (float): not used.

        Returns:
            torch.Tensor: pixelwise loss of shape (B, C, H, W).
        """
        y = self(x).expand(len(self.models), -1, -1, -1, -1)
        loss = func.binary_cross_entropy_with_logits(
            input=y,
            target=target.to(dtype=torch.float32, device=y.device),
            reduction="none",
        )
        return loss.mean(dim=0)
