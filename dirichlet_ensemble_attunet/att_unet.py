from __future__ import annotations

import torch
import torch.nn as nn

from dirichlet_ensemble_attunet.attention import SelfAttention
from dirichlet_ensemble_attunet.basic_blocks import Down, Plain, Up


class AttUNet(nn.Module):
    """AttUNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inner_channels: list[int],
        att_num_heads: int,
        residual: bool = True,
    ):
        """A simple Attention UNet.

        Args:
            in_channels (int): number of channels at the input
            out_channels (int): number of channels at the output
            inner_channels (list[int]): channel descriptions for the downsampling conv net
            att_num_heads (int): number of attention heads per attention module
            residual (bool): whether to have residual connections
        """
        super().__init__()

        # size of image must be multiples of this number
        self.quantize_size = 2 ** (len(inner_channels) - 1)
        self.residual = residual

        # ingest and outgest layers before and after unet
        self.ingest = Plain(in_channels, inner_channels[0])
        self.outgest = Plain(inner_channels[0], out_channels)

        # store the downs and ups in lists
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()

        # dynamically allocate the down and up list
        for i in range(len(inner_channels) - 1):
            self.down_list.append(Down(inner_channels[i], inner_channels[i + 1]))
            self.up_list.append(Up(inner_channels[-i - 1], inner_channels[-i - 2]))

        # init attention modules
        if att_num_heads:
            self.attention = SelfAttention(inner_channels[-1], att_num_heads, context_length=8192)
        else:
            self.attention = lambda x: x

    def _down(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """_down.

        Args:
            x (torch.Tensor): x

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
        """
        x = self.ingest(x)
        intermediates = [x := f(x) for f in self.down_list]
        return x, intermediates

    def _up(self, x: torch.Tensor, intermediates: list[torch.Tensor]) -> torch.Tensor:
        """_up.

        Args:
            x (torch.Tensor): the final output of the `down` component.
            intermediates (list[torch.Tensor]): the list of intermediary outputs from the `down` component.

        Returns:
            torch.Tensor:
        """
        for f, intermediate in zip(self.up_list, reversed(intermediates)):
            x = f(x + intermediate) if self.residual else f(x)
        x = self.outgest(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): an input of (..., C, H, W)

        Returns:
            torch.Tensor: an output of (..., C, H, W)
        """
        x, intermediate = self._down(x)
        x = self.attention(x)
        x = self._up(x, intermediate)

        return x
