from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """A very simple self attention module with sinusoidal positional embeddings."""

    def __init__(self, embed_dim: int, num_heads: int, context_length: int):
        """A very simple self attention module with sinusoidal positional embeddings.

        The internal dimension of the model is embed_dim / num_heads

        Args:
            embed_dim (int): the dimension size of the embeddings
            num_heads (int): the number of heads in the model
            context_length (int): maximum context length allowable
        """
        assert num_heads > 0
        super().__init__()

        self.embed_dim = embed_dim
        self.context_length = context_length
        self._normalization_const = np.sqrt(embed_dim)

        # multihead attention network
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # positional encoding, shape (N, 1, embed_dim)
        self.pos_encoding: torch.Tensor
        self.register_buffer(
            "pos_encoding", self._positional_encoding_1d(context_length, embed_dim)
        )

        # layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects inputs to be (batch_size, embedding_dim, *), where * is any number of dimensions.

        Args:
            x (int): An (batch_size, embedding_dim, *) shaped tensor, where * is any number of dimensions

        Returns:
            torch.Tensor: A (batch_size, embedding_dim, *) tensor
        """
        assert (
            len(x.shape) >= 2
        ), f"Input must be shape (batch_size, embedding_dim, *) where * is any number of dimensions, got {x.shape}."
        assert (
            x.shape[1] == self.embed_dim
        ), f"The size of vector {x.shape[1]=} must equal {self.embed_dim=}."
        assert (
            np.prod(x.shape[2:]) <= self.context_length
        ), f"tensor free dimension must be larger than {self.context_length=}, got {np.prod(x.shape[2:])=}."

        # store the shapes for reconstruction later
        wildcard_shape = x.shape[2:]

        # convert tensor to be (*, B, embed_dim)
        x = x.view(*x.shape[:2], -1).permute(2, 0, 1)

        # add positional encoding, output is (*, B, embed_dim)
        x = x + self.pos_encoding[: x.shape[0]]

        # pass through multihead attention plus residual, output is (*, B, embed_dim)
        y = self.mha(query=x, key=x, value=x, need_weights=False)[0] + x

        # layer norm, shape is (*, B, embed_dim)
        y = self.layer_norm(y)

        # reconstruct output into the expected shape
        y = y.permute((1, 2, 0)).view(*y.shape[-2:], *wildcard_shape)

        return y

    @classmethod
    def _positional_encoding_1d(cls, length: int, embed_dim: int) -> torch.Tensor:
        """positional_encoding_1d.

        Args:
            length (int): context length of the positional encoding
            embed_dim (int): the dimension size of the embeddings

        Returns:
            torch.Tensor: a (length, 1, embed_dim) sinusoidal positional encoding
        """
        if embed_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(embed_dim)
            )
        pe = torch.zeros(length, 1, embed_dim)
        position = torch.arange(0, length).unsqueeze(-1).unsqueeze(-1)
        div_term = (
            torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float)
                * -(np.log(10000.0) / embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        pe[..., 0::2] = torch.sin(position.float() * div_term)
        pe[..., 1::2] = torch.cos(position.float() * div_term)

        return pe


# for testing lol
if __name__ == "__main__":
    att = SelfAttention(256, 8, 10000)
    x = torch.zeros(8, 256, 2, 4, 5, 3, 4)
