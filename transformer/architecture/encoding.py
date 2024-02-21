import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max__seq_length: int, embed_dim: int) -> None:
        """
        Positional encoder.

        Args:
            max__seq_length: Maximum expected sequence length
            embed_dim: Embedding dim, referred to as `d_model` in [1]

        [1] http://arxiv.org/abs/1706.03762
        """
        assert (
            embed_dim % 2 == 0
        ), f"Please choose an embedding dimension that is even!"

        super().__init__()

        # create initial encoding tensor filled with zeros
        pos_encod = torch.zeros(
            max__seq_length,
            embed_dim,
        )

        # sinusodial and cosinusoidal positional encodings
        pos_idx = torch.arange(
            0, max__seq_length, 1, dtype=torch.float32
        ).unsqueeze(
            1
        )  # `(max__seq_length, 1)`
        embed_idx = torch.arange(0, embed_dim // 2, 1, dtype=torch.float32)

        div = torch.exp(
            -2 * embed_idx / embed_dim * math.log(max__seq_length)
        )  # for numerical stability

        pos_encod[:, ::2] = torch.sin(pos_idx * div)
        pos_encod[:, 1::2] = torch.cos(pos_idx * div)
        pos_encod = pos_encod.unsqueeze(0)  # `(1, max__seq_length, embed_dim)`

        # not a trainable param, but make a registered buffer for device
        # handling; positional encoding doesn't need to be part of the state
        # dict, so use `persistent=False`
        self.register_buffer(
            name="pos_encod",
            tensor=pos_encod,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ "
        Forward pass.

        Args:
            x: Input tensor of shape `(N, seq_length, input_dim)`,
                where `input_dim = embed_dim = d_model`

        Returns:
            Tensor to which positional encoding is added in shape
            `(N, seq_length, input_dim)` if `seq_length <= max__seq_length`,
            else `(N, max__seq_length, input_dim)`
        """
        return x + self.pos_encod[:, : x.shape[1]]  # uses registered buffer
