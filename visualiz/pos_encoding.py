import math

import torch
from torch import nn
from matplotlib import pyplot as plt


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


if __name__ == "__main__":
    encoder_class = PositionalEncoding(max__seq_length=96, embed_dim=48)
    pos_encod = encoder_class.pos_encod.squeeze().T  # transpose because of `.imshow`
    print(f"{pos_encod.shape}")

    plt.imshow(pos_encod, cmap="RdGy")
    plt.xlabel("Token position")
    plt.ylabel("Embedding dim")
    plt.colorbar(shrink=0.5)
    plt.savefig(
        f"positional_encoding_all_embedding_dims.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )
    plt.close()

    fig, axs = plt.subplots(nrows=3, ncols=2,)
    for (idx, ax) in enumerate(axs.flat):
        ax.plot(pos_encod[idx])
        ax.set_title(f"Embedding dim: {idx}")
        ax.set_xlabel("Token position")
    plt.tight_layout()
    plt.savefig(
        f"positional_encoding_few_embedding_dims.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )
    plt.close()