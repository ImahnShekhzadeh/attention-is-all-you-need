import torch
from torch import nn

from .attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedfwd: int = 2048,
        dropout: bool = 0.0,
        use_bias: bool = False,
    ) -> None:
        """
        Initialization function.

        Args:
            embed_dim: Embedding dim, referred to as `d_model` in [1]
            num_heads: Number of heads, `h` in [1]
            dim_feedfwd: Hidden dimension when applying two-layer MLP
            dropout: Amount of dropout to be applied.
            use_bias: Whether a bias term is used. Default is `False`

        [1] http://arxiv.org/abs/1706.03762
        """
        super().__init__()

        # check dropout rate
        assert 0 <= dropout <= 1, (
            f"Invalid amount of droput (`{dropout}`) specified. "
            f"Dropout rate should be between `0` and `1`."
        )

        # multi-head attention layer
        self.multihead_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_bias=use_bias,
        )

        # two-layer MLP (called "feed forward" in [1], cf. Eq. (2) in [1])
        self.mlp = PositionwiseFeedForward(
            embed_dim=embed_dim, dim_feedfwd=dim_feedfwd
        )

        # layers applied between the main layers
        self.norm_a = nn.LayerNorm(
            normalized_shape=[embed_dim],
        )
        self.norm_b = nn.LayerNorm(
            normalized_shape=[embed_dim],
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, T, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            mask: Mask for the target sequence, either 2D, 3D or 4D (prevents
                attending to subsequent tokens)

        Returns:
            Output tensor of shape `(N, T, input_dim)`
        """

        # multi-head attention part
        out_a = self.multihead_attn(x=x, attn_mask=mask)
        out_a = self.norm_a(self.dropout(out_a) + x)

        # feed-forward part
        feedfwd_out = self.mlp(out_a)
        out = self.norm_b(self.dropout(feedfwd_out) + out_a)

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dim_feedfwd: int,
    ) -> None:
        """
        Two-layer MLP as described in Eq. (2) in [1] that is applied to each
        position separately and identically.

        Args:
            embed_dim: Embedding dim, referred to as `d_model` in [1]
            dim_feedfwd: Hidden dimension when applying two-layer MLP

        [1] http://arxiv.org/abs/1706.03762
        """
        super().__init__()

        # two-layer MLP (called "feed forward" in [1])
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embed_dim, out_features=dim_feedfwd, bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=dim_feedfwd, out_features=embed_dim, bias=True
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, seq_length, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])

        Returns:
            Output tensor of shape `(N, seq_length, input_dim)`
        """

        return self.mlp(x)
