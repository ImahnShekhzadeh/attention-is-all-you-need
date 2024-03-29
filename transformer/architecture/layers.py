from typing import Optional

import torch
from torch import nn

from .attention import DecoderMultiHeadAttention, MultiHeadAttention


class EncoderBlock(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, S, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            mask: Mask for the source sequence, either 2D, 3D or 4D
            src_key_padding_mask: Mask for source keys, shape: `(N, S)`

        Returns:
            Output tensor of shape `(N, S, input_dim)`
        """

        # multi-head attention part
        out = self.multihead_attn(
            x=x,
            attn_mask=mask,
            key_padding_mask=src_key_padding_mask,
        )
        out = self.norm_a(self.dropout(out) + x)

        # feed-forward part
        feedfwd_out = self.dropout(self.mlp(out))
        out = self.norm_b(feedfwd_out + out)

        return out


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

        # multi-head attention layer, where keys and values come from encoder
        # output
        self.decoder__multihead_attn = DecoderMultiHeadAttention(
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
        self.norm_c = nn.LayerNorm(
            normalized_shape=[embed_dim],
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, T, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            encoder_output: Encoder output, shape: `(N, S, embed_dim)`
                (`embed_dim = d_model` in [1])
            mask: Mask for the target sequence, either 2D, 3D or 4D (prevents
                attending to subsequent tokens)
            tgt_key_padding_mask: Mask for target keys, shape: `(N, T)`

        Returns:
            Output tensor of shape `(N, T, input_dim)`
        """

        # multi-head attention part
        out_a = self.multihead_attn(
            x=x,
            attn_mask=mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        out_a = self.norm_a(self.dropout(out_a) + x)

        # multi-head attention part, where queries and keys come from encoder
        # output
        out_b = self.decoder__multihead_attn(
            x=out_a,
            encoder_output=encoder_output,
        )
        out_b = self.norm_b(self.dropout(out_b) + out_a)

        # feed-forward part
        feedfwd_out = self.dropout(self.mlp(out_b))
        out = self.norm_c(feedfwd_out + out_b)

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
