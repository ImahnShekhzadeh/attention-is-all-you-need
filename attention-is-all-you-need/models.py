from typing import List, Optional

import torch
from torch import Tensor, nn

from attention import DecoderMultiHeadAttention, MultiHeadAttention


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Helper function to support different mask shapes.
    Output shape supports `(batch_size, num_heads, seq_length, seq_length)`
    If 2D: broadcasted over `batch_size` and `num_heads`
    If 3D: broadcasted over `num_heads`
    If 4D: leave as is

    Args:
        mask: Mask.
    """
    assert (
        mask.ndim >= 2
    ), "Mask must be at least 2-dimensional with `seq_length x seq_length`"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


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
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embed_dim, out_features=dim_feedfwd, bias=True
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=dim_feedfwd, out_features=embed_dim, bias=True
            ),
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
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, seq_length, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            mask: Mask, either 2D, 3D or 4D

        Returns:
            Output tensor of shape `(N, seq_length, input_dim)`
        """

        # multi-head attention part
        out = self.multihead_attn(
            x=x,
            mask=mask,
        )
        out = self.norm_a(self.dropout(out) + x)

        # feed-forward part
        feedfwd_out = self.dropout(self.mlp(out))
        out = self.norm_b(feedfwd_out + out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        dim_feedfwd: int,
        dropout: bool = 0.0,
        use_bias: bool = False,
    ) -> None:
        """
        Transformer encoder.

        Args:
            num_layers: Number of times to stack the encoder block.
            embed_dim: Embedding dim, referred to as `d_model` in [1]
            num_heads: Number of heads, `h` in [1]
            dim_feedfwd: Hidden dimension when applying two-layer MLP
            dropout: Amount of dropout to be applied.
            use_bias: Whether a bias term is used. Default is `False`

        [1] http://arxiv.org/abs/1706.03762
        """
        self.num_layers = num_layers
        self.encoder_block = EncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedfwd=dim_feedfwd,
            dropout=dropout,
            use_bias=use_bias,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, seq_length, input_dim)`
            mask: Mask, either 2D, 3D or 4D

        Returns:
            Output tensor of shape `(N, seq_length, input_dim)`
        """
        for _ in range(self.num_layers):
            x = self.encoder_block(
                x=x,
                mask=mask,
            )

        return x

    def _get_attn_maps(
        self, mask: Optional[torch.Tensor] = None
    ) -> List[Tensor]:
        """
        Retrieve the learned attention maps per head.

        Args:
            mask: Mask, either 2D, 3D or 4D

        Returns:
            List of PyTorch tensors containing the attention weights per
            encoder block, where each tensor is of shape `(N, num_heads,
            seq_length, seq_length)`
        """
        attn_maps = []

        for _ in range(self.num_layers):
            _, attn_weights = self.encoder_block.multihead_attn(
                x=x,
                mask=mask,
                return_attention=True,
            )
            x = self.encoder_block(
                x=x,
                mask=mask,
            )
            attn_maps.append(attn_weights)

        return attn_maps


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

        # multi-head attention layer, where queries and keys come from encoder
        # output
        self.decoder__multihead_attn = DecoderMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_bias=use_bias,
        )

        # two-layer MLP (called "feed forward" in [1], cf. Eq. (2) in [1])
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embed_dim, out_features=dim_feedfwd, bias=True
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=dim_feedfwd, out_features=embed_dim, bias=True
            ),
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
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, seq_length, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            encoder_output: Encoder output in shape
                `(N, seq_length, embed_dim)` (`embed_dim = d_model` in [1])
            mask: Mask, either 2D, 3D or 4D (prevents attending to future)

        Returns:
            Output tensor of shape `(N, seq_length, input_dim)`
        """

        # multi-head attention part
        out_a = self.multihead_attn(
            x=x,
            mask=mask,
        )
        out_a = self.norm_a(self.dropout(out_a) + x)

        # multi-head attention part, where queries and keys come from encoder
        # output
        out_b = self.decoder__multihead_attn(
            x=out_a,
            encoder_output=encoder_output,
            mask=mask,
        )
        out_b = self.norm_b(self.dropout(out_b) + out_a)

        # feed-forward part
        feedfwd_out = self.dropout(self.mlp(out_b))
        out = self.norm_c(feedfwd_out + out_b)

        return out
