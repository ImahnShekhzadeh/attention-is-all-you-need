import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import softmax
from torch.nn.init import xavier_uniform_


class PositionalEncoding(nn.Module):
    def __init__(self, max__seq_length: int, embed_dim: int) -> None:
        """
        Positional encoder.

        Args:
            max__seq_length: Maximum expected sequence length, chosen as
                `10000` in [1]
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
        return x + self.pos_encod(x)[:, : x.shape[1]]  # uses registered buffer


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


def scaled_dot_product_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implement scaled dot-product attention for batched queries, keys and
    values.

    Args:
        q: Queries in shape `(N, num_heads, seq_length, d_k)`,
            `(N, seq_length, d_k)` or `(seq_length, d_k)`
        k: Keys in shape `(N, num_heads, seq_length', d_k)`,
            `(N, seq_length', d_k)` or `(seq_length', d_k)`
            (in practice, `seq_length' == seq_length`)
        v: Values in shape `(N, num_heads, seq_length', d_v)`,
            `(N, seq_length', d_v)` or `(seq_length', d_v)`
        mask: Optional mask for padding.

    Returns:
        Weighted values $softmax(qk^T / sqrt(d_k)) v$ in shape
        `(N, num_heads, seq_length, d_v)`, `(N, seq_length, d_v)` or
        `(seq_length, d_v,)` and attention weights
        $softmax(qk^T / sqrt(d_k))$ in shape
        `(N, num_heads, seq_length, seq_length')`,
        `(N, seq_length, seq_length')` or `(seq_length, seq_length')`
    """

    # shape checks
    assert q.shape[-1] == k.shape[-1]
    assert k.shape[-2] == v.shape[-2]

    # calculate attention logits
    # `(N, num_heads, seq_length, seq_length')`
    attn_logits = q @ k.mT / math.sqrt(q.shape[-1])

    # apply mask if provided
    if mask is not None:
        attn_logits.masked_fill_(mask == 0, -1e6)

    # calculate attention weights
    attn_weights = softmax(attn_logits, dim=-1)

    return attn_weights @ v, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        use_bias: bool = False,
    ) -> None:
        """
        Multi-head attention.

        Args:
            input_dim: In the attention paper [1], `d_k = d_v`, which is here
                referred to as the input dimensionality
            embed_dim: Embedding dim, referred to as `d_model` in [1]
            num_heads: Number of heads, `h` in [1]
            use_bias: Whether a bias term is used. Default is `False`

        [1] http://arxiv.org/abs/1706.03762
        """
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "In the original attention paper, `d_model = hd_v = hd_k`"
            f"was chosen, hence `d_model`:`num_heads` cannot be {embed_dim}: "
            f"{num_heads}"
        )

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        # dim of queries, keys and values per self-attention head:
        # (cf. Sec. 3.2.2 of [1])
        self.head_dim = int(embed_dim / num_heads)

        # stack all weight matrices per self-attention head
        self.qkv_proj = nn.Linear(
            in_features=input_dim,
            out_features=3 * embed_dim,
            bias=use_bias,
        )
        self.o_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=use_bias,
        )  # `W^O`

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize (reset) params.
        """

        # weights:
        xavier_uniform_(self.qkv_proj.weight)
        xavier_uniform_(self.o_proj.weight)

        if self.use_bias:
            # biases:
            self.qkv_proj.bias.data.fill_(0)
            self.o_proj.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor in shape `(N, seq_length, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            mask: Mask, either 2D, 3D or 4D
            return_attention: Whether to return the attention weights

        Returns:
            Output in shape `(N, seq_length, self.embed_dim)` and optionally
            attention weights in shape
            `(N, self.num_heads, seq_length, seq_length)`
        """
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)  # `(N, seq_length, 3 * embed_dim)`

        # reshape into `(N, seq_length, self.num_heads, 3 * self.head_dim)`;
        # note that `self.embed_dim = self.num_heads * self.head_dim`
        qkv = qkv.reshape(
            qkv.shape[0], qkv.shape[1], self.num_heads, 3 * self.head_dim
        )

        # `(N, self.num_heads, seq_length, 3 * self.head_dim)`
        qkv = qkv.permute(dims=(0, 2, 1, 3))

        # separate queries, keys and values from reshaped and permuted
        # projection `(N, self.num_heads, seq_length, self.head_dim)`
        q_proj, k_proj, v_proj = qkv.chunk(chunks=3, dim=-1)

        # Determine value outputs
        # shape of `values`: `(N, self.num_heads, seq_length, self.head_dim)`
        # shape of `attn_weights`:
        # `(N, self.num_heads, seq_length, seq_length')`
        values, attn_weights = scaled_dot_product_attn(
            q_proj, k_proj, v_proj, mask=mask
        )
        values = values.permute(
            0, 2, 1, 3
        )  # `(N, seq_length, self.num_heads, self.head_dim)`
        values = values.reshape(
            values.shape[0], values.shape[1], self.embed_dim
        )
        o = self.o_proj(values)  # `(N, seq_length, self.embed_dim)`

        if return_attention:
            return o, attn_weights
        else:
            return o


class DecoderMultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        use_bias: bool = False,
    ) -> None:
        """
        Multi-head attention, where the queries and keys are taken from the
        output of the encoder, and the values come from the previous masked
        multi-head attention part in the decoder.

        Args:
            input_dim: In the attention paper [1], `d_k = d_v`, which is here
                referred to as the input dimensionality
            embed_dim: Embedding dim, referred to as `d_model` in [1]
            num_heads: Number of heads, `h` in [1]
            use_bias: Whether a bias term is used. Default is `False`

        [1] http://arxiv.org/abs/1706.03762
        """
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "In the original attention paper, `d_model = hd_v = hd_k`"
            f"was chosen, hence `d_model`:`num_heads` cannot be {embed_dim}: "
            f"{num_heads}"
        )

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        # dim of queries, keys and values per self-attention head:
        # (cf. Sec. 3.2.2 of [1])
        self.head_dim = int(embed_dim / num_heads)

        # stack querky and key weight matrices per self-attention head
        self.qk_proj = nn.Linear(
            in_features=input_dim,
            out_features=2 * embed_dim,
            bias=use_bias,
        )
        self.v_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=use_bias,
        )
        self.o_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=use_bias,
        )  # `W^O`

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize (reset) params.
        """

        # weights:
        xavier_uniform_(self.qk_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.o_proj.weight)

        if self.use_bias:
            # biases:
            self.qkv_proj.bias.data.fill_(0)
            self.o_proj.bias.data.fill_(0)

    def forward(
        self,
        x_encoder: torch.Tensor,
        x_decoder: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_encoder: Input tensor in shape `(N, seq_length, embed_dim)`
                that comes from the encoder output (`embed_dim = d_model`
                in [1])
            x_decoder: Input tensor in shape `(N, seq_length, embed_dim)`
                that comes from the previous masked multi-head attention
                in the decoder
            mask: Mask, either 2D, 3D or 4D
            return_attention: Whether to return the attention weights

        Returns:
            Output in shape `(N, seq_length, self.embed_dim)` and optionally
            attention weights in shape
            `(N, self.num_heads, seq_length, seq_length)`
        """
        if mask is not None:
            mask = expand_mask(mask)
        qk = self.qk_proj(x_encoder)  # `(N, seq_length, 2 * embed_dim)`
        v = self.v_proj(x_decoder)  # `(N, seq_length, embed_dim)`

        # reshape into `(N, seq_length, self.num_heads, 2 * self.head_dim)`,
        # then permute into
        # `(N, self.num_heads, seq_length, 2 * self.head_dim)`;
        # note that `self.embed_dim = self.num_heads * self.head_dim`
        qk = qk.reshape(
            qk.shape[0], qk.shape[1], self.num_heads, 2 * self.head_dim
        ).permute(dims=(0, 2, 1, 3))

        v = v.reshape(
            v.shape[0], v.shape[1], self.num_heads, self.head_dim
        ).permute(dims=(0, 2, 1, 3))

        # separate queries and keys
        # `(N, self.num_heads, seq_length, self.head_dim)`
        q_proj, k_proj = qk.chunk(chunks=2, dim=-1)

        # Determine value outputs
        # shape of `values`: `(N, self.num_heads, seq_length, self.head_dim)`
        # shape of `attn_weights`:
        # `(N, self.num_heads, seq_length, seq_length')`
        values, attn_weights = scaled_dot_product_attn(
            q_proj, k_proj, v, mask=mask
        )
        # permute and reshape
        # `(N, seq_length, self.embed_dim)`
        values = values.permute(dims=(0, 2, 1, 3)).reshape(
            values.shape[0], values.shape[1], self.embed_dim
        )
        o = self.o_proj(values)  # `(N, seq_length, self.embed_dim)`

        if return_attention:
            return o, attn_weights
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedfwd: int = 2048,
        dropout: bool = 0.0,
        use_bias: bool = False,
    ) -> None:
        """
        Initialization function.

        Args:
            input_dim: In the attention paper [1], `d_k = d_v`, which is here
                referred to as the input dimensionality
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
            input_dim=input_dim,
            embed_dim=input_dim,
            num_heads=num_heads,
            use_bias=use_bias,
        )

        # two-layer MLP (called "feed forward" in [1], cf. Eq. (2) in [1])
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=input_dim, out_features=dim_feedfwd, bias=True
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(
                in_features=dim_feedfwd, out_features=input_dim, bias=True
            ),
        )

        # layers applied between the main layers
        self.norm_a = nn.LayerNorm(
            normalized_shape=[input_dim],
        )
        self.norm_b = nn.LayerNorm(
            normalized_shape=[input_dim],
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: bool = None) -> Tensor:
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
        input_dim: int,
        num_heads: int,
        dim_feedfwd: int,
        dropout: bool = 0.0,
        use_bias: bool = False,
    ) -> None:
        """
        Transformer encoder.

        Args:
            num_layers: Number of times to stack the encoder block.
            input_dim: In the attention paper [1], `d_k = d_v`, which is here
                referred to as the input dimensionality
            num_heads: Number of heads, `h` in [1]
            dim_feedfwd: Hidden dimension when applying two-layer MLP
            dropout: Amount of dropout to be applied.
            use_bias: Whether a bias term is used. Default is `False`

        [1] http://arxiv.org/abs/1706.03762
        """
        self.num_layers = num_layers
        self.encoder_block = EncoderBlock(
            input_dim=input_dim,
            num_heads=num_heads,
            dim_feedfwd=dim_feedfwd,
            dropout=dropout,
            use_bias=use_bias,
        )

    def forward(self, x: torch.Tensor, mask: bool = None) -> Tensor:
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

    def _get_attn_maps(self, mask: bool = None) -> List[Tensor]:
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
