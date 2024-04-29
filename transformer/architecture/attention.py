import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import softmax
from torch.nn.init import xavier_uniform_


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Helper function to support different mask shapes.
    Output shape supports `(N, num_heads, seq_length, seq_length)`
    If 2D: broadcasted over `N` and `num_heads`
    If 3D: broadcasted over `num_heads`
    If 4D: leave as is

    Args:
        mask: Mask.

    Returns:
        Expanded mask of shape `(N, num_heads, seq_length, seq_length)`
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
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
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
        attn_mask: Optional look-ahead mask. All values set to `1` will
            be excluded for attention calculation.
        key_padding_mask: Optional mask for padding of the keys. All values
            set to `1` will be excluded for attention calculation.

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
    # `(N, num_heads, seq_length, seq_length')`,
    # `(N, seq_length, seq_length')` or `(seq_length, seq_length')`
    attn_logits = q @ k.mT / math.sqrt(q.shape[-1])

    # apply masks if provided
    if attn_mask is not None:
        attn_logits.masked_fill_(attn_mask == 1, -float("inf"))

    if key_padding_mask is not None:
        if attn_logits.ndim == 2:
            assert key_padding_mask.ndim <= 2, (
                "For unbatched queries, keys and values, padding mask must be "
                f"1D of shape {attn_logits.shape[1]} or 2D of shape "
                f"``(1, {attn_logits.shape[1]})``. Instead, padding mask has "
                f"shape {key_padding_mask.shape}"
            )
        elif attn_logits.ndim == 3:
            assert key_padding_mask.ndim == 2, (
                "For batched queries, keys and values, padding mask must be "
                f"2D of shape ({attn_logits.shape[0]}, {attn_logits.shape[2]})"
                f". Instead, padding mask has shape {key_padding_mask.shape}."
            )
            key_padding_mask = key_padding_mask.unsqueeze(1)
        elif attn_logits.ndim == 4:
            assert key_padding_mask.ndim == 2, (
                "For batched queries, keys and values, padding mask must be "
                f"2D of shape ({attn_logits.shape[0]}, {attn_logits.shape[2]})"
                f". Instead, padding mask has shape {key_padding_mask.shape}."
            )
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

        attn_logits.masked_fill_(key_padding_mask == 1, -float("inf"))

    # calculate attention weights
    attn_weights = softmax(attn_logits, dim=-1)

    return attn_weights @ v, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_bias: bool = False,
    ) -> None:
        """
        Multi-head attention.

        Args:
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

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        # dim of queries, keys and values per self-attention head:
        # (cf. Sec. 3.2.2 of [1])
        self.head_dim = int(embed_dim / num_heads)

        # stack all weight matrices per self-attention head
        self.qkv_proj = nn.Linear(
            in_features=embed_dim,
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
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor in shape `(N, seq_length, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            attn_mask: Optional look-ahead mask. All values set to `1` will
                be excluded for attention calculation.
            key_padding_mask: Optional mask for padding of the keys. All values
                set to `1` will be excluded for attention calculation.
            return_attention: Whether to return the attention weights

        Returns:
            Output in shape `(N, seq_length, self.embed_dim)` and optionally
            attention weights in shape
            `(N, self.num_heads, seq_length, seq_length)`
        """
        if attn_mask is not None:
            attn_mask = expand_mask(attn_mask)
        qkv_proj = self.qkv_proj(x)  # `(N, seq_length, 3 * embed_dim)`

        # reshape and permute into
        # `(N, self.num_heads, seq_length, 3 * self.head_dim)`
        # note that `self.embed_dim = self.num_heads * self.head_dim`
        qkv_proj = qkv_proj.reshape(
            qkv_proj.shape[0],
            qkv_proj.shape[1],
            self.num_heads,
            3 * self.head_dim,
        ).permute(dims=(0, 2, 1, 3))

        # separate queries, keys and values from reshaped and permuted
        # projection `(N, self.num_heads, seq_length, self.head_dim)`
        q_proj, k_proj, v_proj = qkv_proj.chunk(chunks=3, dim=-1)

        # Determine value outputs
        # `(N, self.num_heads, seq_length, self.head_dim)`
        values, attn_weights = scaled_dot_product_attn(
            q_proj,
            k_proj,
            v_proj,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        values = (values := values.permute(0, 2, 1, 3)).reshape(
            values.shape[0], values.shape[1], self.embed_dim
        )
        o = self.o_proj(values)  # `(N, seq_length, self.embed_dim)`

        if return_attention:
            return o, attn_weights
        else:
            return o
