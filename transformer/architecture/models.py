import math
from typing import Optional

import torch
from torch import nn

from .encoding import PositionalEncoding
from .layers import DecoderBlock


class Decoder(nn.Module):
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
        Transformer decoder.

        Args:
            num_layers: Number of times to stack the encoder block.
            embed_dim: Embedding dim, referred to as `d_model` in [1]
            num_heads: Number of heads, `h` in [1]
            dim_feedfwd: Hidden dimension when applying two-layer MLP
            dropout: Amount of dropout to be applied.
            use_bias: Whether a bias term is used. Default is `False`
        """
        super().__init__()
        self.num_layers = num_layers
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dim_feedfwd=dim_feedfwd,
                    dropout=dropout,
                    use_bias=use_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, T, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            mask: Mask for the target sequence, either 2D, 3D or 4D

        Returns:
            Output tensor of shape `(N, seq_length, input_dim)`

        [1] http://arxiv.org/abs/1706.03762
        """
        for idx in range(self.num_layers):
            x = self.decoder_blocks[idx](x=x, mask=mask)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num__decoder_layers: int,
        embedding_dim: int,
        num_heads: int,
        vocab_size: int,
        max__seq_length: int = int(1e4),
        dim_feedfwd: int = 2048,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
    ) -> None:
        """
        Transformer model.

        Args:
            num__decoder_layers: Number of times to stack the decoder block.
            embedding_dim: Embedding dim, referred to as `d_model` in [1].
            num_heads: Number of heads for the multi-head attention.
            vocab_size: Vocabulary size of the tokenizer.
            max__seq_length: Maximum expected sequence length.
            dim_feedfwd: Hidden dimension when applying two-layer MLP in
                encoder and decoder blocks.
            dropout_rate: Dropout rate.
            use_bias: Whether a bias term is used when performing the
                self-attention calculation. Default is `False`.

        Returns:
            Output tensor of shape `(N, num_classes)`

        [1] http://arxiv.org/abs/1706.03762
        """
        super().__init__()

        self.embed_dim = embedding_dim
        self.decoder = Decoder(
            num_layers=num__decoder_layers,
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dim_feedfwd=dim_feedfwd,
            dropout=dropout_rate,
            use_bias=use_bias,
        )
        self.pos_encod = PositionalEncoding(
            max__seq_length=max__seq_length,
            embed_dim=embedding_dim,
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.pre_softmax_linear = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )
        # weight sharing with the shared embedding
        self.pre_softmax_linear.weight = self.embedding.weight
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Args:
            input: Input tokens to decoder , shape: `(N, block_size)`.
            mask: Mask in shape `(block_size, block_size)`, prevent attending
                to subsequent tokens.

        Returns:
            Output tensor of shape `(N, block_size, vocab_size)`.
        """

        # embedding and positional encoding for the decoder,
        # `(N, block_size, embed_dim)`
        input = math.sqrt(self.embed_dim) * self.embedding(input)
        input = self.pos_encod(input)
        input = self.dropout(input)

        # forward pass through decoder and linear layer
        x = self.decoder(input, x, mask=mask)
        x = self.pre_softmax_linear(x)  # `(N, block_size, vocab_size)`

        return x
