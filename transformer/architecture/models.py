import math
from typing import Optional

import torch
from torch import nn

from .attention import get_subsequent_mask
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
        # weight sharing with the shared embedding (not sure why, but it has
        # to be like this, otherwise loss of randomly initialized model is too
        # high)
        self.embedding.weight = self.pre_softmax_linear.weight
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Args:
            x: Input tokens to decoder , shape: `(N, block_size)`.
            mask: Mask in shape `(block_size, block_size)`, prevent attending
                to subsequent tokens.

        Returns:
            Output tensor of shape `(N, block_size, vocab_size)`.
        """

        # embedding and positional encoding for the decoder,
        # `(N, block_size, embed_dim)`
        x = math.sqrt(self.embed_dim) * self.embedding(x)
        x = self.pos_encod(x)
        x = self.dropout(x)

        # forward pass through decoder and linear layer
        x = self.decoder(x, mask=mask)
        x = self.pre_softmax_linear(x)  # `(N, block_size, vocab_size)`

        return x

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        block_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.tensor:
        """
        Generate text using the transformer model.

        Args:
            x: Input tokens to decoder, shape: `(N, T)`.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Temperature for sampling. For `temperature > 1`,
                predictions will be more diverse, for `temperature < 1`,
                predictions will be more conservative.
            top_k: Top-k sampling.

        Returns:
            Output tensor of shape `(N, T + max_new_tokens)`.
        """
        for _ in range(max_new_tokens):
            # truncate input if it exceeds the block size
            x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
            # generate mask
            mask = get_subsequent_mask(size=x_cond.shape[1], rank=x.device)
            # get model predictions for next token
            logits = self(x_cond, mask=mask)
            # get logits at last token in sequence and scale by temperature
            logits = logits[:, -1, :] / temperature
            # apply top-k sampling
            if top_k is not None:
                max_vals, _ = torch.topk(
                    logits, k=min(top_k, logits.shape[-1]), dim=-1
                )
                logits[logits < max_vals[:, [-1]]] = float("-inf")
            # convert logits to probabilities
            probs = nn.Softmax(dim=-1)(logits)
            # sample from distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append new token to sequence
            x = torch.cat([x, next_token], dim=-1)

        return x
