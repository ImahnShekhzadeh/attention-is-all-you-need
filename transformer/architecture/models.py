import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from .encoding import PositionalEncoding
from .layers import DecoderBlock, EncoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        dim_feedfwd: int = 2048,
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
        super().__init__()
        self.num_layers = num_layers
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
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
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, seq_length, input_dim)`
            mask: Mask, either 2D, 3D or 4D

        Returns:
            Output tensor of shape `(N, seq_length, input_dim)`
        """
        for idx in range(self.num_layers):
            x = self.encoder_blocks[idx](
                x=x,
                mask=mask,
            )

        return x

    def _get_attn_maps(
        self, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Retrieve the learned attention maps per head.

        Args:
            mask: Mask, either 2D, 3D or 4D

        Returns:
            List of PyTorch tensors containing the attention weights per
            encoder block, where each tensor is of shape
            `(N, num_heads, seq_length, seq_length)`
        """
        attn_maps = []

        for idx in range(self.num_layers):
            _, attn_weights = self.encoder_blocks[idx].multihead_attn(
                x=x,
                mask=mask,
                return_attention=True,
            )
            x = self.encoder_blocks[idx](
                x=x,
                mask=mask,
            )
            attn_maps.append(attn_weights)

        return attn_maps


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
            --- cf. `Encoder` ---
        """
        super().__init__()
        self.num_layers = num_layers
        self.decoder_block = nn.ModuleList(
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
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape `(N, seq_length, input_dim)`
                (`input_dim = embed_dim = d_model` in [1])
            mask: Mask, either 2D, 3D or 4D

        Returns:
            Output tensor of shape `(N, seq_length, input_dim)`

        [1] http://arxiv.org/abs/1706.03762
        """
        for idx in range(self.num_layers):
            x = self.decoder_blocks[idx](
                x=x,
                encoder_output=encoder_output,
                mask=mask,
            )

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num__encoder_layers: int,
        num__decoder_layers: int,
        embedding_dim: int,
        num_heads: int,
        vocab_size: int,
        seq_length: int = int(1e4),
        dim_feedfwd: int = 2048,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
    ) -> None:
        """
        Transformer model.

        Args:
            num__encoder_layers: Number of times to stack the encoder block.
            num__decoder_layers: Number of times to stack the decoder block.
            embedding_dim: Embedding dim, referred to as `d_model` in [1].
            num_heads: Number of heads for the multi-head attention.
            vocab_size: Vocabulary size of the tokenizer.
            seq_length: Maximum expected sequence length.
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
        self.encoder = Encoder(
            num_layers=num__encoder_layers,
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dim_feedfwd=dim_feedfwd,
            dropout=dropout_rate,
            use_bias=use_bias,
        )
        self.decoder = Decoder(
            num_layers=num__decoder_layers,
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dim_feedfwd=dim_feedfwd,
            dropout=dropout_rate,
            use_bias=use_bias,
        )
        self.pos_encod = PositionalEncoding(
            max__seq_length=seq_length,
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
        dict_input: Dict[str, Tensor],
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Args:
            dict_input: Dictionary with the keys "source" and "target",
                each containing a tensor (the tokens) of shape
                `(N, seq_length)`. The "source" tensor is the input to the
                encoder and the "target" tensor is the input to the decoder.
            pad_token_id: ID of the pad token.

        Returns:
            Output tensor of shape `(N, num_classes)`
        """

        # embedding and positional encoding for the encoder,
        # `(N, seq_length, embed_dim)`
        encoder_input = math.sqrt(self.embed_dim) * self.embedding(
            dict_input["source"]
        )
        encoder_input = self.pos_encod(encoder_input)
        encoder_input = self.dropout(encoder_input)

        # for the decoder, shift the output tokens to the right
        # (Sec. 3.4 of [1]), then embed and encode,
        # `(N, seq_length, embed_dim)`
        shifted__decoder_input = dict_input["target"].roll(
            shifts=(0, 1), dims=(0, 1)
        )
        shifted__decoder_input[:, 0] = pad_token_id
        shifted__decoder_input = math.sqrt(self.embed_dim) * self.embedding(
            shifted__decoder_input
        )
        shifted__decoder_input = self.pos_encod(shifted__decoder_input)
        shifted__decoder_input = self.dropout(shifted__decoder_input)

        # implement mask for the first self-attention mechanism of shape
        # `(seq_length, seq_length)`, also cf.
        # https://peterbloem.nl/blog/transformers
        mask = torch.tril(
            torch.ones(
                encoder_input.shape[1],
                encoder_input.shape[1],
                device=encoder_input.device,
            ),
            diagonal=0,
        )

        # forward pass through encoder, decoder and linear layer
        x = self.encoder(encoder_input, mask=None)
        x = self.decoder(shifted__decoder_input, x, mask=mask)
        x = self.pre_softmax_linear(x)  # `(N, seq_length, vocab_size)`

        return x
