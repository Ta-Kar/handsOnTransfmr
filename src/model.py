"""Small encoder-only Transformer for sentence classification.

The implementation is intentionally explicit: attention, feed-forward layers,
residual connections, normalization, embeddings, and pooling are all visible.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with a mask for padding tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process x shaped (batch, sequence_length, d_model).

        attention_mask uses 1/True for real tokens and 0/False for padding.
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split the feature dimension into independent attention heads.
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Each query is compared with every key in the same sequence.
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "attention_mask must have shape (batch_size, sequence_length)"
                )

            valid_keys = attention_mask.to(device=x.device, dtype=torch.bool)
            padding_positions = ~valid_keys[:, None, None, :]
            scores = scores.masked_fill(
                padding_positions,
                torch.finfo(scores.dtype).min,
            )

        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        context = weights @ v

        # Join the heads back into the original feature dimension.
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        return self.output_dropout(self.out_proj(context))


class FeedForward(nn.Module):
    """Position-wise two-layer feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerEncoderBlock(nn.Module):
    """One encoder block with pre-LayerNorm and residual connections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN: normalize before each sublayer, then add a residual.
        attention_input = self.attention_norm(x)
        x = x + self.residual_dropout(
            self.attention(attention_input, attention_mask)
        )

        feed_forward_input = self.feed_forward_norm(x)
        x = x + self.residual_dropout(self.feed_forward(feed_forward_input))
        return x


class TransformerClassifier(nn.Module):
    """Encoder-only Transformer that predicts a class for each sequence."""

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        max_seq_len: int = 128,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= pad_idx < vocab_size:
            raise ValueError("pad_idx must be a valid vocabulary index")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )
        # Learned positional embeddings: one vector for each sequence position.
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits with shape (batch_size, num_classes)."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, sequence_length)")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        if attention_mask is None:
            attention_mask = input_ids != self.pad_idx
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )

        if not torch.all(attention_mask.any(dim=1)):
            raise ValueError("Each sequence must contain at least one non-padding token")

        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.final_norm(x)

        # Masked mean pooling: padding positions do not affect the sentence vector.
        mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        summed_tokens = (x * mask).sum(dim=1)
        number_of_tokens = mask.sum(dim=1).clamp_min(1.0)
        sentence_vectors = summed_tokens / number_of_tokens

        return self.classifier(sentence_vectors)


def count_parameters(model: nn.Module) -> int:
    """Count parameters that participate in gradient-based training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
