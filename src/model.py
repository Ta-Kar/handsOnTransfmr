"""Core model components for the Hands-on Transformer project.

We will implement this file incrementally. Do not fill every TODO at once:
start with token and positional embeddings, test them, and then add the next
component together.
"""

import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    """Convert token IDs into vectors that also carry position information."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        pad_idx: int,
    ) -> None:
        super().__init__()

        # TODO 1:
        # Create an nn.Embedding layer for token IDs.
        # Its vocabulary size is vocab_size and each vector has d_model values.
        #
        # TODO 2:
        # Create a second nn.Embedding layer for positions.
        # It should contain max_seq_len position vectors, each of size d_model.

        raise NotImplementedError("We will implement the embedding layers together.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return vectors for input_ids shaped (batch_size, sequence_length)."""

        # TODO 3:
        # Read batch_size and sequence_length from input_ids.shape.
        #
        # TODO 4:
        # Create position IDs: 0, 1, ..., sequence_length - 1.
        # Make sure they are on the same device as input_ids.
        #
        # TODO 5:
        # Look up token embeddings and position embeddings.
        # Add them and return the result.
        #
        # Expected output shape:
        # (batch_size, sequence_length, d_model)

        raise NotImplementedError("We will implement the forward pass together.")
