"""Causal self-attention components."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """A single causal self-attention head."""

    tril: torch.Tensor

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        """Create query, key, and value projections and a causal mask."""
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix positions using current and prior positions."""
        _, sequence_length, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        weights = query @ key.transpose(-2, -1)
        weights = weights * key.shape[-1] ** -0.5
        weights = weights.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf"),
        )
        weights = F.softmax(weights, dim=-1)

        return weights @ value


class MultiHeadAttention(nn.Module):
    """Run several causal self-attention heads in parallel."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
    ) -> None:
        """Create parallel heads whose concatenated output has size n_embd."""
        super().__init__()
        if n_head <= 0:
            raise ValueError("n_head must be a positive integer.")
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                )
                for _ in range(n_head)
            ]
        )
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate head outputs and project them back to embedding size."""
        attended = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(attended)
