"""PyTorch language models for character-level text generation."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from head import Head, MultiHeadAttention


class BigramLanguageModel(nn.Module):
    """Predict each next character using a learned token-to-token table."""

    def __init__(self, vocab_size):
        """Create one next-token distribution per token."""
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Return next-token logits and, when provided, cross-entropy loss."""
        logits = self.token_embedding_table(idx)
        loss = None

        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_length, vocab_size),
                targets.view(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Append sampled next characters to the given token sequence."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


class SingleHeadLanguageModel(nn.Module):
    """Character-level language model using one causal attention head."""

    def __init__(self, vocab_size: int, n_embd: int, block_size: int) -> None:
        """Create token/position embeddings, one attention head, and output."""
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.attention_head = Head(
            n_embd=n_embd,
            head_size=n_embd,
            block_size=block_size,
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return next-character logits and optional cross-entropy loss."""
        _, sequence_length = idx.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Input length {sequence_length} exceeds block size "
                f"{self.block_size}."
            )

        positions = torch.arange(sequence_length, device=idx.device)
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(positions)
        hidden = token_embeddings + position_embeddings
        hidden = self.attention_head(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_length, vocab_size),
                targets.view(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate characters, limiting attention context to block size."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


class MultiHeadLanguageModel(nn.Module):
    """Character-level language model using multiple causal attention heads."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        block_size: int,
    ) -> None:
        """Create embeddings, multi-head attention, and character output."""
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return next-character logits and optional cross-entropy loss."""
        _, sequence_length = idx.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Input length {sequence_length} exceeds block size "
                f"{self.block_size}."
            )

        positions = torch.arange(sequence_length, device=idx.device)
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(positions)
        hidden = self.attention(token_embeddings + position_embeddings)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_length, vocab_size),
                targets.view(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate characters using a context no longer than block_size."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
