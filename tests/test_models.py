"""Unit tests for the language models and attention components."""

import pytest
import torch

from head import Head, MultiHeadAttention
from model import (
    BigramLanguageModel,
    MultiHeadLanguageModel,
    SingleHeadLanguageModel,
)


@pytest.mark.parametrize(
    "model",
    [
        BigramLanguageModel(vocab_size=7),
        SingleHeadLanguageModel(vocab_size=7, n_embd=8, block_size=6),
        MultiHeadLanguageModel(
            vocab_size=7,
            n_embd=8,
            n_head=2,
            block_size=6,
        ),
    ],
)
def test_models_return_logits_loss_and_backpropagate(model):
    inputs = torch.randint(0, 7, (2, 6))
    targets = torch.randint(0, 7, (2, 6))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 6, 7)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    "model",
    [
        BigramLanguageModel(vocab_size=7),
        SingleHeadLanguageModel(vocab_size=7, n_embd=8, block_size=6),
        MultiHeadLanguageModel(
            vocab_size=7,
            n_embd=8,
            n_head=2,
            block_size=6,
        ),
    ],
)
def test_models_generate_requested_number_of_tokens(model):
    initial_tokens = torch.zeros((1, 1), dtype=torch.long)

    generated = model.generate(initial_tokens, max_new_tokens=9)

    assert generated.shape == (1, 10)


@pytest.mark.parametrize(
    "model",
    [
        SingleHeadLanguageModel(vocab_size=7, n_embd=8, block_size=4),
        MultiHeadLanguageModel(
            vocab_size=7,
            n_embd=8,
            n_head=2,
            block_size=4,
        ),
    ],
)
def test_attention_models_reject_inputs_longer_than_block_size(model):
    with pytest.raises(ValueError, match="exceeds block size"):
        model(torch.zeros((1, 5), dtype=torch.long))


def test_single_attention_head_preserves_expected_shape():
    head = Head(n_embd=8, head_size=4, block_size=6)

    assert head(torch.randn(2, 6, 8)).shape == (2, 6, 4)


def test_multihead_attention_preserves_embedding_width():
    attention = MultiHeadAttention(n_embd=12, n_head=3, block_size=6)

    assert attention(torch.randn(2, 6, 12)).shape == (2, 6, 12)


def test_attention_does_not_use_future_positions():
    torch.manual_seed(12)
    attention = MultiHeadAttention(n_embd=8, n_head=2, block_size=6)
    original = torch.randn(1, 6, 8)
    changed_future = original.clone()
    changed_future[:, 4:] = torch.randn(1, 2, 8)

    with torch.no_grad():
        original_output = attention(original)
        changed_output = attention(changed_future)

    assert torch.allclose(
        original_output[:, :4],
        changed_output[:, :4],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("n_embd", "n_head", "message"),
    [
        (8, 0, "positive"),
        (7, 2, "divisible"),
    ],
)
def test_multihead_attention_validates_head_configuration(
    n_embd,
    n_head,
    message,
):
    with pytest.raises(ValueError, match=message):
        MultiHeadAttention(n_embd=n_embd, n_head=n_head, block_size=4)
