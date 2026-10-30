"""Shared fixtures for the language-model test suite."""

import pytest

from utils import RunConfig


@pytest.fixture
def tiny_config() -> RunConfig:
    """Return a small CPU configuration suitable for quick model checks."""
    return {
        "model": "multi_head",
        "optimizer": "adam",
        "device": "cpu",
        "seed": 1337,
        "max_new_tokens": 4,
        "train_file": "train.txt",
        "validation_file": "validation.txt",
        "n_embd": 8,
        "n_head": 2,
        "batch_size": 2,
        "block_size": 4,
        "learning_rate": 0.001,
        "max_iters": 2,
        "eval_interval": 1,
        "eval_iters": 1,
    }
