"""Integration and bounded stress tests for the training workflow."""

import json
import logging

import pytest
import torch

import main as main_module
from utils import (
    create_model,
    create_optimizer,
    get_batch,
    prepare_data,
    select_device,
    train_and_generate,
)


@pytest.mark.parametrize("model_name", ["bigram", "single_head", "multi_head"])
def test_tiny_corpus_train_generate_integration(
    tmp_path,
    tiny_config,
    model_name,
):
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text(("abcde fghij. " * 20), encoding="utf-8")
    validation_path.write_text(("abcde fghij. " * 8), encoding="utf-8")
    tiny_config["model"] = model_name

    train_data, validation_data, _, itos = prepare_data(
        train_path,
        validation_path,
    )
    device = select_device(tiny_config)
    model = create_model(tiny_config, vocab_size=len(itos)).to(device)
    optimizer = create_optimizer(tiny_config, model)

    generated = train_and_generate(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=validation_data,
        itos=itos,
        config=tiny_config,
        device=device,
    )

    assert len(generated) == tiny_config["max_new_tokens"] + 1
    assert set(generated).issubset(itos.values())


def test_main_loads_config_and_runs_tiny_project(
    tmp_path,
    tiny_config,
    monkeypatch,
    caplog,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / tiny_config["train_file"]).write_text(
        "abcde fghij. " * 20,
        encoding="utf-8",
    )
    (data_dir / tiny_config["validation_file"]).write_text(
        "abcde fghij. " * 8,
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(tiny_config),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        main_module,
        "__file__",
        str(tmp_path / "main.py"),
    )

    with caplog.at_level(logging.INFO):
        main_module.main()

    assert "Data ready" in caplog.text
    assert "Training multi_head with adam" in caplog.text
    assert "Training complete" in caplog.text
    assert "Generated text" in caplog.text


def test_main_runs_model_comparison_when_configured(
    tmp_path,
    tiny_config,
    monkeypatch,
    caplog,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / tiny_config["train_file"]).write_text(
        "abcde fghij. " * 20,
        encoding="utf-8",
    )
    (data_dir / tiny_config["validation_file"]).write_text(
        "abcde fghij. " * 8,
        encoding="utf-8",
    )
    tiny_config["model"] = "compare"
    tiny_config["max_iters"] = 1
    (tmp_path / "config.json").write_text(
        json.dumps(tiny_config),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        main_module,
        "__file__",
        str(tmp_path / "main.py"),
    )

    with caplog.at_level(logging.INFO):
        main_module.main()

    assert "bigram | validation loss" in caplog.text
    assert "single_head | validation loss" in caplog.text
    assert "multi_head | validation loss" in caplog.text


def test_stress_prepare_data_batches_and_attention_near_block_size(tmp_path):
    train_path = tmp_path / "large_train.txt"
    validation_path = tmp_path / "large_validation.txt"
    train_path.write_text("abc def. " * 12_000, encoding="utf-8")
    validation_path.write_text("abc def. " * 2_000, encoding="utf-8")

    train_data, validation_data, stoi, _ = prepare_data(
        train_path,
        validation_path,
    )
    block_size = 128
    batch_inputs, batch_targets = get_batch(
        "train",
        train_data,
        validation_data,
        batch_size=8,
        block_size=block_size,
        device=torch.device("cpu"),
    )
    model = create_model(
        {
            "model": "multi_head",
            "optimizer": "adam",
            "device": "cpu",
            "seed": 1,
            "max_new_tokens": 16,
            "train_file": "train.txt",
            "validation_file": "validation.txt",
            "n_embd": 32,
            "n_head": 4,
            "batch_size": 8,
            "block_size": block_size,
            "learning_rate": 0.001,
            "max_iters": 1,
            "eval_interval": 1,
            "eval_iters": 1,
        },
        vocab_size=len(stoi),
    )

    logits, loss = model(batch_inputs, batch_targets)
    generated = model.generate(
        torch.zeros((1, 1), dtype=torch.long),
        max_new_tokens=block_size + 8,
    )

    assert len(train_data) > 100_000
    assert logits.shape == (8, block_size, len(stoi))
    assert torch.isfinite(loss)
    assert generated.shape == (1, block_size + 9)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_multihead_model_runs_on_cuda():
    device = torch.device("cuda")
    model = create_model(
        {
            "model": "multi_head",
            "optimizer": "adam",
            "device": "cuda",
            "seed": 1,
            "max_new_tokens": 4,
            "train_file": "train.txt",
            "validation_file": "validation.txt",
            "n_embd": 8,
            "n_head": 2,
            "batch_size": 2,
            "block_size": 4,
            "learning_rate": 0.001,
            "max_iters": 1,
            "eval_interval": 1,
            "eval_iters": 1,
        },
        vocab_size=5,
    ).to(device)

    logits, loss = model(
        torch.randint(0, 5, (2, 4), device=device),
        torch.randint(0, 5, (2, 4), device=device),
    )

    assert logits.device.type == "cuda"
    assert loss is not None and torch.isfinite(loss)
