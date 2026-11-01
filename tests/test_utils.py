"""Unit tests for data, configuration, training, and device helpers."""

import logging

import pytest
import torch

from model import (
    BigramLanguageModel,
    MultiHeadLanguageModel,
    SingleHeadLanguageModel,
)
from utils import (
    calculate_perplexity,
    clean_text,
    compare_models,
    create_model,
    create_optimizer,
    decode,
    encode,
    estimate_loss,
    get_batch,
    prepare_data,
    select_device,
    train_model,
)


def test_cleaning_lowercases_and_removes_unsupported_characters():
    assert clean_text("Hello 123! WORLD.") == "hello ! world."


def test_prepare_data_builds_reversible_character_vocabulary(tmp_path):
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text("Abca\n", encoding="utf-8")
    validation_path.write_text("caba\n", encoding="utf-8")

    train, validation, stoi, itos = prepare_data(
        train_path,
        validation_path,
    )

    assert train.dtype == torch.uint8
    assert validation.dtype == torch.uint8
    assert decode([int(token) for token in train], itos) == "abca\n"
    assert decode([int(token) for token in validation], itos) == "caba\n"
    assert {itos[index]: index for index in itos} == stoi


@pytest.mark.parametrize("existing_files", [(), ("train.txt",)])
def test_prepare_data_reports_missing_files(tmp_path, existing_files):
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    for filename in existing_files:
        (tmp_path / filename).write_text("abc", encoding="utf-8")

    with pytest.raises(
        FileNotFoundError, match="Required data file"
    ) as error:
        prepare_data(train_path, validation_path)
    for filename in {"train.txt", "validation.txt"} - set(existing_files):
        assert filename in str(error.value)


@pytest.mark.parametrize(
    ("train_text", "validation_text", "message"),
    [
        ("123", "", "training text"),
        ("abc", "z", "missing from the training vocabulary"),
    ],
)
def test_prepare_data_rejects_empty_or_unknown_text(
    tmp_path,
    train_text,
    validation_text,
    message,
):
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text(train_text, encoding="utf-8")
    validation_path.write_text(validation_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_data(train_path, validation_path)


def test_encode_and_decode_round_trip():
    stoi = {"a": 0, "b": 1, " ": 2}
    itos = {index: character for character, index in stoi.items()}

    assert decode(encode("ab ba", stoi), itos) == "ab ba"


def test_encode_reports_unknown_character():
    with pytest.raises(ValueError, match="not in the vocabulary"):
        encode("z", {"a": 0})


def test_calculate_perplexity_is_exponential_of_mean_loss():
    import math

    assert calculate_perplexity(1.5) == pytest.approx(math.exp(1.5))


@pytest.mark.parametrize("split", ["train", "val"])
def test_get_batch_shapes_and_next_character_targets(split):
    data = torch.arange(64, dtype=torch.uint8)
    train_data, val_data = data, data

    inputs, targets = get_batch(
        split,
        train_data,
        val_data,
        batch_size=3,
        block_size=8,
        device=torch.device("cpu"),
    )

    assert inputs.shape == targets.shape == (3, 8)
    assert inputs.dtype == targets.dtype == torch.long
    assert torch.equal(targets, inputs + 1)


def test_get_batch_rejects_invalid_split_and_short_data():
    data = torch.arange(4, dtype=torch.uint8)
    arguments = (
        data,
        data,
        2,
        4,
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="split must be"):
        get_batch("test", *arguments)
    with pytest.raises(ValueError, match="needs more than"):
        get_batch("train", *arguments)


@pytest.mark.parametrize(
    ("model_name", "model_type"),
    [
        ("bigram", BigramLanguageModel),
        ("single_head", SingleHeadLanguageModel),
        ("multi_head", MultiHeadLanguageModel),
    ],
)
def test_model_factory_selects_requested_model(
    tiny_config,
    model_name,
    model_type,
):
    tiny_config["model"] = model_name

    assert isinstance(create_model(tiny_config, vocab_size=5), model_type)


def test_model_factory_rejects_unknown_model(tiny_config):
    tiny_config["model"] = "transformer"

    with pytest.raises(ValueError, match="Unknown model"):
        create_model(tiny_config, vocab_size=5)


@pytest.mark.parametrize(
    ("optimizer_name", "optimizer_type"),
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
    ],
)
def test_optimizer_factory_selects_requested_optimizer(
    tiny_config,
    optimizer_name,
    optimizer_type,
):
    tiny_config["optimizer"] = optimizer_name
    model = BigramLanguageModel(vocab_size=5)

    optimizer = create_optimizer(tiny_config, model)

    assert isinstance(optimizer, optimizer_type)
    assert optimizer.param_groups[0]["lr"] == tiny_config["learning_rate"]


def test_optimizer_factory_rejects_unknown_optimizer(tiny_config):
    tiny_config["optimizer"] = "sgd"

    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer(tiny_config, BigramLanguageModel(vocab_size=5))


def test_select_device_auto_uses_available_device(tiny_config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    tiny_config["device"] = "auto"

    assert select_device(tiny_config) == torch.device("cuda")


def test_select_device_rejects_unavailable_cuda(tiny_config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tiny_config["device"] = "cuda"

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device(tiny_config)


@pytest.mark.parametrize("was_training", [True, False])
def test_estimate_loss_returns_both_splits_and_restores_mode(was_training):
    train_data = torch.arange(64, dtype=torch.uint8)
    validation_data = torch.arange(64, dtype=torch.uint8)
    model = BigramLanguageModel(vocab_size=64)
    model.train(was_training)

    losses = estimate_loss(
        model=model,
        train_data=train_data,
        val_data=validation_data,
        batch_size=2,
        block_size=4,
        eval_iters=2,
        device=torch.device("cpu"),
    )

    assert set(losses) == {"train", "val"}
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses.values())
    assert model.training is was_training


def test_train_model_updates_parameters_and_logs_progress(caplog):
    train_data = torch.arange(64, dtype=torch.uint8)
    validation_data = torch.arange(64, dtype=torch.uint8)
    model = BigramLanguageModel(vocab_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    with caplog.at_level(logging.INFO, logger="utils"):
        train_model(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=validation_data,
            batch_size=2,
            block_size=4,
            max_iters=2,
            eval_interval=1,
            eval_iters=1,
            device=torch.device("cpu"),
        )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, model.parameters())
    )
    assert "train loss" in caplog.text
    assert "train perplexity" in caplog.text
    assert "validation loss" in caplog.text
    assert "validation perplexity" in caplog.text
    assert "Training complete" in caplog.text


def test_compare_models_trains_and_reports_all_models(tiny_config, caplog):
    tiny_config["max_iters"] = 1
    train_data = torch.tensor([0, 1, 2, 3, 4] * 8, dtype=torch.uint8)
    validation_data = torch.tensor([0, 1, 2, 3, 4] * 4, dtype=torch.uint8)

    with caplog.at_level(logging.INFO, logger="utils"):
        results = compare_models(
            config=tiny_config,
            vocab_size=5,
            train_data=train_data,
            val_data=validation_data,
            device=torch.device("cpu"),
        )

    assert set(results) == {"bigram", "single_head", "multi_head"}
    for metrics in results.values():
        assert set(metrics) == {
            "train_loss",
            "validation_loss",
            "train_perplexity",
            "validation_perplexity",
            "training_seconds",
        }
        assert metrics["training_seconds"] >= 0
        assert metrics["validation_perplexity"] == pytest.approx(
            calculate_perplexity(metrics["validation_loss"])
        )
    assert "Model comparison" in caplog.text
