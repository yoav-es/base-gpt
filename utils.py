"""Data preparation and training helpers for character-level models."""

import logging
import math
import re
import time
from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

import torch
from torch import Tensor, nn

from model import (
    BigramLanguageModel,
    MultiHeadLanguageModel,
    SingleHeadLanguageModel,
)

logger = logging.getLogger(__name__)
_CLEANER = re.compile(r'[^a-zA-Z\s,.:;?_&$!"()\-\*\[\]]')


class RunConfig(TypedDict):
    """Settings loaded from config.json for one training run."""

    model: Literal["bigram", "single_head", "multi_head", "compare"]
    optimizer: Literal["adam", "adamw"]
    device: Literal["auto", "cpu", "cuda"]
    seed: int
    max_new_tokens: int
    train_file: str
    validation_file: str
    n_embd: int
    n_head: int
    batch_size: int
    block_size: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    eval_iters: int


LanguageModel = Union[
    BigramLanguageModel,
    SingleHeadLanguageModel,
    MultiHeadLanguageModel,
]
ModelName = Literal["bigram", "single_head", "multi_head"]


def clean_text(text: str) -> str:
    """Lowercase text and remove unsupported characters."""
    return _CLEANER.sub("", text.lower()).strip()


def _clean_lines(path: Path):
    """Yield cleaned lines while preserving their whitespace separators."""
    with path.open(encoding="utf-8") as text_file:
        for line in text_file:
            yield _CLEANER.sub("", line.lower())


def _encode_file(path: Path, stoi: dict[str, int]) -> bytearray:
    """Encode a corpus file as compact, one-byte character IDs."""
    encoded = bytearray()
    for line in _clean_lines(path):
        for character in line:
            try:
                encoded.append(stoi[character])
            except KeyError as error:
                raise ValueError(
                    f"{path} contains a character missing from the training "
                    f"vocabulary: {character!r}"
                ) from error

    return encoded


def prepare_data(
    train_path: Path, val_path: Path
) -> tuple[Tensor, Tensor, dict[str, int], dict[int, str]]:
    """Encode the corpus train and validation files using a shared vocabulary.

    The vocabulary is collected from the training file only. The files are read
    in passes so the large corpus does not need to exist as a Python list of
    integer token IDs in memory.
    """
    chars = set()
    for line in _clean_lines(train_path):
        chars.update(line)

    if not chars:
        raise ValueError(
            f"The training text in {train_path} is empty after cleaning."
        )
    if len(chars) > 256:
        raise ValueError("The character vocabulary must fit in one byte.")

    sorted_chars = sorted(chars)
    stoi = {character: index for index, character in enumerate(sorted_chars)}
    itos = {index: character for character, index in stoi.items()}

    train_buffer = _encode_file(train_path, stoi)
    val_buffer = _encode_file(val_path, stoi)
    if not train_buffer:
        raise ValueError(
            f"The training text in {train_path} is empty after cleaning."
        )
    if not val_buffer:
        raise ValueError(
            f"The validation text in {val_path} is empty after cleaning."
        )

    train_data = torch.frombuffer(train_buffer, dtype=torch.uint8)
    val_data = torch.frombuffer(val_buffer, dtype=torch.uint8)
    return train_data, val_data, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Convert text into vocabulary IDs; report unknown characters."""
    try:
        return [stoi[character] for character in text]
    except KeyError as error:
        raise ValueError(
            f"Character {error.args[0]!r} is not in the vocabulary."
        ) from error


def decode(tokens: list[int], itos: dict[int, str]) -> str:
    """Convert vocabulary IDs back into text."""
    return "".join(itos[token] for token in tokens)


def calculate_perplexity(mean_loss: float) -> float:
    """Convert mean cross-entropy loss to character-level perplexity."""
    return math.exp(mean_loss)


def create_model(
    config: RunConfig,
    vocab_size: int,
    model_name: Optional[ModelName] = None,
) -> LanguageModel:
    """Create the configured language model."""
    selected_model = model_name or config["model"]

    if selected_model == "bigram":
        return BigramLanguageModel(vocab_size)
    if selected_model == "single_head":
        return SingleHeadLanguageModel(
            vocab_size=vocab_size,
            n_embd=config["n_embd"],
            block_size=config["block_size"],
        )
    if selected_model == "multi_head":
        return MultiHeadLanguageModel(
            vocab_size=vocab_size,
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            block_size=config["block_size"],
        )

    raise ValueError(
        f"Unknown model {selected_model!r}. Choose 'bigram', 'single_head', "
        "or 'multi_head'."
    )


def create_optimizer(
    config: RunConfig, model: nn.Module
) -> torch.optim.Optimizer:
    """Create the configured optimizer for a model."""
    optimizer_name = config["optimizer"]
    learning_rate = config["learning_rate"]

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)

    raise ValueError(
        f"Unknown optimizer {optimizer_name!r}. Choose 'adam' or 'adamw'."
    )


def select_device(config: RunConfig) -> torch.device:
    """Choose the configured execution device, checking CUDA availability."""
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")

    return torch.device(device_name)


def get_batch(
    split: str,
    train_data: Tensor,
    val_data: Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample input sequences and their next-character targets."""
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError("split must be either 'train' or 'val'.")

    if len(data) <= block_size:
        raise ValueError(
            f"The {split} dataset needs more than {block_size} tokens "
            "to create a batch."
        )

    starts = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack(
        [data[start : start + block_size] for start in starts]
    )
    targets = torch.stack(
        [data[start + 1 : start + block_size + 1] for start in starts]
    )
    return inputs.to(device=device, dtype=torch.long), targets.to(
        device=device, dtype=torch.long
    )


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: Tensor,
    val_data: Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    """Estimate average loss without updating model weights."""
    was_training = model.training
    model.eval()
    losses = {}

    for split in ("train", "val"):
        split_losses = torch.zeros(eval_iters)
        for iteration in range(eval_iters):
            inputs, targets = get_batch(
                split, train_data, val_data, batch_size, block_size, device
            )
            _, loss = model(inputs, targets)
            split_losses[iteration] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train(was_training)
    return losses


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Tensor,
    val_data: Tensor,
    batch_size: int,
    block_size: int,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    device: torch.device,
) -> None:
    """Train the model and log periodic training and validation losses."""
    logger.info("Starting training for %s steps.", max_iters)
    for iteration in range(max_iters):
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                block_size,
                eval_iters,
                device,
            )
            train_perplexity = calculate_perplexity(losses["train"])
            validation_perplexity = calculate_perplexity(losses["val"])
            logger.info(
                "Step %d/%d | train loss: %.4f | train perplexity: %.4f | "
                "validation loss: %.4f | validation perplexity: %.4f",
                iteration,
                max_iters,
                losses["train"],
                train_perplexity,
                losses["val"],
                validation_perplexity,
            )

        inputs, targets = get_batch(
            "train", train_data, val_data, batch_size, block_size, device
        )
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    logger.info("Training complete.")
    logger.info("Training complete.")


def compare_models(
    config: RunConfig,
    vocab_size: int,
    train_data: Tensor,
    val_data: Tensor,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Train each model with shared settings and log validation comparisons."""
    results = {}
    model_names: tuple[ModelName, ...] = (
        "bigram",
        "single_head",
        "multi_head",
    )

    for model_name in model_names:
        torch.manual_seed(config["seed"])
        model = create_model(config, vocab_size, model_name).to(device)
        optimizer = create_optimizer(config, model)

        # Re-seed after model initialization so each model sees the same
        # sequence of randomly sampled training and evaluation batches.
        torch.manual_seed(config["seed"] + 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        logger.info("Comparing model: %s", model_name)
        train_model(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=val_data,
            batch_size=config["batch_size"],
            block_size=config["block_size"],
            max_iters=config["max_iters"],
            eval_interval=config["eval_interval"],
            eval_iters=config["eval_iters"],
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - start_time
        losses = estimate_loss(
            model=model,
            train_data=train_data,
            val_data=val_data,
            batch_size=config["batch_size"],
            block_size=config["block_size"],
            eval_iters=config["eval_iters"],
            device=device,
        )
        results[model_name] = {
            "train_loss": losses["train"],
            "validation_loss": losses["val"],
            "train_perplexity": calculate_perplexity(losses["train"]),
            "validation_perplexity": calculate_perplexity(losses["val"]),
            "training_seconds": training_seconds,
        }

    logger.info("Model comparison (validation metrics):")
    for model_name, metrics in results.items():
        logger.info(
            "%s | validation loss: %.4f | validation perplexity: %.4f "
            "| training time: %.2fs",
            model_name,
            metrics["validation_loss"],
            metrics["validation_perplexity"],
            metrics["training_seconds"],
        )

    return results


def train_and_generate(
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    train_data: Tensor,
    val_data: Tensor,
    itos: dict[int, str],
    config: RunConfig,
    device: torch.device,
) -> str:
    """Train a model and return a decoded sample generated from it."""
    train_model(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        batch_size=config["batch_size"],
        block_size=config["block_size"],
        max_iters=config["max_iters"],
        eval_interval=config["eval_interval"],
        eval_iters=config["eval_iters"],
        device=device,
    )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(
        context, max_new_tokens=config["max_new_tokens"]
    )[0].tolist()
    return decode(generated, itos)
