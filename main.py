"""Train the configured character-level language model and sample text."""

import json
import logging
from pathlib import Path

import torch

from utils import (
    RunConfig,
    compare_models,
    create_model,
    create_optimizer,
    prepare_data,
    select_device,
    train_and_generate,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Load project data and settings, train the model, and generate text."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    project_dir = Path(__file__).resolve().parent
    with (project_dir / "config.json").open(encoding="utf-8") as config_file:
        config: RunConfig = json.load(config_file)

    data_dir = project_dir / "data"
    logger.info("Loading training and validation data.")
    train_data, val_data, _, itos = prepare_data(
        data_dir / config["train_file"],
        data_dir / config["validation_file"],
    )
    device = select_device(config)
    torch.manual_seed(config["seed"])
    logger.info(
        "Data ready: %d training tokens, %d validation tokens, "
        "%d unique characters. Device: %s.",
        len(train_data),
        len(val_data),
        len(itos),
        device,
    )
    logger.info(
        "PyTorch %s | CUDA build: %s",
        torch.__version__,
        torch.version.cuda or "not available",
    )
    if device.type == "cuda":
        logger.info("Using NVIDIA GPU: %s", torch.cuda.get_device_name(device))

    if config["model"] == "compare":
        compare_models(
            config=config,
            vocab_size=len(itos),
            train_data=train_data,
            val_data=val_data,
            device=device,
        )
        return

    model = create_model(config, vocab_size=len(itos)).to(device)
    logger.info(
        "Training %s with %s.",
        config["model"],
        config["optimizer"],
    )
    optimizer = create_optimizer(config, model)
    generated_text = train_and_generate(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        itos=itos,
        config=config,
        device=device,
    )
    logger.info("Generated text:\n%s", generated_text)


if __name__ == "__main__":
    main()
