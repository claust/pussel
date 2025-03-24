#!/usr/bin/env python
"""Training script for puzzle piece prediction model."""

import argparse
import os
from typing import Any, Dict

import pytorch_lightning as pl
from config import get_default_config
from dataset import PuzzleDataModule
from model import PuzzleCNN
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments and update default config.

    Returns:
        Updated configuration dictionary
    """
    # Get default configuration
    config = get_default_config()

    # Create argument parser
    parser = argparse.ArgumentParser(description="Train puzzle piece prediction model")

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default=config["model"]["backbone_name"],
        help="Name of the timm backbone to use",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=config["model"]["pretrained"],
        help="Whether to use pretrained weights",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config["model"]["learning_rate"],
        help="Initial learning rate",
    )
    parser.add_argument(
        "--position_weight",
        type=float,
        default=config["model"]["position_weight"],
        help="Weight for position loss (α)",
    )
    parser.add_argument(
        "--rotation_weight",
        type=float,
        default=config["model"]["rotation_weight"],
        help="Weight for rotation loss (β)",
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config["data"]["data_dir"],
        help="Directory containing puzzle data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["data"]["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config["data"]["num_workers"],
        help="Number of data loading workers",
    )

    # Training arguments
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=config["training"]["max_epochs"],
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=config["training"]["early_stop_patience"],
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=config["training"]["checkpoint_dir"],
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=config["training"]["log_dir"],
        help="Directory to save training logs",
    )

    # Parse arguments
    args = parser.parse_args()

    # Update configuration with parsed arguments
    config["model"]["backbone_name"] = args.backbone
    config["model"]["pretrained"] = args.pretrained
    config["model"]["learning_rate"] = args.learning_rate
    config["model"]["position_weight"] = args.position_weight
    config["model"]["rotation_weight"] = args.rotation_weight

    config["data"]["data_dir"] = args.data_dir
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers

    config["training"]["max_epochs"] = args.max_epochs
    config["training"]["early_stop_patience"] = args.early_stop_patience
    config["training"]["checkpoint_dir"] = args.checkpoint_dir
    config["training"]["log_dir"] = args.log_dir

    return config


def main():
    """Main training function."""
    # Parse command line arguments
    config = parse_args()

    # Create directories if they don't exist
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["training"]["log_dir"], exist_ok=True)

    # Initialize data module
    data_module = PuzzleDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        input_size=config["data"]["input_size"],
    )

    # Initialize model
    model = PuzzleCNN(
        backbone_name=config["model"]["backbone_name"],
        pretrained=config["model"]["pretrained"],
        learning_rate=config["model"]["learning_rate"],
        position_weight=config["model"]["position_weight"],
        rotation_weight=config["model"]["rotation_weight"],
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["training"]["checkpoint_dir"],
        filename="puzzle-cnn-{epoch:02d}-{val_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        mode="min",
        patience=config["training"]["early_stop_patience"],
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config["training"]["log_dir"],
        name="puzzle_cnn",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train model
    trainer.fit(model, datamodule=data_module)

    # Print path to best model checkpoint
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
