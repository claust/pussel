#!/usr/bin/env python
"""Training script for puzzle piece prediction model."""

import argparse
import os
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_default_config
from dataset import PuzzleDataModule
from model import PuzzleCNN


def str_to_bool(value: str) -> bool:
    """Convert string to boolean for argparse.

    Args:
        value: String representation of boolean

    Returns:
        Boolean value

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed as boolean
    """
    if value.lower() in ("true", "1", "yes"):
        return True
    elif value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


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
    parser.add_argument(
        "--use_spatial_correlation",
        type=str_to_bool,
        default=config["model"]["use_spatial_correlation"],
        help="Use spatial correlation module (true/false)",
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
    parser.add_argument(
        "--piece_size",
        type=int,
        nargs=2,
        default=config["data"]["piece_size"],
        help="Size of piece images (height width)",
    )
    parser.add_argument(
        "--puzzle_size",
        type=int,
        nargs=2,
        default=config["data"]["puzzle_size"],
        help="Size of puzzle images (height width)",
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
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=config["training"]["experiment_name"],
        help="Name for the experiment run",
    )

    # Parse arguments
    args = parser.parse_args()

    # Update configuration with parsed arguments
    config["model"]["backbone_name"] = args.backbone
    config["model"]["pretrained"] = args.pretrained
    config["model"]["learning_rate"] = args.learning_rate
    config["model"]["position_weight"] = args.position_weight
    config["model"]["rotation_weight"] = args.rotation_weight
    config["model"]["use_spatial_correlation"] = args.use_spatial_correlation

    config["data"]["data_dir"] = args.data_dir
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["data"]["piece_size"] = tuple(args.piece_size)
    config["data"]["puzzle_size"] = tuple(args.puzzle_size)

    config["training"]["max_epochs"] = args.max_epochs
    config["training"]["early_stop_patience"] = args.early_stop_patience
    config["training"]["checkpoint_dir"] = args.checkpoint_dir
    config["training"]["log_dir"] = args.log_dir
    config["training"]["experiment_name"] = args.experiment_name

    return config


def main():
    """Main training function."""
    # Parse command line arguments
    config = parse_args()

    # Create directories if they don't exist
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["training"]["log_dir"], exist_ok=True)

    # Initialize data module for dual-input model
    data_module = PuzzleDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        piece_size=config["data"]["piece_size"],
        puzzle_size=config["data"]["puzzle_size"],
    )

    # Initialize dual-input model with spatial correlation
    model = PuzzleCNN(
        backbone_name=config["model"]["backbone_name"],
        pretrained=config["model"]["pretrained"],
        learning_rate=config["model"]["learning_rate"],
        position_weight=config["model"]["position_weight"],
        rotation_weight=config["model"]["rotation_weight"],
        use_spatial_correlation=config["model"]["use_spatial_correlation"],
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["training"]["checkpoint_dir"],
        filename="puzzle-dual-{epoch:02d}-{val/total_loss:.4f}",
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
        name=config["training"]["experiment_name"],
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
