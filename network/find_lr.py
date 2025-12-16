#!/usr/bin/env python
"""Learning rate finder script."""

import pytorch_lightning as pl

from config import get_default_config
from dataset import PuzzleDataModule
from model import PuzzleCNN


def main():
    """Run learning rate finder."""
    # Get config
    config = get_default_config()

    # Initialize data module
    data_module = PuzzleDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        piece_size=config["data"]["piece_size"],
        puzzle_size=config["data"]["puzzle_size"],
    )

    # Initialize model
    model = PuzzleCNN(
        backbone_name=config["model"]["backbone_name"],
        pretrained=config["model"]["pretrained"],
        learning_rate=config["model"]["learning_rate"],
        position_weight=config["model"]["position_weight"],
        rotation_weight=config["model"]["rotation_weight"],
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )

    # Run learning rate finder
    print("Running learning rate finder...")
    print("This will test learning rates from 1e-7 to 1.0")
    print()

    tuner = pl.tuner.Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        datamodule=data_module,
        min_lr=1e-7,
        max_lr=1.0,
        num_training=100,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Learning Rate Finder Results")
    print("=" * 60)
    print(f"Suggested learning rate: {lr_finder.suggestion()}")
    print()
    print("To visualize the full curve, run:")
    print("  fig = lr_finder.plot(suggest=True)")
    print("  fig.savefig('lr_finder_plot.png')")
    print()
    print(f"Current LR in config: {config['model']['learning_rate']}")
    print("=" * 60)

    # Optionally save the plot
    try:
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder_plot.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to: lr_finder_plot.png")
    except Exception as e:
        print(f"\nCouldn't save plot: {e}")


if __name__ == "__main__":
    main()
