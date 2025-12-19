#!/usr/bin/env python
"""Learning rate finder script."""

import argparse

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

from config import get_default_config
from dataset import PuzzleDataModule
from model import PuzzleCNN


def main():
    """Run learning rate finder."""
    # Get config
    config = get_default_config()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Find optimal learning rate")
    parser.add_argument(
        "--backbone",
        type=str,
        default=config["model"]["backbone_name"],
        help="Backbone model name (e.g., mobilenetv3_small_100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["data"]["batch_size"],
        help="Batch size for LR finder",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config["data"]["num_workers"],
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--num_training",
        type=int,
        default=100,
        help="Number of training steps for LR finder",
    )
    args = parser.parse_args()

    # Override config with CLI args
    backbone = args.backbone
    batch_size = args.batch_size
    num_workers = args.num_workers

    print(f"Running LR finder with backbone: {backbone}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")

    # Initialize data module
    data_module = PuzzleDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=batch_size,
        num_workers=num_workers,
        piece_size=config["data"]["piece_size"],
        puzzle_size=config["data"]["puzzle_size"],
    )

    # Initialize model
    model = PuzzleCNN(
        backbone_name=backbone,
        pretrained=config["model"]["pretrained"],
        learning_rate=config["model"]["learning_rate"],
        position_weight=config["model"]["position_weight"],
        rotation_weight=config["model"]["rotation_weight"],
        use_spatial_correlation=config["model"]["use_spatial_correlation"],
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

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        datamodule=data_module,
        min_lr=1e-7,
        max_lr=1.0,
        num_training=args.num_training,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Learning Rate Finder Results")
    print("=" * 60)
    suggested_lr = lr_finder.suggestion() if lr_finder else None
    print(f"Suggested learning rate: {suggested_lr}")
    print()
    print("To visualize the full curve, run:")
    print("  fig = lr_finder.plot(suggest=True)")
    print("  fig.savefig('lr_finder_plot.png')")
    print()
    print(f"Backbone tested: {backbone}")
    print(f"Current LR in config: {config['model']['learning_rate']}")
    print("=" * 60)

    # Optionally save the plot
    try:
        import matplotlib.pyplot as plt

        fig = lr_finder.plot(suggest=True) if lr_finder else None
        if fig is not None:
            plt.title(f"LR Finder - {backbone}")
            fig.savefig("lr_finder_plot.png", dpi=150, bbox_inches="tight")
            print("\nPlot saved to: lr_finder_plot.png")
        else:
            print("\nCouldn't generate plot: lr_finder returned None")
    except Exception as e:
        print(f"\nCouldn't save plot: {e}")


if __name__ == "__main__":
    main()
