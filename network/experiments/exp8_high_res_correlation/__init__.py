# Experiment 8: High Resolution Coarse Regression (512x512 puzzle)

"""Experiment 8: High Resolution Coarse Regression.

This experiment tests whether increasing the puzzle resolution from 256x256
to 512x512 improves cross-puzzle generalization for 2x2 quadrant prediction.

Building on exp7's success with spatial correlation (67% test accuracy),
this experiment provides more detail for template matching.

Key components:
- model.py: DualInputRegressorWithCorrelation with MobileNetV3-Small backbone
- dataset.py: QuadrantDataset for loading 2x2 pieces
- train.py: Training script with Phase 1 (frozen backbone)
- visualize.py: Visualization utilities
- generate_pieces.py: Piece generation script
"""

from .dataset import QUADRANT_CENTERS, QuadrantDataset, create_datasets, get_puzzle_ids
from .model import DualInputRegressorWithCorrelation, count_parameters

__all__ = [
    "DualInputRegressorWithCorrelation",
    "QuadrantDataset",
    "QUADRANT_CENTERS",
    "create_datasets",
    "get_puzzle_ids",
    "count_parameters",
]
