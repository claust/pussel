# Experiment 7: Coarse Regression with Multi-Puzzle Training

"""Experiment 7: Coarse Regression with Multi-Puzzle Training.

This experiment tests whether cross-puzzle generalization is achievable
by simplifying the task to 2x2 quadrant prediction using coordinate regression.

Key components:
- model.py: DualInputRegressor with MobileNetV3-Small backbone
- dataset.py: QuadrantDataset for loading 2x2 pieces
- train.py: Training script with Phase 1 (frozen backbone)
- visualize.py: Visualization utilities
- generate_pieces.py: Piece generation script
"""

from .dataset import QUADRANT_CENTERS, QuadrantDataset, create_datasets, get_puzzle_ids
from .model import DualInputRegressor, count_parameters

__all__ = [
    "DualInputRegressor",
    "QuadrantDataset",
    "QUADRANT_CENTERS",
    "create_datasets",
    "get_puzzle_ids",
    "count_parameters",
]
