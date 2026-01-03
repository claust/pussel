"""Experiment 21: Masked Rotation Correlation.

This experiment investigates whether using masks to ignore black background
regions in piece images improves rotation prediction accuracy. In exp20,
rotation correlation achieved 95% on training but only 25% (random) on test.

Key changes from exp20:
- Dataset returns masks alongside piece images
- Rotation correlation module uses masks when comparing features
- Masks are derived from black (0,0,0) background detection
"""

from .dataset import (
    CELL_CENTERS,
    GRID_SIZE,
    NUM_CELLS,
    MaskedPieceDataset,
    MaskedPieceTestDataset,
    create_datasets,
    generate_mask,
    get_puzzle_ids,
)
from .model import MaskedRotationModel, count_parameters

__all__ = [
    "GRID_SIZE",
    "NUM_CELLS",
    "CELL_CENTERS",
    "MaskedPieceDataset",
    "MaskedPieceTestDataset",
    "create_datasets",
    "generate_mask",
    "get_puzzle_ids",
    "MaskedRotationModel",
    "count_parameters",
]
