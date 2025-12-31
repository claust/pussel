"""Experiment 20: Realistic Puzzle Pieces with 4x4 Grid.

This experiment trains a model using realistically-shaped puzzle pieces
(Bezier curve tabs/blanks) instead of square-cut rectangles. Uses a 4x4
grid for position prediction.

Key changes from exp18:
- Realistic piece shapes using puzzle_shapes library
- 4x4 grid (16 cells) instead of 3x3 (9 cells)
- Pre-generated pieces with black background
- Filename encodes center coordinates: puzzle_id_x{cx}_y{cy}_rot{rot}.png
"""

from .dataset import (
    CELL_CENTERS,
    GRID_SIZE,
    NUM_CELLS,
    RealisticPieceDataset,
    RealisticPieceTestDataset,
    create_datasets,
    get_puzzle_ids,
)
from .model import FastBackboneModel, count_parameters

__all__ = [
    "GRID_SIZE",
    "NUM_CELLS",
    "CELL_CENTERS",
    "RealisticPieceDataset",
    "RealisticPieceTestDataset",
    "create_datasets",
    "get_puzzle_ids",
    "FastBackboneModel",
    "count_parameters",
]
