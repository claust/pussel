"""Snap predicted piece positions onto the puzzle's estimated grid.

The matchers predict a continuous piece center that can land anywhere —
including slightly outside the puzzle for edge and corner pieces. The client
displays pieces at real grid slots, so the piece endpoint attaches the nearest
cell and its center here; the raw prediction is kept untouched alongside.
"""

from typing import Optional, Tuple

from app.models.puzzle_model import PieceResponse, Position


def snap_to_grid(response: PieceResponse, grid: Optional[Tuple[int, int]]) -> PieceResponse:
    """Attach the nearest grid cell and its center to a piece prediction.

    Args:
        response: The matcher's prediction, with a continuous `position`.
        grid: The puzzle's estimated (rows, cols), or None when unknown.

    Returns:
        The response with `grid_row`, `grid_col`, and `snapped_position` set
        (a copy), or the response unchanged when no grid is known.
    """
    if grid is None:
        return response
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        return response
    # A center at exactly 1.0 (or a prediction outside [0, 1]) clamps to the
    # nearest edge cell, so edge/corner pieces always land inside the puzzle.
    row = min(max(int(response.position.y * rows), 0), rows - 1)
    col = min(max(int(response.position.x * cols), 0), cols - 1)
    return response.model_copy(
        update={
            "grid_row": row,
            "grid_col": col,
            "snapped_position": Position(x=(col + 0.5) / cols, y=(row + 0.5) / rows),
        }
    )
