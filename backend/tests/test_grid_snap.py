"""Tests for snapping predicted piece positions onto the estimated grid."""

import pytest

from app.models.puzzle_model import PieceResponse, Position
from app.services.grid_snap import snap_to_grid


def make_response(x: float, y: float) -> PieceResponse:
    """Build a minimal PieceResponse with the given predicted center."""
    return PieceResponse(
        position=Position(x=x, y=y),
        position_confidence=0.8,
        rotation=0,
        rotation_confidence=0.9,
    )


def test_no_grid_leaves_response_unchanged() -> None:
    """Without a known grid, the response passes through with null snap fields."""
    response = snap_to_grid(make_response(0.3, 0.7), None)
    assert response.grid_row is None
    assert response.grid_col is None
    assert response.snapped_position is None


def test_interior_position_snaps_to_containing_cell_center() -> None:
    """A prediction inside the puzzle snaps to the center of its cell."""
    response = snap_to_grid(make_response(0.3, 0.7), (4, 6))
    assert (response.grid_row, response.grid_col) == (2, 1)
    assert response.snapped_position is not None
    assert response.snapped_position.x == pytest.approx(1.5 / 6)
    assert response.snapped_position.y == pytest.approx(2.5 / 4)
    # The raw prediction is kept as-is.
    assert (response.position.x, response.position.y) == (0.3, 0.7)


@pytest.mark.parametrize(
    ("x", "y", "expected_row", "expected_col"),
    [
        (-0.1, -0.2, 0, 0),  # beyond the top-left corner
        (1.1, 1.3, 3, 5),  # beyond the bottom-right corner
        (1.0, 1.0, 3, 5),  # exactly on the far edges
        (0.0, 0.0, 0, 0),  # exactly on the near edges
    ],
)
def test_out_of_range_positions_clamp_to_edge_cells(x: float, y: float, expected_row: int, expected_col: int) -> None:
    """Predictions outside [0, 1] snap to the nearest edge cell, never outside the puzzle."""
    response = snap_to_grid(make_response(x, y), (4, 6))
    assert (response.grid_row, response.grid_col) == (expected_row, expected_col)
    assert response.snapped_position is not None
    assert 0.0 < response.snapped_position.x < 1.0
    assert 0.0 < response.snapped_position.y < 1.0
