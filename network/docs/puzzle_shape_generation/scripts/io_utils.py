"""I/O utilities for puzzle piece configurations."""

import json
from pathlib import Path
from typing import List

from models import PieceConfig


def load_pieces_from_json(json_path: str | Path) -> List[PieceConfig]:
    """Load piece configurations from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    pieces = []
    for piece_data in data.get("pieces", []):
        pieces.append(PieceConfig.from_dict(piece_data))

    return pieces


def save_pieces_to_json(pieces: List[PieceConfig], json_path: str | Path) -> None:
    """Save piece configurations to a JSON file."""
    data = {"pieces": [p.to_dict() for p in pieces]}

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
