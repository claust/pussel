"""In-memory puzzle store: owner-scoped records with FIFO eviction.

Puzzles are stored purely in memory — a deliberate decision, not a stopgap;
this backend intentionally has no database. Each `puzzle_id` maps to a
`PuzzleRecord` recording who owns it, where its image lives on disk, and its
optional estimated grid. `MAX_PUZZLES` caps memory growth: once the cap is
exceeded, the oldest puzzle is evicted (FIFO) and its resources (the stored
file, matcher caches, and piece-geometry entries) are cleaned up so nothing
leaks.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

# Cap on concurrently-tracked puzzles. The dict would otherwise grow without
# bound; once this is exceeded, the oldest puzzle (by insertion order) is
# evicted to make room for the new one.
MAX_PUZZLES = 100


@dataclass(frozen=True)
class PuzzleRecord:
    """One uploaded puzzle: who owns it, where its image lives, and its grid.

    Attributes:
        puzzle_id: The puzzle's id.
        owner_id: The uploading user's id (Google `sub`).
        file_path: Path to the puzzle's stored image on disk.
        grid: Estimated (rows, cols), when a `piece_count` was supplied at
            upload time; None otherwise.
    """

    puzzle_id: str
    owner_id: str
    file_path: str
    grid: Optional[Tuple[int, int]]


def _evict(record: PuzzleRecord) -> None:
    """Clean up an evicted puzzle's resources: its file, matcher caches, and geometry store.

    Imports are local to this function (rather than module-level) so that
    importing `puzzle_store` doesn't eagerly pull in the classical matcher,
    the CNN image processor, or the piece-geometry store — and to avoid an
    import cycle, since those modules may in turn end up depending on this
    one.

    Args:
        record: The evicted puzzle's record.
    """
    import os

    from app.services.classical_matcher import get_classical_matcher
    from app.services.image_processor import get_image_processor
    from app.services.piece_geometry.store import get_piece_geometry_store

    try:
        os.remove(record.file_path)
    except OSError:
        pass

    get_classical_matcher().clear_puzzle_cache(record.puzzle_id)
    get_image_processor().clear_puzzle_cache(record.puzzle_id)
    get_piece_geometry_store().drop(record.puzzle_id)


class PuzzleStore:
    """In-memory puzzle records keyed by puzzle_id, with FIFO eviction at `MAX_PUZZLES`."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._records: "OrderedDict[str, PuzzleRecord]" = OrderedDict()

    def add(self, record: PuzzleRecord) -> None:
        """Add a new puzzle record, evicting the oldest one if now over capacity.

        Args:
            record: The puzzle record to add.
        """
        self._records[record.puzzle_id] = record
        if len(self._records) > MAX_PUZZLES:
            _, oldest = self._records.popitem(last=False)
            _evict(oldest)

    def get(self, puzzle_id: str) -> Optional[PuzzleRecord]:
        """Look up a puzzle by id.

        Args:
            puzzle_id: The puzzle's id.

        Returns:
            The puzzle's record, or None if it doesn't exist (never
            uploaded, or evicted).
        """
        return self._records.get(puzzle_id)

    def list_for_owner(self, owner_id: str) -> List[PuzzleRecord]:
        """List all puzzles owned by a given user, in upload order.

        Args:
            owner_id: The owning user's id.

        Returns:
            The owner's puzzle records, oldest first.
        """
        return [record for record in self._records.values() if record.owner_id == owner_id]


@lru_cache()
def get_puzzle_store() -> PuzzleStore:
    """Get or create the singleton PuzzleStore instance.

    Returns:
        The shared PuzzleStore instance.
    """
    return PuzzleStore()
