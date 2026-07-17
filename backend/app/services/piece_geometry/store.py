"""Per-puzzle piece store: fingerprint dedupe ("new piece" vs "seen before").

Ported from ``network/experiments/exp28_piece_geometry/collision_study.py``'s
two-threshold scan-lock recipe (M7) — keep algorithm changes in sync.

For each incoming photo's fingerprint, the store compares it against every
piece already enrolled for the puzzle, combines shape + spatial-color
distance into a z-score (`app.services.piece_geometry.scoring.combined_z`)
against the closest match, and returns one of three verdicts:

- ``matched`` (z < t_accept): the photo is the same physical piece as an
  already-enrolled one; not re-enrolled, the existing piece_id is returned.
- ``new`` (z > t_new, or the store is empty): a piece not seen before;
  enrolled under a freshly minted piece_id.
- ``uncertain`` (the gray zone in between): NOT enrolled; the closest
  existing piece_id and the z-score are returned so the caller can ask the
  user to rescan.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.services.piece_geometry.fingerprint import PieceFingerprint, shape_pair_distance, spatial_pair_distance
from app.services.piece_geometry.scoring import (
    FALLBACK_STATS,
    MIN_GALLERY_FOR_STATS,
    GalleryStats,
    combined_z,
    gallery_impostor_stats,
)


class MatchVerdict(str, Enum):
    """Outcome of comparing a new piece-photo fingerprint against a puzzle's store."""

    MATCHED = "matched"
    NEW = "new"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class EnrolledPiece:
    """One physical piece enrolled in a puzzle's store.

    Attributes:
        piece_id: The piece's stable id within this puzzle (e.g. "p001").
        fingerprint: The piece's shape + spatial-color fingerprint.
        is_clean: Whether the enrolling photo's contour passed the quality gate.
        corner_disagreement: The enrolling photo's corner_disagreement flag.
    """

    piece_id: str
    fingerprint: PieceFingerprint
    is_clean: bool
    corner_disagreement: bool


@dataclass(frozen=True)
class MatchResult:
    """The verdict of `PuzzlePieceStore.add_or_match`.

    Attributes:
        verdict: matched / new / uncertain.
        piece_id: This photo's piece id — the matched piece's id (matched),
            a freshly minted id (new), or None (uncertain: not enrolled).
        match_piece_id: The closest existing piece's id, when a comparison
            was made (matched and uncertain); None otherwise.
        z_score: The closest match's combined z-score, when a comparison was
            made; None when the store was empty.
    """

    verdict: MatchVerdict
    piece_id: Optional[str]
    match_piece_id: Optional[str]
    z_score: Optional[float]


@dataclass
class PuzzlePieceStore:
    """In-memory fingerprint gallery + dedupe for one puzzle's scanned pieces."""

    _pieces: List[EnrolledPiece] = field(default_factory=list)
    _next_id: int = 1
    # Cached impostor stats and the gallery size they were computed for.
    # `_gallery_stats` is O(n^2) in the gallery and runs on every locked frame,
    # but the stats only change when the gallery does — so cache them and let
    # `_enroll` and `remove` invalidate. Pieces are never mutated in place, so
    # every change routes through one of those two.
    #
    # The size alone is NOT a sufficient cache key now that pieces can be
    # removed: a remove followed by an enroll returns to the same size with a
    # different gallery. Both mutators clear `_cached_stats` outright, so the
    # size only guards the cache against a stale hit, never validates it alone.
    _cached_stats: Optional[GalleryStats] = None
    _cached_stats_n: int = -1

    def _mint_piece_id(self) -> str:
        """Allocate the next stable piece id for this puzzle.

        Returns:
            A short id like "p001", unique within this puzzle's store.
        """
        piece_id = f"p{self._next_id:03d}"
        self._next_id += 1
        return piece_id

    def _enroll(self, fingerprint: PieceFingerprint, is_clean: bool, corner_disagreement: bool) -> str:
        """Mint an id, append the piece, and invalidate the cached impostor stats.

        All enrollments route through here so the `_gallery_stats` cache can
        never go stale behind an append.

        Args:
            fingerprint: The new piece's fingerprint.
            is_clean: The enrolling photo's contour-quality gate result.
            corner_disagreement: The enrolling photo's corner_disagreement flag.

        Returns:
            The freshly minted piece id.
        """
        piece_id = self._mint_piece_id()
        self._pieces.append(EnrolledPiece(piece_id, fingerprint, is_clean, corner_disagreement))
        self._cached_stats = None
        return piece_id

    def _gallery_stats(self) -> GalleryStats:
        """Impostor shape/spatial statistics for the current gallery, or the M7 fallback.

        Returns:
            `FALLBACK_STATS` when fewer than `MIN_GALLERY_FOR_STATS` pieces
            are enrolled; otherwise the gallery's own impostor statistics.
        """
        if len(self._pieces) < MIN_GALLERY_FOR_STATS:
            return FALLBACK_STATS

        if self._cached_stats is not None and self._cached_stats_n == len(self._pieces):
            return self._cached_stats

        pairwise_shape: List[float] = []
        pairwise_spatial: List[float] = []
        for i, piece_a in enumerate(self._pieces):
            for piece_b in self._pieces[i + 1 :]:
                d_shape, k = shape_pair_distance(
                    piece_a.fingerprint.edges_canonical,
                    piece_a.fingerprint.edge_types,
                    piece_b.fingerprint.edges_canonical,
                    piece_b.fingerprint.edge_types,
                )
                d_spatial = spatial_pair_distance(piece_a.fingerprint, piece_b.fingerprint, k)
                pairwise_shape.append(d_shape)
                pairwise_spatial.append(d_spatial)
        stats = gallery_impostor_stats(pairwise_shape, pairwise_spatial)
        self._cached_stats = stats
        self._cached_stats_n = len(self._pieces)
        return stats

    def _closest_match(self, fingerprint: PieceFingerprint, stats: GalleryStats) -> Tuple[str, float]:
        """Find the enrolled piece closest to `fingerprint` by combined z-score.

        Args:
            fingerprint: The query piece's fingerprint.
            stats: Normalization statistics to combine distances with.

        Returns:
            Tuple of (closest piece's id, its combined z-score).
        """
        best_z = float("inf")
        best_piece_id = self._pieces[0].piece_id
        for enrolled in self._pieces:
            d_shape, k = shape_pair_distance(
                fingerprint.edges_canonical,
                fingerprint.edge_types,
                enrolled.fingerprint.edges_canonical,
                enrolled.fingerprint.edge_types,
            )
            d_spatial = spatial_pair_distance(fingerprint, enrolled.fingerprint, k)
            z = combined_z(d_shape, d_spatial, stats)
            if z < best_z:
                best_z = z
                best_piece_id = enrolled.piece_id
        return best_piece_id, best_z

    def add_or_match(
        self,
        fingerprint: PieceFingerprint,
        is_clean: bool,
        corner_disagreement: bool,
        enroll_uncertain: bool = False,
    ) -> MatchResult:
        """Compare a piece-photo fingerprint against this puzzle's store and enroll if new.

        Args:
            fingerprint: The photo's shape + spatial-color fingerprint.
            is_clean: The photo's contour-quality gate result.
            corner_disagreement: The photo's corner_disagreement flag.
            enroll_uncertain: When True, a gray-zone (uncertain) verdict
                enrolls the photo as a NEW piece instead of reporting
                uncertainty — the client's escape hatch after its own
                confirmation UX (M7 measured that 97.9% of genuinely-new
                pieces land in the gray zone, so a session could otherwise
                never enroll a second piece). Matched and new verdicts are
                unaffected.

        Returns:
            The `MatchResult`. See the module docstring for the verdict semantics.
        """
        if not self._pieces:
            piece_id = self._enroll(fingerprint, is_clean, corner_disagreement)
            return MatchResult(MatchVerdict.NEW, piece_id, None, None)

        stats = self._gallery_stats()
        match_piece_id, z_score = self._closest_match(fingerprint, stats)

        if z_score < settings.PIECE_GEOMETRY_T_ACCEPT:
            return MatchResult(MatchVerdict.MATCHED, match_piece_id, match_piece_id, z_score)

        if z_score > settings.PIECE_GEOMETRY_T_NEW or enroll_uncertain:
            piece_id = self._enroll(fingerprint, is_clean, corner_disagreement)
            return MatchResult(MatchVerdict.NEW, piece_id, None, z_score)

        return MatchResult(MatchVerdict.UNCERTAIN, None, match_piece_id, z_score)

    def remove(self, piece_id: str) -> bool:
        """Un-enroll one piece, so a later photo of it reads as new again.

        Ids are never recycled (`_next_id` only counts up), so a removed id
        cannot come back attached to a different physical piece.

        Args:
            piece_id: The piece to remove.

        Returns:
            True if the piece was enrolled and is now gone; False if this
            puzzle never had a piece with that id.
        """
        before = len(self._pieces)
        self._pieces = [piece for piece in self._pieces if piece.piece_id != piece_id]
        if len(self._pieces) == before:
            return False
        self._cached_stats = None
        return True

    def list_pieces(self) -> List[EnrolledPiece]:
        """List all pieces enrolled so far, in enrollment order.

        Returns:
            A shallow copy of the enrolled-piece list.
        """
        return list(self._pieces)


class PieceGeometryStore:
    """Piece-geometry stores for every puzzle, keyed by puzzle_id."""

    def __init__(self) -> None:
        """Initialize an empty collection of per-puzzle stores."""
        self._stores: Dict[str, PuzzlePieceStore] = {}

    def _get_or_create(self, puzzle_id: str) -> PuzzlePieceStore:
        """Get a puzzle's store, creating an empty one on first access.

        Args:
            puzzle_id: The puzzle's id.

        Returns:
            The puzzle's `PuzzlePieceStore`.
        """
        if puzzle_id not in self._stores:
            self._stores[puzzle_id] = PuzzlePieceStore()
        return self._stores[puzzle_id]

    def add_or_match(
        self,
        puzzle_id: str,
        fingerprint: Optional[PieceFingerprint],
        is_clean: bool,
        corner_disagreement: bool,
        enroll_uncertain: bool = False,
    ) -> MatchResult:
        """Compare a piece-photo fingerprint against one puzzle's store and enroll if new.

        Args:
            puzzle_id: The puzzle the photo belongs to.
            fingerprint: The photo's fingerprint, or None when a prior
                pipeline stage (contour/corners/edges) failed.
            is_clean: The photo's contour-quality gate result.
            corner_disagreement: The photo's corner_disagreement flag.
            enroll_uncertain: When True, a gray-zone verdict enrolls the
                photo as a new piece (see `PuzzlePieceStore.add_or_match`).
                Has no effect when `fingerprint` is None: a failed pipeline
                produced nothing that could be enrolled.

        Returns:
            `MatchResult(UNCERTAIN, None, None, None)` when `fingerprint` is
            None (nothing to compare or enroll); otherwise the puzzle
            store's `add_or_match` result.
        """
        if fingerprint is None:
            return MatchResult(MatchVerdict.UNCERTAIN, None, None, None)
        return self._get_or_create(puzzle_id).add_or_match(
            fingerprint, is_clean, corner_disagreement, enroll_uncertain=enroll_uncertain
        )

    def remove(self, puzzle_id: str, piece_id: str) -> bool:
        """Un-enroll one piece from one puzzle's store.

        Called when the user deletes a scanned piece from the piece list: the
        piece must stop appearing in the scanner's gallery, and a fresh photo
        of it must read as new rather than as an already-scanned duplicate.

        Args:
            puzzle_id: The puzzle the piece belongs to.
            piece_id: The piece to remove.

        Returns:
            True if the piece was enrolled and is now gone; False when the
            puzzle has no store yet or holds no such piece.
        """
        store = self._stores.get(puzzle_id)
        return store.remove(piece_id) if store else False

    def list_pieces(self, puzzle_id: str) -> List[EnrolledPiece]:
        """List all pieces enrolled for one puzzle.

        Args:
            puzzle_id: The puzzle's id.

        Returns:
            The puzzle's enrolled pieces, or an empty list when the puzzle
            has no store yet.
        """
        store = self._stores.get(puzzle_id)
        return store.list_pieces() if store else []


_piece_geometry_store: Optional[PieceGeometryStore] = None


def get_piece_geometry_store() -> PieceGeometryStore:
    """Get or create the singleton PieceGeometryStore instance.

    Returns:
        The shared PieceGeometryStore instance.
    """
    global _piece_geometry_store
    if _piece_geometry_store is None:
        _piece_geometry_store = PieceGeometryStore()
    return _piece_geometry_store
