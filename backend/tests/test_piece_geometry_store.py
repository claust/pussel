"""Tests for app.services.piece_geometry.store."""

from typing import List, Tuple

import pytest
from piece_geometry_fixtures import (
    PIECE_A_COLORS,
    PIECE_A_EDGE_TYPES,
    PIECE_B_COLORS,
    PIECE_B_EDGE_TYPES,
    deterministic_config,
    paint_quadrants,
    rasterize_piece,
)

from app.services.piece_geometry import store as store_module
from app.services.piece_geometry.contour import mask_to_contour
from app.services.piece_geometry.edges import split_edges
from app.services.piece_geometry.fingerprint import PieceFingerprint, build_fingerprint
from app.services.piece_geometry.store import MatchVerdict, PieceGeometryStore, PuzzlePieceStore


def _fingerprint(edge_types: List[str], colors: List[Tuple[int, int, int]]) -> PieceFingerprint:
    """Build a real PieceFingerprint for a synthetic piece."""
    config = deterministic_config(edge_types)
    mask, _ = rasterize_piece(config)
    contour = mask_to_contour(mask)
    assert contour is not None
    split = split_edges(contour)
    assert split is not None
    edges, _, _ = split
    image_bgr = paint_quadrants(mask, colors)
    return build_fingerprint(edges, image_bgr, contour)


class TestPuzzlePieceStoreVerdicts:
    """new -> matched -> uncertain transitions using real synthetic fingerprints."""

    def test_first_piece_is_always_new(self) -> None:
        """An empty store always enrolls the first photo as new, with no z-score."""
        store = PuzzlePieceStore()
        fp = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)

        result = store.add_or_match(fp, is_clean=True, corner_disagreement=False)

        assert result.verdict == MatchVerdict.NEW
        assert result.piece_id == "p001"
        assert result.match_piece_id is None
        assert result.z_score is None
        assert [p.piece_id for p in store.list_pieces()] == ["p001"]

    def test_rescanning_the_same_piece_matches(self) -> None:
        """Re-submitting the same physical piece's fingerprint matches the enrolled piece_id."""
        store = PuzzlePieceStore()
        fp1 = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        store.add_or_match(fp1, is_clean=True, corner_disagreement=False)

        fp1_again = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        result = store.add_or_match(fp1_again, is_clean=True, corner_disagreement=False)

        assert result.verdict == MatchVerdict.MATCHED
        assert result.piece_id == "p001"
        assert result.match_piece_id == "p001"
        assert result.z_score is not None
        assert result.z_score < 0
        # Not re-enrolled: still only one piece in the store.
        assert len(store.list_pieces()) == 1

    def test_a_clearly_different_piece_is_enrolled_as_new(self) -> None:
        """A geometrically and visually distinct piece is enrolled under a fresh piece_id."""
        store = PuzzlePieceStore()
        fp_a = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        store.add_or_match(fp_a, is_clean=True, corner_disagreement=False)

        fp_b = _fingerprint(PIECE_B_EDGE_TYPES, PIECE_B_COLORS)
        result = store.add_or_match(fp_b, is_clean=True, corner_disagreement=False)

        assert result.verdict == MatchVerdict.NEW
        assert result.piece_id == "p002"
        assert result.match_piece_id is None
        assert result.z_score is not None
        assert len(store.list_pieces()) == 2

    def test_gray_zone_is_uncertain_and_not_enrolled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A z-score between t_accept and t_new is uncertain and does not enroll."""
        store = PuzzlePieceStore()
        fp_a = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        store.add_or_match(fp_a, is_clean=True, corner_disagreement=False)

        # Patch the exact settings object app.services.piece_geometry.store reads at
        # call time (via store_module, not a fresh `from app.config import settings`,
        # since some other test module's fixture may have re-imported app.config and
        # left a *different* Settings instance bound to that name in between).
        monkeypatch.setattr(store_module.settings, "PIECE_GEOMETRY_T_ACCEPT", -1e9)  # nothing can be "matched"
        monkeypatch.setattr(store_module.settings, "PIECE_GEOMETRY_T_NEW", 1e9)  # nothing can be "new"

        fp_b = _fingerprint(PIECE_B_EDGE_TYPES, PIECE_B_COLORS)
        result = store.add_or_match(fp_b, is_clean=True, corner_disagreement=False)

        assert result.verdict == MatchVerdict.UNCERTAIN
        assert result.piece_id is None
        assert result.match_piece_id == "p001"
        assert result.z_score is not None
        # Not enrolled: the store still has only the first piece.
        assert len(store.list_pieces()) == 1

    def test_enroll_uncertain_turns_a_gray_zone_verdict_into_new(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With enroll_uncertain=True, a gray-zone photo is enrolled as a new piece."""
        store = PuzzlePieceStore()
        fp_a = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        store.add_or_match(fp_a, is_clean=True, corner_disagreement=False)

        # Force every comparison into the gray zone (same trick as above).
        monkeypatch.setattr(store_module.settings, "PIECE_GEOMETRY_T_ACCEPT", -1e9)
        monkeypatch.setattr(store_module.settings, "PIECE_GEOMETRY_T_NEW", 1e9)

        fp_b = _fingerprint(PIECE_B_EDGE_TYPES, PIECE_B_COLORS)
        result = store.add_or_match(fp_b, is_clean=True, corner_disagreement=False, enroll_uncertain=True)

        assert result.verdict == MatchVerdict.NEW
        assert result.piece_id == "p002"
        assert result.z_score is not None
        assert len(store.list_pieces()) == 2


class TestGalleryStatsFallback:
    """The store falls back to M7 constants below MIN_GALLERY_FOR_STATS pieces."""

    def test_small_gallery_uses_fallback_stats(self) -> None:
        """With fewer than 12 enrolled pieces, _gallery_stats returns the fallback constants."""
        from app.services.piece_geometry.scoring import FALLBACK_STATS

        store = PuzzlePieceStore()
        store.add_or_match(_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS), True, False)

        stats = store._gallery_stats()  # inspects the fallback boundary directly

        assert stats == FALLBACK_STATS


class TestGalleryStatsCaching:
    """_gallery_stats is O(n^2); it caches and invalidates only when the gallery grows."""

    def test_stats_are_cached_between_calls_and_invalidated_on_enroll(self) -> None:
        """Repeated _gallery_stats calls reuse the cached object until a piece is enrolled."""
        from app.services.piece_geometry.scoring import MIN_GALLERY_FOR_STATS

        store = PuzzlePieceStore()
        fp = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        # Enroll exactly enough pieces to leave the fallback regime, so
        # _gallery_stats takes the real (cacheable) branch.
        for _ in range(MIN_GALLERY_FOR_STATS):
            store._enroll(fp, True, False)

        first = store._gallery_stats()
        second = store._gallery_stats()
        # Same object identity => the second call reused the cache, not recomputed.
        assert first is second

        # Enrolling a piece must invalidate the cache: the next call recomputes.
        store._enroll(fp, True, False)
        third = store._gallery_stats()
        assert third is not first


class TestPieceGeometryStore:
    """Tests for the puzzle_id-keyed PieceGeometryStore wrapper."""

    def test_stores_are_isolated_per_puzzle(self) -> None:
        """Two puzzles' piece stores don't share enrolled pieces or id sequences."""
        store = PieceGeometryStore()
        fp_a = _fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        fp_b = _fingerprint(PIECE_B_EDGE_TYPES, PIECE_B_COLORS)

        result_1 = store.add_or_match("puzzle-1", fp_a, True, False)
        result_2 = store.add_or_match("puzzle-2", fp_b, True, False)

        assert result_1.piece_id == "p001"
        assert result_2.piece_id == "p001"
        assert len(store.list_pieces("puzzle-1")) == 1
        assert len(store.list_pieces("puzzle-2")) == 1

    def test_none_fingerprint_is_uncertain_and_not_enrolled(self) -> None:
        """A None fingerprint (an earlier pipeline stage failed) is reported uncertain, never enrolled."""
        store = PieceGeometryStore()

        result = store.add_or_match("puzzle-1", None, is_clean=False, corner_disagreement=True)

        assert result.verdict == MatchVerdict.UNCERTAIN
        assert result.piece_id is None
        assert result.match_piece_id is None
        assert result.z_score is None
        assert store.list_pieces("puzzle-1") == []

    def test_list_pieces_for_unknown_puzzle_is_empty(self) -> None:
        """Listing a puzzle with no store yet returns an empty list, not an error."""
        store = PieceGeometryStore()

        assert store.list_pieces("never-seen") == []
