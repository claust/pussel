"""Tests for the box-shot piece-count estimator.

The candidate filter is tested purely; `estimate_piece_count` end-to-end
tests render synthetic box shots with cv2's Hershey font and are skipped
when the tesseract binary isn't installed (the estimator's documented
degraded mode, itself covered by `test_missing_tesseract_disables_estimation`).
"""

import shutil
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.services.piece_count_estimator import _extract_candidates, _OcrWord, estimate_piece_count

requires_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")


def word(text: str, height: int = 150, confidence: float = 95.0) -> _OcrWord:
    """Build an OCR word with plausible badge-sized defaults."""
    return _OcrWord(text=text, height=height, confidence=confidence)


def synthetic_box_jpeg(text: str, color: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    """Render a box-shot stand-in: `text` large in the top-left, like the real badge.

    Args:
        text: The badge text to print (empty for a blank box).
        color: BGR text color.

    Returns:
        JPEG bytes of the rendered image.
    """
    image = np.full((750, 1000, 3), 255, dtype=np.uint8)
    if text:
        cv2.putText(image, text, (60, 190), cv2.FONT_HERSHEY_DUPLEX, 5.5, color, 14, cv2.LINE_AA)
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


class TestExtractCandidates:
    """The token filter that turns OCR words into piece-count candidates."""

    def test_plain_counts_accepted(self) -> None:
        """Numbers in the known catalog set become candidates."""
        candidates = _extract_candidates([word("1000"), word("500"), word("100")])
        assert [c.value for c in candidates] == [1000, 500, 100]

    def test_unknown_numbers_rejected(self) -> None:
        """Years, article fragments, and off-catalog numbers never match."""
        assert _extract_candidates([word("1891"), word("900"), word("12"), word("12345")]) == []

    def test_decorations_stripped(self) -> None:
        """Quotes, punctuation, and thousands separators are cleaned before matching."""
        candidates = _extract_candidates([word('"500'), word("1.000"), word("1,000")])
        assert [c.value for c in candidates] == [500, 1000, 1000]

    def test_multipack_yields_per_puzzle_count(self) -> None:
        """An 'N x M' kids multipack estimates M — the pieces of one puzzle."""
        candidates = _extract_candidates([word("2x12"), word("3 x 49"), word("2×24")])
        assert [c.value for c in candidates] == [12, 49, 24]

    def test_box_dimensions_rejected(self) -> None:
        """Centimetre dimensions like 70x50 don't parse as multipacks."""
        assert _extract_candidates([word("70x50"), word("2x37")]) == []

    def test_non_numeric_ignored(self) -> None:
        """Plain words produce no candidates."""
        assert _extract_candidates([word("Ravensburger"), word("Puzzle")]) == []


class TestEstimatePieceCount:
    """End-to-end behavior of `estimate_piece_count`."""

    def test_undecodable_bytes_yield_none(self) -> None:
        """Garbage bytes are a graceful None, not an exception."""
        assert estimate_piece_count(b"not-a-jpeg" * 100) is None

    def test_tiny_image_yields_none(self) -> None:
        """Images too small to carry a readable badge are skipped without OCR."""
        _, encoded = cv2.imencode(".jpg", np.full((64, 64, 3), 255, dtype=np.uint8))
        assert estimate_piece_count(encoded.tobytes()) is None

    def test_missing_tesseract_disables_estimation(self) -> None:
        """Without the tesseract binary the estimator degrades to None."""
        with patch("app.services.piece_count_estimator.shutil.which", return_value=None):
            assert estimate_piece_count(synthetic_box_jpeg("1000")) is None

    @requires_tesseract
    def test_reads_badge_count(self) -> None:
        """A big top-left '1000' — the standard badge — is read."""
        assert estimate_piece_count(synthetic_box_jpeg("1000")) == 1000

    @requires_tesseract
    def test_reads_colored_badge(self) -> None:
        """A pink-on-white badge (which plain binarization washes out) is read."""
        assert estimate_piece_count(synthetic_box_jpeg("1000", color=(140, 30, 233))) == 1000

    @requires_tesseract
    def test_blank_box_yields_none(self) -> None:
        """A box with no printed count yields None."""
        assert estimate_piece_count(synthetic_box_jpeg("")) is None

    @requires_tesseract
    def test_off_catalog_number_yields_none(self) -> None:
        """A prominent number outside the catalog set (the 1891 tagline) is not a count."""
        assert estimate_piece_count(synthetic_box_jpeg("1891")) is None
