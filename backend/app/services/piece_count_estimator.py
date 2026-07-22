"""OCR-based piece-count estimation from Ravensburger box shots.

Every Ravensburger box prints its piece count prominently near the top-left
corner of the front (8-digit "12xxxxxx" boxes and square gift boxes) or
rotated 90° along the left edge (classic 5-digit softclick boxes, Nathan
boxes). The clean puzzle motif (`_1` shot) deliberately has no text, so the
estimate always reads the box shot.

The pipeline OCRs a few targeted crops of the box shot with tesseract and
keeps tokens that exactly match a piece count Ravensburger actually sells
(plus "N x M" kids multipacks, where the per-puzzle M is the estimate).
The tallest such token wins — the count is always the biggest number on the
box, which is what beats look-alike numerals like the "Since 1891" tagline,
age badges, and centimetre dimensions. Validated against 21 live box shots
(19 matching the expected label, 2 conservative Nones, 0 wrong guesses):
an unreadable box returns None rather than a wrong guess, so the app's
piece-count field just stays empty.

Tesseract is invoked via subprocess (no extra Python dependency); when the
binary is not installed the estimator degrades to always returning None.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Piece counts Ravensburger sells as single-puzzle boxes. Deliberately a
# closed set: exact-match filtering is what keeps OCR noise (years, article
# numbers, box dimensions) from ever becoming an estimate.
PLAIN_COUNTS = frozenset({24, 35, 49, 54, 60, 80, 100, 125, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000})
# Kids multipacks are printed "N x M" (e.g. "2x12", "3x49"); N is the number
# of puzzles in the box and M — the estimate — the pieces per puzzle. The
# small-N requirement rejects box dimensions like "70x50".
MULTIPACK_PUZZLES = frozenset({2, 3, 4})
MULTIPACK_PIECES = frozenset({12, 20, 24, 49, 100})

_PACK_RE = re.compile(r"^([234])\s*[xX]\s*(\d{1,3})$")
_NUM_RE = re.compile(r"^\d{2,4}$")
# A thousands separator inside a count ("1.000" / "1,000").
_SEP_RE = re.compile(r"^(\d)[.,](\d{3})$")
_STRIP_CHARS = "\"'.,;:()|«»“”*"

# Tokens shorter than this (pixels, in the upscaled OCR input) are fine print
# — the "1000 Teile" contents line, not the front-of-box count.
_MIN_TOKEN_HEIGHT = 40
# Tokens below this tesseract confidence are dropped outright. On the live
# validation set every correct badge read scored ≥ 63 while every misread
# (e.g. a stylized "500" read as "200") scored below 58 — a threshold between
# the two turns would-be wrong guesses into safe Nones.
_MIN_TOKEN_CONFIDENCE = 60.0
# A candidate this tall and confident ends the variant sweep early.
_EARLY_EXIT_HEIGHT = 100
_EARLY_EXIT_CONFIDENCE = 80.0
# Box shots smaller than this can't carry a readable count badge.
_MIN_IMAGE_DIMENSION = 200

_TESSERACT_TIMEOUT_SECONDS = 20.0
_warned_missing_tesseract = False


@dataclass(frozen=True)
class _Candidate:
    """A piece-count reading from one OCR token."""

    value: int
    height: int
    confidence: float


@dataclass(frozen=True)
class _OcrWord:
    """One word row from tesseract's TSV output."""

    text: str
    height: int
    confidence: float


def _tesseract_available() -> bool:
    """Report whether the tesseract binary is on PATH, warning once when not.

    Returns:
        True when tesseract can be invoked.
    """
    global _warned_missing_tesseract
    if shutil.which("tesseract") is not None:
        return True
    if not _warned_missing_tesseract:
        logger.warning("tesseract binary not found; piece-count estimation is disabled")
        _warned_missing_tesseract = True
    return False


def _ocr_words(image: np.ndarray, psm: int) -> list[_OcrWord]:
    """Run tesseract on a grayscale/binary image and parse its TSV words.

    Args:
        image: Single-channel image to OCR.
        psm: Tesseract page-segmentation mode (11 = sparse text, 7 = single
            line).

    Returns:
        The recognized words with glyph heights and confidences; empty on any
        tesseract failure (never raises).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ocr.png"
        cv2.imwrite(str(path), image)
        try:
            result = subprocess.run(
                ["tesseract", str(path), "-", "--psm", str(psm), "tsv"],
                capture_output=True,
                text=True,
                timeout=_TESSERACT_TIMEOUT_SECONDS,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("tesseract invocation failed: %s", exc)
            return []
    words = []
    for line in result.stdout.splitlines()[1:]:
        columns = line.split("\t")
        if len(columns) < 12 or not columns[11].strip():
            continue
        try:
            height, confidence = int(columns[9]), float(columns[10])
        except ValueError:
            continue
        words.append(_OcrWord(text=columns[11].strip(), height=height, confidence=confidence))
    return words


def _extract_candidates(words: Sequence[_OcrWord]) -> list[_Candidate]:
    """Filter OCR words down to plausible piece counts.

    Args:
        words: Recognized words from one OCR pass.

    Returns:
        A candidate per word that exactly matches a known plain count or an
        "N x M" multipack (whose per-puzzle M becomes the value).
    """
    candidates = []
    for word in words:
        token = word.text.strip(_STRIP_CHARS).replace("×", "x")
        token = _SEP_RE.sub(r"\1\2", token)
        pack = _PACK_RE.match(token)
        if pack and int(pack.group(1)) in MULTIPACK_PUZZLES and int(pack.group(2)) in MULTIPACK_PIECES:
            candidates.append(_Candidate(value=int(pack.group(2)), height=word.height, confidence=word.confidence))
        elif _NUM_RE.match(token) and int(token) in PLAIN_COUNTS:
            candidates.append(_Candidate(value=int(token), height=word.height, confidence=word.confidence))
    return candidates


def _channel_variants(crop: np.ndarray, scale: int) -> Iterator[np.ndarray]:
    """Yield single-channel OCR inputs for a BGR crop.

    Plain grayscale reads most badges; an Otsu threshold of the darkest
    channel isolates colored text on light box borders (tesseract's own
    binarization loses e.g. pink-on-white), and an inverted threshold of the
    brightest channel isolates light text on dark artwork.

    Args:
        crop: BGR crop of the box shot.
        scale: Upscale factor applied before OCR.

    Yields:
        Grayscale, dark-text-binarized, and light-text-binarized images.
    """
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    yield cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(resized.min(axis=2), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield dark
    _, light = cv2.threshold(resized.max(axis=2), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    yield light


def _attempts(image: np.ndarray) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (OCR input, psm) attempts in decreasing hit-rate order.

    The top band catches upright counts (8-digit-era boxes); the left band,
    rotated to upright both ways, catches the 90°-rotated counts of classic
    5-digit boxes; a single-line pass over the top-left quadrant rescues
    stylized digits that sparse-text segmentation splits apart (square gift
    boxes).

    Args:
        image: The full BGR box shot.

    Yields:
        (single-channel image, tesseract psm) pairs.
    """
    height, width = image.shape[:2]
    top_band = image[: int(height * 0.32), :]
    left_band = image[:, : int(width * 0.30)]
    top_left = image[: int(height * 0.32), : width // 2]
    for variant in _channel_variants(top_band, scale=3):
        yield variant, 11
    for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        for variant in _channel_variants(cv2.rotate(left_band, rotation), scale=3):
            yield variant, 11
    for variant in _channel_variants(top_left, scale=2):
        yield variant, 7


def estimate_piece_count(box_jpeg: bytes) -> Optional[int]:
    """Estimate a puzzle's piece count from its Ravensburger box shot.

    Args:
        box_jpeg: The box-shot image bytes as fetched from ravensburger.org.

    Returns:
        The most likely piece count, or None when no confident reading exists
        (undecodable/tiny image, tesseract unavailable, or no token matching
        a known count) — never a low-confidence guess.
    """
    if not _tesseract_available():
        return None
    image = cv2.imdecode(np.frombuffer(box_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < _MIN_IMAGE_DIMENSION:
        return None
    best: Optional[_Candidate] = None
    for variant, psm in _attempts(image):
        for candidate in _extract_candidates(_ocr_words(variant, psm)):
            if candidate.height < _MIN_TOKEN_HEIGHT or candidate.confidence < _MIN_TOKEN_CONFIDENCE:
                continue
            if best is None or (candidate.height, candidate.confidence) > (best.height, best.confidence):
                best = candidate
        if best is not None and best.height >= _EARLY_EXIT_HEIGHT and best.confidence >= _EARLY_EXIT_CONFIDENCE:
            break
    return best.value if best else None
