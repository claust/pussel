"""Shared utilities for exp28: piece geometry extraction and corner detection.

Reuses the general segmentation approach from
``backend/app/services/piece_detector.py`` (harden alpha, largest connected
component, morphological cleanup) but is implemented locally so this
experiment stays self-contained apart from the ``puzzle_shapes`` workspace
package.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter1d

# rembg model used for segmentation, matching backend/app/services/background_remover.py.
REMBG_MODEL = "u2net"

# Alpha value above which a rembg pixel counts as opaque (matches piece_detector.py).
ALPHA_THRESHOLD = 128

# Morphological open+close kernel used to clean up masks before contour extraction.
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# A connected component must be at least this fraction of the largest component's
# area to count as a distinct "large" component for quality scoring.
LARGE_COMPONENT_AREA_FRAC = 0.02

# A contour point within this many pixels of the crop edge counts as border-touching.
BORDER_TOUCH_MARGIN_PX = 2.0


@dataclass(frozen=True)
class PieceRecord:
    """One row of `north_star/v1/metadata.csv`: a single piece photo.

    Attributes:
        puzzle_id: Puzzle identifier, e.g. "puzzle01_frozen_scene".
        piece_file: Path to the piece photo, relative to the dataset root.
        rows: Number of rows in the puzzle's grid.
        cols: Number of columns in the puzzle's grid.
        row: Piece's row index (0-indexed).
        col: Piece's column index (0-indexed).
        rotation: Applied clockwise rotation label (0/90/180/270).
        background: Background surface name (red_carpet, gray_fabric, cardboard, wood).
        source_image: Original capture filename (HEIC).
        captured_at: Capture timestamp string.
        device: Capture device string.
        image_w: Full photo width in pixels.
        image_h: Full photo height in pixels.
        bbox: Piece bounding box (x1, y1, x2, y2), inclusive pixel coordinates
            in the upright photo.
        applied_k: Number of 90-degree rotations applied to make the shot upright.
        orientation_source: Provenance of the upright rotation (e.g. "arrow+match").
        flagged: Whether the ingest pipeline flagged this piece for review.
        bbox_suspect: Whether the bounding box detection was flagged as suspect.
    """

    puzzle_id: str
    piece_file: str
    rows: int
    cols: int
    row: int
    col: int
    rotation: int
    background: str
    source_image: str
    captured_at: str
    device: str
    image_w: int
    image_h: int
    bbox: Tuple[int, int, int, int]
    applied_k: int
    orientation_source: str
    flagged: bool
    bbox_suspect: bool

    @property
    def piece_stem(self) -> str:
        """Filename stem for outputs derived from this piece photo, e.g. "piece_r00_c00_red_carpet"."""
        return Path(self.piece_file).stem


def load_metadata(dataset_root: Path) -> List[PieceRecord]:
    """Load `metadata.csv` from a north_star dataset root.

    Args:
        dataset_root: Directory containing `metadata.csv` and per-puzzle folders.

    Returns:
        One `PieceRecord` per row, in file order.
    """
    csv_path = dataset_root / "metadata.csv"
    records: List[PieceRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                PieceRecord(
                    puzzle_id=row["puzzle_id"],
                    piece_file=row["piece_file"],
                    rows=int(row["rows"]),
                    cols=int(row["cols"]),
                    row=int(row["row"]),
                    col=int(row["col"]),
                    rotation=int(row["rotation"]),
                    background=row["background"],
                    source_image=row["source_image"],
                    captured_at=row["captured_at"],
                    device=row["device"],
                    image_w=int(row["image_w"]),
                    image_h=int(row["image_h"]),
                    bbox=(
                        int(row["piece_x1"]),
                        int(row["piece_y1"]),
                        int(row["piece_x2"]),
                        int(row["piece_y2"]),
                    ),
                    applied_k=int(row["applied_k"]),
                    orientation_source=row["orientation_source"],
                    flagged=row["flagged"] == "1",
                    bbox_suspect=row["bbox_suspect"] == "1",
                )
            )
    return records


def crop_with_margin(
    image: np.ndarray, bbox: Tuple[int, int, int, int], margin_frac: float = 0.15
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop an image to a bounding box expanded by a margin, clamped to image bounds.

    Args:
        image: Source image as a numpy array (H, W, C) or (H, W).
        bbox: Bounding box (x1, y1, x2, y2), inclusive pixel coordinates.
        margin_frac: Extra margin added on each side, as a fraction of the
            bbox width/height.

    Returns:
        Tuple of (crop, (offset_x, offset_y)) where offset is the crop's
        top-left corner in the original image's pixel coordinates.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = round(box_w * margin_frac)
    pad_y = round(box_h * margin_frac)

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x + 1)
    bottom = min(height, y2 + pad_y + 1)

    crop = image[top:bottom, left:right]
    return crop, (left, top)


_rembg_session = None


def get_rembg_session():  # type: ignore[no-untyped-def]
    """Get or create the shared rembg session (created once, reused across calls).

    Returns:
        The rembg session object for `REMBG_MODEL`.
    """
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session(REMBG_MODEL)
    return _rembg_session


def remove_background_rgba(bgr_image: np.ndarray) -> np.ndarray:
    """Run rembg segmentation on a BGR image crop.

    Args:
        bgr_image: Image crop in OpenCV BGR channel order.

    Returns:
        RGBA numpy array (H, W, 4) with the segmentation alpha matte.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    session = get_rembg_session()
    output = remove(pil_image, session=session)
    return np.asarray(output.convert("RGBA"))


def alpha_to_mask(rgba: np.ndarray, alpha_threshold: int = ALPHA_THRESHOLD) -> np.ndarray:
    """Harden a soft rembg alpha matte into a binary mask.

    Mirrors `harden_alpha` in `backend/app/services/piece_detector.py`: pixels
    at or below the threshold become background, pixels above it become
    foreground.

    Args:
        rgba: RGBA image array (H, W, 4).
        alpha_threshold: Alpha value above which a pixel counts as opaque.

    Returns:
        Binary mask (H, W) with values in {0, 255}.
    """
    alpha = rgba[..., 3]
    return np.where(alpha > alpha_threshold, np.uint8(255), np.uint8(0))


def otsu_masks(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Threshold a grayscale image with Otsu's method at both polarities.

    Args:
        gray: Single-channel grayscale image (H, W).

    Returns:
        Tuple of (normal_mask, inverted_mask), each a binary mask in {0, 255}.
    """
    _, normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return normal, inverted


def mask_to_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Clean up a binary mask and extract its largest external contour.

    Applies a morphological open then close (5px ellipse) to remove speckle
    noise and close small gaps, then returns the largest external contour at
    full point density.

    Args:
        mask: Binary mask (H, W) with values in {0, 255}.

    Returns:
        Nx2 float array of contour points in the mask's own pixel coordinates
        (row/col frame of `mask`), or None when no contour is found.
    """
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    return largest.reshape(-1, 2).astype(np.float64)


def smooth_contour(contour: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth a closed contour with circular Gaussian filtering.

    Args:
        contour: Nx2 array of contour points, implicitly closed (first point
            follows the last).
        sigma: Gaussian standard deviation in samples.

    Returns:
        Nx2 smoothed contour, same length as the input.
    """
    x = gaussian_filter1d(contour[:, 0], sigma=sigma, mode="wrap")
    y = gaussian_filter1d(contour[:, 1], sigma=sigma, mode="wrap")
    return np.column_stack([x, y])


def resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to `n` arc-length-equidistant points.

    Adapted from `network/docs/puzzle_shape_generation/scripts/shape_comparator.py`,
    but treats the contour as a closed loop (wraps back to the first point)
    rather than an open polyline.

    Args:
        contour: Nx2 array of contour points, implicitly closed.
        n: Number of output points.

    Returns:
        Nx2 resampled contour.
    """
    closed = np.vstack([contour, contour[:1]])
    diffs = np.diff(closed, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length == 0:
        idx = np.linspace(0, len(contour) - 1, n).astype(int)
        return contour[idx]

    target_lengths = np.linspace(0, total_length, n, endpoint=False)
    resampled_x = np.interp(target_lengths, cumulative_length, closed[:, 0])
    resampled_y = np.interp(target_lengths, cumulative_length, closed[:, 1])
    return np.column_stack([resampled_x, resampled_y])


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample an OPEN polyline to `n` arc-length-equidistant points.

    Unlike `resample_contour`, the polyline is not closed: the first and last
    output points coincide with the input endpoints.

    Args:
        points: Mx2 array of polyline points.
        n: Number of output points (>= 2).

    Returns:
        Nx2 resampled polyline.
    """
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length == 0:
        return np.repeat(points[:1], n, axis=0)

    target_lengths = np.linspace(0, total_length, n)
    resampled_x = np.interp(target_lengths, cumulative_length, points[:, 0])
    resampled_y = np.interp(target_lengths, cumulative_length, points[:, 1])
    return np.column_stack([resampled_x, resampled_y])


# --- Contact-sheet plumbing (shared by the review scripts) -----------------

SHEET_TITLE_HEIGHT = 18


def make_titled_cell(image: np.ndarray, label: str, cell_width: int) -> np.ndarray:
    """Resize an image to a fixed cell width and stack a title strip on top.

    Args:
        image: BGR image for the cell body.
        label: Title text (e.g. "r00c03").
        cell_width: Output cell width in pixels.

    Returns:
        BGR cell image of width `cell_width` including the title strip.
    """
    scale = cell_width / image.shape[1]
    resized = cv2.resize(image, (cell_width, max(1, round(image.shape[0] * scale))))

    title = np.full((SHEET_TITLE_HEIGHT, cell_width, 3), 255, dtype=np.uint8)
    cv2.putText(title, label, (2, SHEET_TITLE_HEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([title, resized])


def assemble_grid_sheet(cells: dict, rows: int, cols: int, cell_width: int) -> np.ndarray:
    """Assemble per-position cells into one contact-sheet image.

    Missing positions get a gray placeholder; cells in a row are bottom-padded
    to a common height so rows stack cleanly.

    Args:
        cells: Dict mapping (row, col) to a BGR cell image of width `cell_width`.
        rows: Grid row count.
        cols: Grid column count.
        cell_width: Cell width in pixels (all cells must match).

    Returns:
        The assembled BGR contact sheet.
    """
    placeholder = np.full((SHEET_TITLE_HEIGHT + cell_width, cell_width, 3), 128, dtype=np.uint8)
    sheet_rows: List[np.ndarray] = []
    for row in range(rows):
        row_cells = [cells.get((row, col), placeholder) for col in range(cols)]
        row_h = max(cell.shape[0] for cell in row_cells)
        padded = []
        for cell in row_cells:
            if cell.shape[0] < row_h:
                pad = np.full((row_h - cell.shape[0], cell_width, 3), 255, dtype=np.uint8)
                cell = np.vstack([cell, pad])
            padded.append(cell)
        sheet_rows.append(np.hstack(padded))
    return np.vstack(sheet_rows)


@dataclass(frozen=True)
class QualityMetrics:
    """Quality signals for a contour extracted from a piece photo.

    Attributes:
        n_large_components: Number of connected components in the source mask
            with area at least `LARGE_COMPONENT_AREA_FRAC` of the largest one.
        border_touching: Whether any contour point lies within
            `BORDER_TOUCH_MARGIN_PX` of the crop edge (segmentation likely
            clipped the piece).
        area_ratio: Contour area divided by the crop's pixel area.
        solidity: Contour area divided by its convex hull area (low solidity
            indicates a ragged or multi-lobed contour).
        is_clean: Overall pass/fail: exactly one large component, not
            border-touching, plausible area ratio, and plausible solidity.
    """

    n_large_components: int
    border_touching: bool
    area_ratio: float
    solidity: float
    is_clean: bool

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "n_large_components": self.n_large_components,
            "border_touching": self.border_touching,
            "area_ratio": self.area_ratio,
            "solidity": self.solidity,
            "is_clean": self.is_clean,
        }


def contour_quality(contour: np.ndarray, mask: np.ndarray, crop_shape: Tuple[int, int]) -> QualityMetrics:
    """Score how trustworthy an extracted contour is.

    Args:
        contour: Nx2 contour points in the mask's own pixel coordinates.
        mask: The (uncleaned) binary mask the contour was derived from,
            used to count connected components.
        crop_shape: (height, width) of the crop the mask was computed on.

    Returns:
        The computed `QualityMetrics`.
    """
    height, width = crop_shape
    crop_area = float(height * width)

    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_areas = stats[1:n_components, cv2.CC_STAT_AREA] if n_components > 1 else np.array([])
    if len(component_areas) == 0:
        n_large_components = 0
    else:
        largest_area = float(component_areas.max())
        n_large_components = int(np.sum(component_areas >= LARGE_COMPONENT_AREA_FRAC * largest_area))

    border_touching = bool(
        np.any(contour[:, 0] <= BORDER_TOUCH_MARGIN_PX)
        or np.any(contour[:, 1] <= BORDER_TOUCH_MARGIN_PX)
        or np.any(contour[:, 0] >= width - 1 - BORDER_TOUCH_MARGIN_PX)
        or np.any(contour[:, 1] >= height - 1 - BORDER_TOUCH_MARGIN_PX)
    )

    contour_area = float(cv2.contourArea(contour.astype(np.float32)))
    area_ratio = contour_area / crop_area if crop_area > 0 else 0.0

    hull = cv2.convexHull(contour.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    solidity = contour_area / hull_area if hull_area > 0 else 0.0

    is_clean = (
        n_large_components == 1 and not border_touching and 0.05 <= area_ratio <= 0.9 and 0.6 <= solidity <= 0.995
    )

    return QualityMetrics(
        n_large_components=n_large_components,
        border_touching=border_touching,
        area_ratio=area_ratio,
        solidity=solidity,
        is_clean=is_clean,
    )
