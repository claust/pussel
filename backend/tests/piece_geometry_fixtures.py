"""Shared synthetic-piece helpers for piece-geometry tests.

Builds deterministic puzzle_shapes pieces (fixed `TabParameters()`, no
reliance on the global `random` module) and rasterizes them into masks and
painted BGR images, mirroring
`network/experiments/exp28_piece_geometry/synth_benchmark.py`'s
`rasterize_piece` closely enough to reuse for corner/edge ground truth.
"""

from io import BytesIO
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from puzzle_shapes import PieceConfig, TabParameters, generate_piece_path

PIECE_SIZE_PX = 500.0
CANVAS_PAD_FRAC = 0.15

BGRColor = Tuple[int, int, int]


def deterministic_config(edge_types: Sequence[str]) -> PieceConfig:
    """Build a PieceConfig with fixed (non-random) tab parameters.

    Args:
        edge_types: 4 edge types ("tab"/"blank"/"flat"), contour order
            [bottom, right, top, left] per `puzzle_shapes.generate_piece_geometry`.

    Returns:
        A `PieceConfig` that rasterizes identically on every call.
    """
    return PieceConfig(edge_types=list(edge_types), edge_params=[TabParameters()] * 4)


def rasterize_piece(config: PieceConfig, piece_size_px: float = PIECE_SIZE_PX) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize a piece config's path to a filled mask, with matching ground-truth corners.

    Args:
        config: The piece configuration to rasterize.
        piece_size_px: Pixel size of the base square's side.

    Returns:
        Tuple of (mask (H, W) uint8 in {0, 255}, ground-truth corners (4x2)
        in the SAME order as `generate_piece_geometry`'s base square:
        bottom-left, bottom-right, top-right, top-left, in pixel coordinates).
    """
    x, y = generate_piece_path(config)
    path = np.column_stack([x, y])
    gt_local = np.array(
        [[0.0, 0.0], [config.size, 0.0], [config.size, config.size], [0.0, config.size]],
    )

    scale = piece_size_px / config.size
    pad = CANVAS_PAD_FRAC * config.size

    min_xy = path.min(axis=0)
    max_y = path[:, 1].max()

    def to_pixels(pts: np.ndarray) -> np.ndarray:
        px = (pts[:, 0] - min_xy[0] + pad) * scale
        py = (max_y - pts[:, 1] + pad) * scale
        return np.column_stack([px, py])

    path_px = to_pixels(path)
    gt_px = to_pixels(gt_local)

    extent = path.max(axis=0) - min_xy
    canvas_w = int(np.ceil((extent[0] + 2 * pad) * scale))
    canvas_h = int(np.ceil((extent[1] + 2 * pad) * scale))

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillPoly(mask, [path_px.astype(np.int32)], (255,))
    return mask, gt_px


def paint_quadrants(mask: np.ndarray, colors: Sequence[BGRColor]) -> np.ndarray:
    """Paint a BGR image with 4 distinct colors over the mask's bbox quadrants.

    Distinct per-quadrant colors give the 3x3 spatial color descriptor real
    signal to distinguish pieces on (a flat single-color fill gray-world-
    normalizes to the same neutral color regardless of hue, which would make
    every piece's spatial descriptor identical).

    Args:
        mask: Binary mask (H, W) with values in {0, 255}.
        colors: 4 BGR colors for (top-left, top-right, bottom-left, bottom-right).

    Returns:
        A BGR image (H, W, 3): background gray, piece region painted by quadrant.
    """
    height, width = mask.shape
    image = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
    half_h, half_w = height // 2, width // 2
    image[:half_h, :half_w] = colors[0]
    image[:half_h, half_w:] = colors[1]
    image[half_h:, :half_w] = colors[2]
    image[half_h:, half_w:] = colors[3]
    return image


def rgba_from_mask_and_image(mask: np.ndarray, image_bgr: np.ndarray) -> Image.Image:
    """Build a PIL RGBA image from a binary mask and a BGR color image, mimicking rembg output.

    Args:
        mask: Binary mask (H, W) with values in {0, 255}, used as the alpha channel.
        image_bgr: BGR color image (H, W, 3), same shape as `mask`.

    Returns:
        An RGBA `PIL.Image.Image` with `image_bgr`'s colors and `mask` as alpha.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, mask])
    return Image.fromarray(rgba, mode="RGBA")


def encode_png(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes.

    Args:
        image: The image to encode.

    Returns:
        PNG-encoded bytes.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# Two visually and geometrically distinct pieces, used across fingerprint/store/API tests.
PIECE_A_EDGE_TYPES: List[str] = ["tab", "blank", "flat", "tab"]
PIECE_A_COLORS: List[BGRColor] = [(80, 120, 200), (20, 200, 20), (200, 20, 20), (20, 20, 200)]

PIECE_B_EDGE_TYPES: List[str] = ["blank", "tab", "tab", "blank"]
PIECE_B_COLORS: List[BGRColor] = [(200, 20, 20), (20, 20, 200), (80, 120, 200), (20, 200, 20)]


def embed_in_frame(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    frame_shape: Tuple[int, int],
    offset: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Paste a piece mask + color image into a larger (camera-frame-sized) canvas.

    Simulates a real capture where the piece covers only part of the frame,
    exercising the service's largest-component crop step.

    Args:
        mask: Piece binary mask (h, w) in {0, 255}.
        image_bgr: Piece BGR color image (h, w, 3), same shape as `mask`.
        frame_shape: (frame_height, frame_width) of the output canvas.
        offset: (x, y) top-left position to paste the piece at.

    Returns:
        Tuple of (frame_mask (H, W), frame_bgr (H, W, 3)): transparent gray
        frame with the piece pasted at `offset`.
    """
    frame_h, frame_w = frame_shape
    piece_h, piece_w = mask.shape
    x, y = offset
    frame_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    frame_bgr = np.full((frame_h, frame_w, 3), (128, 128, 128), dtype=np.uint8)
    frame_mask[y : y + piece_h, x : x + piece_w] = mask
    frame_bgr[y : y + piece_h, x : x + piece_w] = image_bgr
    return frame_mask, frame_bgr


def make_piece_rgba(edge_types: Sequence[str], colors: Sequence[BGRColor]) -> Image.Image:
    """Build a full synthetic piece photo (RGBA, alpha = piece mask) end to end.

    Args:
        edge_types: 4 edge types for `deterministic_config`.
        colors: 4 quadrant BGR colors for `paint_quadrants`.

    Returns:
        The synthetic piece as an RGBA `PIL.Image.Image`.
    """
    config = deterministic_config(edge_types)
    mask, _ = rasterize_piece(config)
    image_bgr = paint_quadrants(mask, colors)
    return rgba_from_mask_and_image(mask, image_bgr)
