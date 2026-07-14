"""Shared image-preparation helpers for the piece classifier.

These implement the single input protocol used by both the training data
builders and the backend's live inference: take the segmented subject
(rembg RGBA output, or a generator piece on a black canvas), crop it to
the bounding box of its largest connected component with a small margin,
composite it on black, and pad the result to a square.

The backend mirrors this protocol in ``app/services/piece_classifier.py``;
change both places together.
"""

from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Margin around the subject bounding box, as a fraction of its size.
CROP_MARGIN = 0.08

# Alpha value above which a pixel counts as opaque (rembg output).
ALPHA_THRESHOLD = 128

# Grayscale value above which a pixel counts as content on a black canvas
# (generator output where transparency was already filled with black).
BLACK_BG_THRESHOLD = 10


def largest_component_bbox(mask: "np.ndarray") -> Optional[tuple[int, int, int, int]]:
    """Find the bounding box of the largest connected component in a binary mask.

    Args:
        mask: 2D uint8 array where nonzero marks subject pixels.

    Returns:
        Bounding box (x, y, w, h), or None when the mask is empty.
    """
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an RGB image with black to a square canvas, keeping it centered.

    Args:
        image: RGB image of any aspect ratio.

    Returns:
        A square RGB image.
    """
    side = max(image.width, image.height)
    if image.width == side and image.height == side:
        return image
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def _crop_with_margin(image: Image.Image, bbox: tuple[int, int, int, int], margin: float) -> Image.Image:
    """Crop an image to a bounding box expanded by a relative margin.

    Args:
        image: The image to crop.
        bbox: Bounding box (x, y, w, h).
        margin: Extra margin as a fraction of the box size.

    Returns:
        The cropped image.
    """
    x, y, w, h = bbox
    pad_x = round(w * margin)
    pad_y = round(h * margin)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image.width, x + w + pad_x)
    bottom = min(image.height, y + h + pad_y)
    return image.crop((left, top, right, bottom))


def rgba_to_classifier_input(rgba: Image.Image, margin: float = CROP_MARGIN) -> Optional[Image.Image]:
    """Convert a rembg RGBA output into the classifier's square black-backed crop.

    Args:
        rgba: RGBA image where alpha marks the segmented subject.
        margin: Extra margin around the subject bounding box.

    Returns:
        A square RGB image with the subject composited on black, or None when
        the image has no opaque region.
    """
    if rgba.mode != "RGBA":
        return None
    alpha = np.asarray(rgba.split()[3])
    mask = (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255
    bbox = largest_component_bbox(mask)
    if bbox is None:
        return None
    black = Image.new("RGB", rgba.size, (0, 0, 0))
    black.paste(rgba, mask=rgba.split()[3])
    return pad_to_square(_crop_with_margin(black, bbox, margin))


def black_canvas_to_classifier_input(image: Image.Image, margin: float = CROP_MARGIN) -> Optional[Image.Image]:
    """Convert a piece-on-black-canvas image into the classifier's square crop.

    Used for generator output (exp20 realistic pieces) where transparency has
    already been filled with black, so content is detected by brightness.

    Args:
        image: RGB image with the subject on a black background.
        margin: Extra margin around the subject bounding box.

    Returns:
        A square RGB image, or None when the image is entirely black.
    """
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    mask = (gray > BLACK_BG_THRESHOLD).astype(np.uint8) * 255
    bbox = largest_component_bbox(mask)
    if bbox is None:
        return None
    return pad_to_square(_crop_with_margin(rgb, bbox, margin))
