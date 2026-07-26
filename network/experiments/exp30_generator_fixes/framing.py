"""Lossless 90-degree rotation and real-path framing — the whole exp30 fix.

Two label-leaking generator bugs and one framing mismatch are fixed here
(see ``docs/synthetic-dataset-realism.html`` sections 4.1, 4.2 and 7):

1. **Border-touch rotation shortcut (4.1).** ``cut_piece`` crops tight to
   the silhouette, and exp20/exp26 then rotate that *non-square* canvas
   with ``expand=False``. 0deg/180deg content fills the canvas (alpha
   touches all four borders) while 90deg/270deg content is clipped, losing
   ~60 px of tabs and leaving black bands. "Touches all four borders =>
   rotation in {0, 180}" is a 100%-precision rule in synthetic train *and*
   test data, and real crops never touch a border at all.
2. **Half-pixel blur cue (4.2).** ``rotate(expand=False, BILINEAR)`` on a
   non-square canvas puts 90deg/270deg on half-pixel offsets, making them
   ~32% blurrier than the pixel-exact 0deg/180deg. Sharpness alone leaks
   the rotation label.
3. **Aspect/framing mismatch (7, Test 1).** Training resized the tight
   crop straight to 128x128 (squashing aspect); deployment pads to square
   with an 8% margin first. They must agree.

The fixes:

- :func:`rotate_lossless` replaces every multiple-of-90 rotation with
  ``Image.transpose`` — an exact pixel permutation. Dimensions swap
  naturally, so nothing is clipped and nothing is resampled.
- :func:`frame_rgba` reproduces exp24's real preparation geometry
  (largest opaque component -> 8% margin -> pad to square) on the RGBA
  piece, so every rotation lands on a square canvas with a transparent
  margin. No rotation can touch the input border, and the 128x128 resize
  no longer squashes aspect.

Because :func:`frame_rgba` yields a **square, rotation-symmetric** canvas,
everything downstream of it (exp26's ``augment_piece``, the background
composite, the resize) is statistically identical across the four rotation
classes — that is what makes the fix airtight rather than merely narrower.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..exp24_piece_classifier.data_prep import ALPHA_THRESHOLD, CROP_MARGIN, largest_component_bbox

# Clockwise degrees -> PIL transpose op. PIL's ``Transpose.ROTATE_90`` is
# COUNTERCLOCKWISE, while exp20/exp26 label rotations CLOCKWISE (they call
# ``image.rotate(-rotation)``), so the mapping is deliberately crossed:
# clockwise 90deg == Transpose.ROTATE_270. ``test_exp30.py`` proves this
# against ``image.rotate(-rot, expand=True)`` on an asymmetric pattern.
CLOCKWISE_TRANSPOSE: dict[int, Image.Transpose] = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def rotate_lossless(image: Image.Image, rotation: int) -> Image.Image:
    """Rotate an image clockwise by a multiple of 90 degrees, losslessly.

    Uses ``Image.transpose``, an exact pixel permutation: no interpolation
    (so no rotation-correlated blur cue) and no clipping (the canvas
    dimensions swap for 90deg/270deg instead of the content being cut).

    Args:
        image: Image to rotate (any mode).
        rotation: Clockwise rotation in degrees; must be 0, 90, 180 or 270.

    Returns:
        The rotated image (a copy for ``rotation == 0``).

    Raises:
        ValueError: If ``rotation`` is not a multiple of 90 in [0, 360).
    """
    normalized = rotation % 360
    if normalized == 0:
        return image.copy()
    transpose = CLOCKWISE_TRANSPOSE.get(normalized)
    if transpose is None:
        raise ValueError(f"rotation must be a multiple of 90 degrees, got {rotation}")
    return image.transpose(transpose)


def alpha_bbox(piece_rgba: Image.Image, alpha_threshold: int = ALPHA_THRESHOLD) -> tuple[int, int, int, int] | None:
    """Return the (x, y, w, h) bbox of the piece's largest opaque component.

    Delegates to exp24's ``largest_component_bbox`` so the synthetic path
    and the deployed real path agree on what "the subject" is, down to the
    5x5 morphological close and the alpha threshold.

    Args:
        piece_rgba: RGBA piece image.
        alpha_threshold: Alpha value above which a pixel counts as opaque.

    Returns:
        Bounding box (x, y, w, h), or None when nothing is opaque.
    """
    alpha = np.asarray(piece_rgba.getchannel("A"))
    mask = (alpha > alpha_threshold).astype(np.uint8) * 255
    return largest_component_bbox(mask)


def frame_rgba(piece_rgba: Image.Image, margin: float = CROP_MARGIN) -> Image.Image:
    """Frame an RGBA piece exactly as the real deployment path frames a photo.

    Crops to the largest opaque component, **expands** by ``margin`` on
    every side and pads the result to a square with transparent fill. This
    is the RGBA-preserving twin of exp24's
    ``pad_to_square(_crop_with_margin(...))``: black-compositing the output
    reproduces ``rgba_to_classifier_input`` pixel-for-pixel whenever the
    input has room for the margin. Unlike exp24's version the margin is
    *added* rather than clamped to the source bounds, because generator
    pieces arrive cropped tight to the silhouette with no room to spare.

    Alpha is preserved so the training path can still composite the piece
    onto an arbitrary background (exp26 domain randomization); the margin
    simply takes on the background colour.

    Args:
        piece_rgba: RGBA piece image (alpha = true piece mask).
        margin: Extra margin per side as a fraction of the bbox size.

    Returns:
        A square RGBA image whose opaque content never reaches the border.
    """
    piece = piece_rgba if piece_rgba.mode == "RGBA" else piece_rgba.convert("RGBA")
    bbox = alpha_bbox(piece)
    if bbox is None:
        # Fully transparent piece: keep the canvas, just square it off.
        crop = piece
    else:
        x, y, w, h = bbox
        crop = piece.crop((x, y, x + w, y + h))

    width, height = crop.size
    pad_x = round(width * margin)
    pad_y = round(height * margin)
    side = max(width + 2 * pad_x, height + 2 * pad_y)

    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - width) // 2, (side - height) // 2))
    return canvas


def frame_and_black_composite(piece_rgba: Image.Image, margin: float = CROP_MARGIN) -> Image.Image:
    """Frame an RGBA piece and composite it on black — the eval/deploy input.

    Args:
        piece_rgba: RGBA piece image.
        margin: Extra margin per side as a fraction of the bbox size.

    Returns:
        A square RGB image with the piece centred on black, matching what
        ``exp24_piece_classifier.data_prep.rgba_to_classifier_input``
        produces for a real rembg cut-out.
    """
    framed = frame_rgba(piece_rgba, margin=margin)
    black = Image.new("RGB", framed.size, (0, 0, 0))
    black.paste(framed, mask=framed.getchannel("A"))
    return black


__all__ = [
    "CLOCKWISE_TRANSPOSE",
    "CROP_MARGIN",
    "alpha_bbox",
    "frame_and_black_composite",
    "frame_rgba",
    "rotate_lossless",
]
