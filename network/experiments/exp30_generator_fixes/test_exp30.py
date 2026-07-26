"""Unit tests for the exp30 generator fixes.

These lock in the four properties that make exp30 different from exp26,
using tiny in-memory fixtures only (no generated dataset required):

1. ``rotate_lossless`` matches the exp20/exp26 **clockwise** label
   convention (``image.rotate(-rot, expand=True)``) on an asymmetric
   pattern — PIL's ``Transpose.ROTATE_90`` is counterclockwise, so getting
   this backwards would silently mislabel every piece.
2. Rotation loses no content: the opaque-pixel count is invariant across
   all four rotations (the exp26 ``expand=False`` path destroys ~30-50% of
   the tab silhouette at 90deg/270deg).
3. ``frame_rgba``'s margin/pad geometry equals exp24's real preparation
   path (``rgba_to_classifier_input``) pixel-for-pixel.
4. The load path's output never has opaque content touching the input
   border, for any rotation — the shortcut from section 4.1 is gone.

Run from the network/ directory:
    uv run pytest experiments/exp30_generator_fixes/test_exp30.py -q
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ..exp20_realistic_pieces.dataset import ROTATION_ANGLES
from ..exp24_piece_classifier.data_prep import CROP_MARGIN, rgba_to_classifier_input
from .framed_dataset import FramedBlackCompositeTestDataset, piece_to_model_input
from .framing import frame_and_black_composite, frame_rgba, rotate_lossless

ALL_ROTATIONS = [0, 90, 180, 270]


def make_piece(width: int = 41, height: int = 67) -> Image.Image:
    """Build a small asymmetric RGBA "piece" cropped tight to its silhouette.

    The shape is a rounded blob with one protruding tab, so no rotation of
    it can be confused with another, and its RGB carries a colour gradient
    so pixel comparisons are meaningful.

    Args:
        width: Silhouette width in pixels.
        height: Silhouette height in pixels.

    Returns:
        An RGBA image cropped tight to the opaque silhouette.
    """
    piece = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(20, 240, width, dtype=np.uint8)[None, :]
    gradient[:, :, 1] = np.linspace(240, 20, height, dtype=np.uint8)[:, None]
    gradient[:, :, 2] = 128
    rgb = Image.fromarray(gradient, mode="RGB")

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((0, 6, width - 1, height - 1), fill=255)
    # One asymmetric tab poking out of the top-left quadrant.
    draw.ellipse((2, 0, 16, 14), fill=255)
    piece.paste(rgb, mask=mask)
    return piece.crop(piece.getbbox() or (0, 0, width, height))


def opaque_count(image: Image.Image, threshold: int = 128) -> int:
    """Count pixels whose alpha exceeds ``threshold``.

    Args:
        image: RGBA image.
        threshold: Alpha threshold.

    Returns:
        Number of opaque pixels.
    """
    return int((np.asarray(image.getchannel("A")) > threshold).sum())


def border_alpha_max(image: Image.Image) -> int:
    """Return the maximum alpha value on the one-pixel border of an image.

    Args:
        image: RGBA image.

    Returns:
        The largest alpha found on the outermost row/column ring.
    """
    alpha = np.asarray(image.getchannel("A"))
    ring = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
    return int(ring.max())


@pytest.mark.parametrize("rotation", ALL_ROTATIONS)
def test_rotate_lossless_matches_clockwise_convention(rotation: int) -> None:
    """rotate_lossless equals exp20's ``rotate(-rot, expand=True)``.

    Args:
        rotation: Clockwise rotation under test.
    """
    piece = make_piece()
    expected = piece.rotate(-rotation, expand=True, resample=Image.Resampling.NEAREST, fillcolor=(0, 0, 0, 0))
    actual = rotate_lossless(piece, rotation)

    assert actual.size == expected.size
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_rotate_lossless_swaps_dimensions_for_odd_rotations() -> None:
    """90deg/270deg swap the canvas dimensions instead of clipping content."""
    piece = make_piece(width=41, height=67)
    assert rotate_lossless(piece, 0).size == piece.size
    assert rotate_lossless(piece, 180).size == piece.size
    assert rotate_lossless(piece, 90).size == (piece.height, piece.width)
    assert rotate_lossless(piece, 270).size == (piece.height, piece.width)


def test_rotate_lossless_rejects_non_multiples_of_90() -> None:
    """A non-multiple-of-90 rotation is a programming error, not a silent warp."""
    with pytest.raises(ValueError, match="multiple of 90"):
        rotate_lossless(make_piece(), 45)


def test_rotation_preserves_all_content() -> None:
    """No opaque pixel is lost at any rotation (exp26's expand=False loses ~30-50%)."""
    piece = make_piece()
    baseline = opaque_count(piece)
    assert baseline > 0

    counts = {rot: opaque_count(rotate_lossless(piece, rot)) for rot in ALL_ROTATIONS}
    assert set(counts.values()) == {baseline}, counts

    # The bug being fixed: the exp26 path really does destroy content here.
    clipped = piece.rotate(-90, expand=False, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))
    assert opaque_count(clipped) < baseline


def test_framing_matches_exp24_real_path_geometry() -> None:
    """frame_rgba + black composite reproduces exp24's rgba_to_classifier_input.

    The real path receives a piece sitting inside a larger frame (a photo),
    so its 8% margin comes out of the surrounding pixels; the synthetic
    path receives a tight crop and must *add* the margin. Both must land on
    the same square image.
    """
    piece = make_piece()
    # A "photo"-like RGBA: the same piece with generous transparent surround.
    photo = Image.new("RGBA", (piece.width + 60, piece.height + 60), (0, 0, 0, 0))
    photo.paste(piece, (30, 30))

    real_path = rgba_to_classifier_input(photo, margin=CROP_MARGIN)
    assert real_path is not None
    synthetic_path = frame_and_black_composite(piece, margin=CROP_MARGIN)

    assert synthetic_path.size == real_path.size
    assert synthetic_path.width == synthetic_path.height
    assert np.array_equal(np.asarray(synthetic_path), np.asarray(real_path))


def test_framing_is_square_and_rotation_symmetric() -> None:
    """All four rotations produce the same square canvas size and pixel count."""
    piece = make_piece()
    framed = {rot: frame_rgba(rotate_lossless(piece, rot)) for rot in ALL_ROTATIONS}

    sizes = {img.size for img in framed.values()}
    assert len(sizes) == 1, sizes
    side = framed[0].width
    assert framed[0].height == side

    counts = {opaque_count(img) for img in framed.values()}
    assert len(counts) == 1, counts


@pytest.mark.parametrize("rotation", ALL_ROTATIONS)
def test_framed_piece_never_touches_the_border(rotation: int) -> None:
    """The section 4.1 shortcut is gone: content never reaches the input border.

    Checked on the framed canvas and again after the model's 128x128
    resize, since that is the tensor the network actually sees.

    Args:
        rotation: Clockwise rotation under test.
    """
    piece = make_piece()
    framed = frame_rgba(rotate_lossless(piece, rotation))
    assert border_alpha_max(framed) == 0

    resized = framed.resize((128, 128), Image.Resampling.BILINEAR)
    assert border_alpha_max(resized) == 0


def test_exp26_scheme_leaks_the_rotation_label() -> None:
    """Regression guard: the old scheme really did leak, so the fix is load-bearing.

    Reproduces exp26's ``rotate(expand=False)`` on a tight non-square crop
    and shows the border-touch signature splits cleanly by rotation parity.
    """
    piece = make_piece(width=41, height=67)
    touches_all_borders: dict[int, bool] = {}
    for rot in ALL_ROTATIONS:
        legacy = (
            piece.copy()
            if rot == 0
            else piece.rotate(-rot, expand=False, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))
        )
        alpha = np.asarray(legacy.getchannel("A")) > 128
        touches_all_borders[rot] = bool(
            alpha[0, :].any() and alpha[-1, :].any() and alpha[:, 0].any() and alpha[:, -1].any()
        )

    assert touches_all_borders[0] and touches_all_borders[180]
    assert not touches_all_borders[90] and not touches_all_borders[270]


@pytest.mark.parametrize("rotation_idx", [0, 1, 2, 3])
def test_model_input_hook_matches_the_eval_dataset(tmp_path: Path, rotation_idx: int) -> None:
    """The ``piece_to_model_input`` hook reproduces the eval dataset's piece branch.

    ``probes.py`` reads the exp30 eval view through this hook, so it must
    not drift from what ``FramedBlackCompositeTestDataset`` actually feeds
    the model.

    Args:
        tmp_path: pytest temporary directory.
        rotation_idx: Applied rotation index under test.
    """
    piece_path = tmp_path / "puzzle_test_x0.125_y0.125_rot0.png"
    make_piece().save(piece_path, "PNG")

    hook_output = piece_to_model_input(piece_path, rotation_idx, piece_size=128)
    assert hook_output.size == (128, 128)

    dataset = FramedBlackCompositeTestDataset.__new__(FramedBlackCompositeTestDataset)
    with Image.open(piece_path) as raw:
        loaded = raw.convert("RGBA")
    dataset_output = dataset._rotate_piece(loaded, rotation_idx).resize((128, 128), Image.Resampling.BILINEAR)

    assert np.array_equal(np.asarray(hook_output), np.asarray(dataset_output.convert("RGB")))


def test_eval_rotation_labels_compose_like_exp20() -> None:
    """The framed eval transform still composes base + applied rotation correctly."""
    piece = make_piece()
    for base in ALL_ROTATIONS:
        baked = rotate_lossless(piece, base)
        for applied_idx, applied in enumerate(ROTATION_ANGLES):
            total = (base + applied) % 360
            assert total // 90 == (base // 90 + applied_idx) % 4
            # Applying base then applied equals applying the total in one go.
            assert np.array_equal(
                np.asarray(rotate_lossless(baked, applied)),
                np.asarray(rotate_lossless(piece, total)),
            )
