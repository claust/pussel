"""Domain-randomization augmentations for exp26.

exp25 showed the exp20 CNN collapses on real photos (72.2% -> 14.8%
both-correct) because it learned pixel-identical template lookup: the
synthetic "piece" is a byte-exact crop of the very image fed as the
puzzle, so matching by raw pixels is enough. A photographed piece shares
*content* with the box art but not *pixels* (different camera, lighting,
white balance, scale, perspective, sensor noise, JPEG, and an imperfect
segmentation mask). This module rewrites the training piece so that the
pixel shortcut no longer works and the model is forced to learn
appearance-invariant matching.

Design goals:

- **Every augmentation is individually toggleable** (``AugmentConfig``
  booleans) so an ablation can turn each one off and measure its effect.
- **Photometric jitter is drawn independently for piece and puzzle.**
  This is the single most important lever: identical jitter would keep
  the pixel-identity shortcut intact, independent jitter destroys it.
- Pieces are consumed as **RGBA** (alpha = true piece mask) so a piece
  can be composited onto an arbitrary background; the deployed backend
  black-composites via rembg, so a configurable fraction stays black.

The augmentations operate on PIL images and return PIL images; tensor
conversion/resize happens in the dataset, exactly as in exp20.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter, RandomPerspective
from torchvision.transforms import functional as TF


@dataclass
class AugmentConfig:
    """Toggleable configuration for the exp26 realism augmentations.

    Each boolean turns one augmentation on or off; the numeric fields set
    its strength. Turning a flag off (e.g. ``photometric=False``) is how
    the ablation isolates that augmentation's contribution. ``enabled``
    is the master switch — with it False the piece is simply
    black-composited, reproducing the exp20 training appearance.
    """

    enabled: bool = True

    # --- Independent photometric jitter (piece and puzzle drawn separately) ---
    # Kept moderate on purpose: strong brightness/contrast clips light or
    # dark pieces to a featureless blob, destroying the very content the
    # model must match on. The goal is to vary lighting, not erase identity.
    photometric: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.04
    # Puzzle jitter is milder: the box art is one fixed photo, the piece
    # is the thing photographed under varying conditions.
    puzzle_photometric_scale: float = 0.5

    # --- Geometry on the piece ---
    scale_jitter: bool = True
    scale_min: float = 0.85
    scale_max: float = 1.15

    perspective: bool = True
    perspective_distortion: float = 0.2
    perspective_p: float = 0.4

    rotation_jitter: bool = True
    rotation_jitter_deg: float = 8.0

    # --- Background compositing (needs RGBA piece) ---
    # Weighted toward black so the majority of training pieces look like the
    # deployed rembg output; the coloured/textured modes are the minority
    # that teach robustness to segmentation leakage.
    background: bool = True
    bg_black_weight: float = 0.6
    bg_solid_weight: float = 0.15
    bg_gradient_weight: float = 0.1
    bg_texture_weight: float = 0.15

    # Mask erosion/dilation before compositing simulates rembg mask error
    # (eroded = piece edge eaten, dilated = background ring bleeds in).
    halo: bool = True
    halo_p: float = 0.3
    halo_max_px: int = 3

    # --- Sensor / codec ---
    noise: bool = True
    noise_std: float = 0.02

    jpeg: bool = True
    jpeg_quality_min: int = 55
    jpeg_quality_max: int = 95
    jpeg_p: float = 0.6

    # --- Legacy exp20 augmentations (kept for continuity, off by default) ---
    grayscale_p: float = 0.0
    blur: bool = False
    blur_sigma_max: float = 1.0

    def ablation_flags(self) -> dict[str, bool]:
        """Return the current on/off state of every toggleable augmentation.

        Returns:
            Mapping of augmentation name to whether it is enabled. Written
            into the results JSON so an ablation run is self-describing.
        """
        names = (
            "enabled",
            "photometric",
            "scale_jitter",
            "perspective",
            "rotation_jitter",
            "background",
            "halo",
            "noise",
            "jpeg",
            "blur",
        )
        return {name: bool(getattr(self, name)) for name in names}


# Preset configs for the ablation. Each disables exactly one augmentation
# family relative to ``full`` so its marginal effect is measurable.
def _ablation_presets() -> dict[str, AugmentConfig]:
    """Build the named ablation presets used by ``--aug-preset``.

    Returns:
        Mapping of preset name to an ``AugmentConfig``.
    """
    presets: dict[str, AugmentConfig] = {
        "full": AugmentConfig(),
        "none": AugmentConfig(enabled=False),
        "no_photometric": AugmentConfig(photometric=False),
        "no_geometry": AugmentConfig(scale_jitter=False, perspective=False, rotation_jitter=False),
        "no_background": AugmentConfig(background=False),
        "no_sensor": AugmentConfig(noise=False, jpeg=False),
        "photometric_only": AugmentConfig(
            scale_jitter=False,
            perspective=False,
            rotation_jitter=False,
            background=False,
            noise=False,
            jpeg=False,
        ),
    }
    return presets


AUG_PRESETS = _ablation_presets()


class BackgroundSampler:
    """Samples realistic backgrounds to composite behind a piece.

    Backgrounds are procedural (solid colour, two-colour gradient) or
    natural (a random crop of another source puzzle image). Using other
    puzzles as texture keeps the augmentation self-contained (no external
    texture download) and — importantly — never touches the north-star
    photos, so the real-photo benchmark stays untouched.
    """

    def __init__(self, texture_paths: list[Path] | None = None, cache_size: int = 64) -> None:
        """Initialize the sampler.

        Args:
            texture_paths: Source images used for the "texture" background
                mode (typically the training puzzle JPEGs). If empty, the
                texture mode falls back to a gradient.
            cache_size: Number of decoded texture images to keep in memory.
        """
        self.texture_paths = texture_paths or []
        self.cache_size = cache_size
        self._cache: dict[Path, Image.Image] = {}

    def _load_texture(self, path: Path) -> Image.Image:
        """Load and cache a texture source image (RGB)."""
        cached = self._cache.get(path)
        if cached is None:
            with Image.open(path) as img:
                cached = img.convert("RGB")
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = cached
        return cached

    def solid(self, size: tuple[int, int]) -> Image.Image:
        """Return a solid random-colour background of ``size`` (w, h)."""
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return Image.new("RGB", size, colour)

    def gradient(self, size: tuple[int, int]) -> Image.Image:
        """Return a smooth two-colour linear gradient background."""
        width, height = size
        c0 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        if random.random() < 0.5:
            ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        else:
            ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        arr = c0[None, None, :] * (1.0 - ramp) + c1[None, None, :] * ramp
        arr = np.broadcast_to(arr, (height, width, 3))
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    def texture(self, size: tuple[int, int]) -> Image.Image:
        """Return a random crop of a random source puzzle, resized to ``size``.

        Falls back to a gradient when no texture paths are configured or
        the chosen image fails to load.
        """
        if not self.texture_paths:
            return self.gradient(size)
        path = random.choice(self.texture_paths)
        try:
            src = self._load_texture(path)
        except (OSError, ValueError):
            return self.gradient(size)

        width, height = size
        sw, sh = src.size
        crop_w = min(sw, max(width, sw // 3))
        crop_h = min(sh, max(height, sh // 3))
        left = random.randint(0, max(0, sw - crop_w))
        top = random.randint(0, max(0, sh - crop_h))
        crop = src.crop((left, top, left + crop_w, top + crop_h))
        return crop.resize(size, Image.Resampling.BILINEAR)

    def sample(self, size: tuple[int, int], config: AugmentConfig) -> Image.Image | None:
        """Sample a background image, or None for the black-composite mode.

        Args:
            size: Target (width, height).
            config: Active augmentation config (for the mode weights).

        Returns:
            An RGB background image, or None meaning "composite on black"
            (the deployed rembg appearance).
        """
        modes = ("black", "solid", "gradient", "texture")
        weights = (
            config.bg_black_weight,
            config.bg_solid_weight,
            config.bg_gradient_weight,
            config.bg_texture_weight,
        )
        if sum(weights) <= 0:
            return None
        mode = random.choices(modes, weights=weights, k=1)[0]
        if mode == "black":
            return None
        if mode == "solid":
            return self.solid(size)
        if mode == "gradient":
            return self.gradient(size)
        return self.texture(size)


def _color_jitter_image(img: Image.Image, config: AugmentConfig, scale: float) -> Image.Image:
    """Apply one independent draw of brightness/contrast/saturation/hue.

    Args:
        img: RGB image to jitter.
        config: Active config (jitter magnitudes).
        scale: Multiplier on all magnitudes (puzzle uses a milder scale).

    Returns:
        The jittered RGB image.
    """
    jitter = ColorJitter(
        brightness=max(0.0, config.brightness * scale),
        contrast=max(0.0, config.contrast * scale),
        saturation=max(0.0, config.saturation * scale),
        hue=max(0.0, config.hue * scale),
    )
    out = jitter(img)
    assert isinstance(out, Image.Image)
    return out


def _perturb_alpha(piece: Image.Image, config: AugmentConfig) -> Image.Image:
    """Randomly erode or dilate the alpha mask to mimic segmentation error.

    Args:
        piece: RGBA piece.
        config: Active config (halo settings).

    Returns:
        RGBA piece with a perturbed alpha channel (RGB untouched).
    """
    radius = random.randint(1, max(1, config.halo_max_px))
    size = 2 * radius + 1
    alpha = piece.getchannel("A")
    if random.random() < 0.5:
        alpha = alpha.filter(ImageFilter.MaxFilter(size))  # dilate: background ring bleeds in
    else:
        alpha = alpha.filter(ImageFilter.MinFilter(size))  # erode: piece edge eaten
    out = piece.copy()
    out.putalpha(alpha)
    return out


def _rotate_rgba(piece: Image.Image, degrees: float) -> Image.Image:
    """Rotate an RGBA piece about its centre with a transparent fill.

    ``expand=True`` grows the canvas so tab protrusions are never clipped
    at the corners (the geometry bug called out in CRITICAL_REVIEW.md #7).

    Args:
        piece: RGBA piece.
        degrees: Signed rotation in degrees (CCW positive, PIL convention).

    Returns:
        The rotated RGBA piece on a possibly larger transparent canvas.
    """
    return piece.rotate(
        degrees,
        expand=True,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0, 0),
    )


def _scale_rgba(piece: Image.Image, factor: float) -> Image.Image:
    """Scale an RGBA piece by ``factor`` (alpha scaled with it)."""
    w, h = piece.size
    new_size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return piece.resize(new_size, Image.Resampling.BILINEAR)


def _perspective_rgba(piece: Image.Image, config: AugmentConfig) -> Image.Image:
    """Apply a mild random perspective warp to an RGBA piece.

    Args:
        piece: RGBA piece.
        config: Active config (distortion scale).

    Returns:
        The warped RGBA piece (transparent fill outside the warped quad).
    """
    w, h = piece.size
    start, end = RandomPerspective.get_params(w, h, config.perspective_distortion)
    warped = TF.perspective(
        piece,
        start,
        end,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=[0, 0, 0, 0],
    )
    assert isinstance(warped, Image.Image)
    return warped


def _composite(piece: Image.Image, background: Image.Image | None) -> Image.Image:
    """Composite an RGBA piece onto a background (or black), returning RGB.

    Args:
        piece: RGBA piece.
        background: RGB background of the same size, or None for black.

    Returns:
        The composited RGB image.
    """
    if background is None:
        background = Image.new("RGB", piece.size, (0, 0, 0))
    elif background.size != piece.size:
        background = background.resize(piece.size, Image.Resampling.BILINEAR)
    base = background.convert("RGB").copy()
    base.paste(piece, mask=piece.getchannel("A"))
    return base


def _add_noise(img: Image.Image, std: float) -> Image.Image:
    """Add zero-mean Gaussian sensor noise to an RGB image."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def _jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode an RGB image as JPEG at ``quality`` to inject block artifacts."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def black_composite(piece_rgba: Image.Image) -> Image.Image:
    """Black-composite an RGBA piece to RGB (the exp20 / deployed appearance).

    Used for the un-augmented path and for val/test so those stay
    byte-comparable to the exp20 frozen benchmark.

    Args:
        piece_rgba: RGBA piece.

    Returns:
        RGB piece on a black background.
    """
    return _composite(piece_rgba, None)


def _augment_geometry(piece: Image.Image, config: AugmentConfig) -> Image.Image:
    """Apply the geometric augmentations to an RGBA piece.

    Args:
        piece: RGBA piece.
        config: Active config.

    Returns:
        The geometrically augmented RGBA piece (rotation jitter ->
        perspective -> scale -> mask erode/dilate).
    """
    if config.rotation_jitter and config.rotation_jitter_deg > 0:
        piece = _rotate_rgba(piece, random.uniform(-config.rotation_jitter_deg, config.rotation_jitter_deg))
    if config.perspective and random.random() < config.perspective_p:
        piece = _perspective_rgba(piece, config)
    if config.scale_jitter:
        piece = _scale_rgba(piece, random.uniform(config.scale_min, config.scale_max))
    if config.halo and random.random() < config.halo_p:
        piece = _perturb_alpha(piece, config)
    return piece


def _augment_appearance(rgb: Image.Image, config: AugmentConfig) -> Image.Image:
    """Apply the photometric / sensor augmentations to a composited RGB image.

    Args:
        rgb: The composited RGB piece.
        config: Active config.

    Returns:
        The RGB piece after photometric jitter, optional grayscale/blur,
        sensor noise and JPEG recompression.
    """
    if config.photometric:
        rgb = _color_jitter_image(rgb, config, scale=1.0)
    if config.grayscale_p > 0 and random.random() < config.grayscale_p:
        gray = TF.to_grayscale(rgb, num_output_channels=3)
        assert isinstance(gray, Image.Image)
        rgb = gray
    if config.blur and random.random() < 0.5:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, config.blur_sigma_max)))
    if config.noise and config.noise_std > 0:
        rgb = _add_noise(rgb, config.noise_std)
    if config.jpeg and random.random() < config.jpeg_p:
        rgb = _jpeg_recompress(rgb, random.randint(config.jpeg_quality_min, config.jpeg_quality_max))
    return rgb


def augment_piece(
    piece_rgba: Image.Image,
    config: AugmentConfig,
    background_sampler: BackgroundSampler | None = None,
) -> Image.Image:
    """Apply the full domain-randomization pipeline to one RGBA piece.

    Pipeline order (each step gated by its config flag): small rotation
    jitter -> perspective -> scale -> mask erode/dilate -> composite on a
    sampled background (or black) -> independent photometric jitter ->
    grayscale/blur -> sensor noise -> JPEG. The result is an RGB image
    ready for the dataset's resize+ToTensor.

    Args:
        piece_rgba: The RGBA piece (alpha = true mask), already carrying
            its discrete 90-degree label rotation.
        config: Active augmentation config.
        background_sampler: Background sampler (required for the texture
            background mode; None uses procedural/black only).

    Returns:
        The augmented RGB piece.
    """
    if not config.enabled:
        return black_composite(piece_rgba)

    piece = piece_rgba if piece_rgba.mode == "RGBA" else piece_rgba.convert("RGBA")
    piece = _augment_geometry(piece, config)

    background: Image.Image | None = None
    if config.background:
        sampler = background_sampler or BackgroundSampler()
        background = sampler.sample(piece.size, config)

    return _augment_appearance(_composite(piece, background), config)


def augment_puzzle(puzzle_rgb: Image.Image, config: AugmentConfig) -> Image.Image:
    """Apply independent, milder photometric jitter to the puzzle image.

    Drawing the puzzle's photometric jitter *separately* from the piece's
    is the core of the anti-shortcut design: it removes the pixel-identity
    the exp20 CNN exploited.

    Args:
        puzzle_rgb: The RGB puzzle (box art) image.
        config: Active augmentation config.

    Returns:
        The jittered RGB puzzle image (unchanged if photometric is off).
    """
    if not config.enabled or not config.photometric:
        return puzzle_rgb
    return _color_jitter_image(puzzle_rgb, config, scale=config.puzzle_photometric_scale)


def config_to_dict(config: AugmentConfig) -> dict[str, object]:
    """Serialize an ``AugmentConfig`` to a plain dict for the results JSON.

    Args:
        config: The config to serialize.

    Returns:
        A JSON-serializable dict of every config field.
    """
    out: dict[str, object] = {}
    for f in fields(config):
        out[f.name] = getattr(config, f.name)
    return out


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible augmentation.

    Args:
        seed: The base seed.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
