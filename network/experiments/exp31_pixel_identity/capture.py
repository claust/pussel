"""Two-independent-captures augmentation for exp31 — Test 2 of the realism plan.

exp26 tried to break the piece<->overview pixel-identity shortcut with
photometric and geometric jitter and failed at the correlation level: masked
NCC at the ground-truth location stayed at 0.937 (raw synthetic 0.990) against
a real 0.730 (``docs/synthetic-dataset-realism.html`` section 4.3). The reason
is structural — exp26's photometric jitter is a near-affine intensity map that
any contrast-normalised matcher cancels, and its geometric jitter is absorbed
by a scale/rotation search. exp30 then removed both label-leaking generator
bugs and real transfer still did not move (13.2% vs 12.7%), which leaves
pixel identity as the prime suspect.

This module models the production truth instead: **the piece and the overview
are two independent captures of two different physical prints and never share
a pipeline instance.** Six components, every one of them drawn independently
per sample *and* per view (the face-anti-spoofing rule from section 7 — an
identical simulated artifact in both views just becomes the next shortcut):

1. :func:`apply_view_chain` — a full per-view degradation chain: subpixel
   phase shift, optical blur, resample down/up with independently drawn
   kernels, sensor noise at the sampled resolution, ISP sharpening, a
   *non-linear* per-channel tone curve, and JPEG with an **independent 8x8
   block grid**. Nothing is shared between the two views, not even an RNG
   draw.
2. :func:`apply_arp` — Asymmetric Random Patching (Chuah et al., 2106.08486;
   synthetic->real stereo error 28.0% -> 4.0%): with p=0.5 paste 2-4 random
   patches into **exactly one** view.
3. :func:`apply_segmentation_slop` — a cheap explicit model of the rembg
   segmentation the real path uses (mask blur + soft re-threshold, boundary
   noise, randomised dilate/erode), because real masks have softness,
   tab-neck rounding and pixel slop while synthetic masks are exact lattice
   cuts. Validated against true rembg by ``segmentation_validation.py``.
4. :func:`apply_die_cut_edge` + :func:`composite_with_shadow` — the piece is a
   3 mm die-cut 3D object with an exposed light cardboard core, not a flat
   page: a **bright** rim (section 5 measured real 1.08 vs synthetic 0.98,
   and noted exp26's halo pushes the wrong way at 0.69) plus a directional
   cast-shadow band.
5. :func:`apply_box_photo` and :func:`apply_piece_lighting` — both views are
   photographed under the same room light, independently drawn. The overview is
   a glossy box: mild residual perspective (the iOS pipeline rectifies imperfectly),
   specular glare blobs, a lighting gradient and vignetting. Section 5 also
   found ``augment_puzzle()`` leaves the overview essentially noise-free
   (0.07 vs real 0.84); the overview's own chain noise closes that.
6. :func:`frame_rgba_jittered` — explicit crop/bbox jitter, section 7's
   flagged gap (b): synthetic pieces are cut on an exact lattice, real ones
   are segmented with pixel slop.

Everything else is exp26/exp30 **unchanged**: :class:`CaptureConfig` extends
exp26's ``AugmentConfig``, the geometry/background/photometric stages are
exp26's own functions, and the framing is exp30's validated real-path
geometry. Three exp26 flags are flipped off by default and superseded, each
for a measured reason recorded on the field.

Resolution-asymmetry caveat (see README): the source corpus is 256x256, so a
4x4 cell is 64 px natively and the doc's ">=350 px piece source" is
unreachable. What this module does instead is establish the asymmetry
*direction* and break matched MTF — the overview is degraded on a coarser
independent chain than the piece, so the two views share neither a modulation
transfer function nor a pixel lattice. ``ViewChainConfig.target_px`` is the
documented knob that turns this into the doc's full asymmetry the moment a
higher-resolution corpus lands.
"""

from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import RandomPerspective
from torchvision.transforms import functional as TF

from ..exp24_piece_classifier.data_prep import ALPHA_THRESHOLD
from ..exp26_domain_randomization.augment import (
    AugmentConfig,
    BackgroundSampler,
    _augment_appearance,
    _augment_geometry,
    _color_jitter_image,
    _composite,
    augment_puzzle,
)
from ..exp30_generator_fixes.framing import alpha_bbox

# Resample kernels the chains draw from. Mixing kernels (and not just scales)
# is what stops the two views from sharing a modulation transfer function:
# a box downsample and a lanczos downsample of the same content differ in the
# high-frequency band by far more than any contrast normalisation can undo.
RESAMPLE_KERNELS: dict[str, Image.Resampling] = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
    "hamming": Image.Resampling.HAMMING,
}

# JPEG's DCT grid is anchored to the top-left pixel. Encoding after an
# edge-replicated pad of (ox, oy) and cropping back moves the grid to
# (-ox, -oy), so two independent draws give two misaligned block grids.
JPEG_BLOCK = 8


@dataclass
class ViewChainConfig:
    """Degradation chain for **one** view (piece or overview).

    One instance per view, never shared: the piece and the overview each own
    a config *and* draw their own randoms from it. Stage order follows the
    physical path — optics, sampling, sensor, ISP, codec — so the artifacts
    compose the way a real capture's do.
    """

    enabled: bool = True

    # --- Optics + sampling (the MTF / lattice break) ---
    resample: bool = True
    # Fraction of the view's own size the content is sampled down to before
    # being brought back up. Disjoint ranges between the two views is what
    # encodes "the piece is resolved finer than the overview".
    scale_min: float = 0.90
    scale_max: float = 1.00
    # Absolute override: the longer side is sampled down to this many pixels
    # instead of ``scale_*`` times the view size. This is the documented knob
    # for a higher-resolution corpus (see README): with 1024 px sources, set
    # the overview to ~500 px (125 px/cell, the real budget) and leave the
    # piece at None so it keeps its native >=350 px detail.
    target_px: int | None = None
    # Subpixel translation applied before sampling, so the two views do not
    # even share a pixel lattice phase.
    phase_px: float = 0.5
    down_kernels: tuple[str, ...] = ("bilinear", "bicubic", "lanczos", "box", "hamming")
    up_kernels: tuple[str, ...] = ("bilinear", "bicubic", "lanczos")
    optical_blur_p: float = 0.5
    optical_blur_sigma_min: float = 0.2
    optical_blur_sigma_max: float = 0.8

    # --- Sensor ---
    # Gaussian sigma in 0-255 units, applied at the *sampled* resolution so
    # the upsample correlates it like a real low-resolution capture's grain.
    noise: bool = True
    noise_sigma_min: float = 1.5
    noise_sigma_max: float = 7.0

    # --- ISP ---
    # The iPhone ISP sharpens aggressively; the close-up piece shot shows it
    # much more than a wide box shot does.
    sharpen_p: float = 0.5
    sharpen_radius_min: float = 0.6
    sharpen_radius_max: float = 1.6
    sharpen_percent_min: int = 40
    sharpen_percent_max: int = 140

    # --- Substrate / uneven illumination ---
    # A low-frequency *multiplicative* field, independently drawn per view.
    # This is the one photometric effect a zero-mean masked NCC genuinely
    # cannot cancel: it cancels global affine maps, and a smooth full-view ramp
    # is nearly affine in space, but a field whose correlation length is a
    # fraction of the view is neither. Physically it is the two substrates
    # (Ravensburger laminates linen-structured paper; the box is coated offset
    # stock) plus uneven room light.
    # Two octaves, because a single scale is not what either physical effect
    # looks like: a coarse octave for uneven illumination across the view and a
    # fine octave for substrate/gloss texture. Together they cover the band the
    # NCC actually feeds on. This is empirically the strongest single lever on
    # the section 8 headroom (GT median 0.847 -> 0.760 when it was added).
    substrate: bool = True
    substrate_amp_min: float = 0.06
    substrate_amp_max: float = 0.18
    substrate_cell_frac_min: float = 0.12
    substrate_cell_frac_max: float = 0.30
    substrate_fine_amp_min: float = 0.03
    substrate_fine_amp_max: float = 0.10
    substrate_fine_cell_frac_min: float = 0.03
    substrate_fine_cell_frac_max: float = 0.09

    # A *non-linear* tone curve is the point: masked NCC cancels affine
    # intensity maps (which is exactly why exp26's ColorJitter did nothing to
    # the correlation), but not a per-channel gamma.
    tone: bool = True
    gamma_min: float = 0.85
    gamma_max: float = 1.20
    channel_gamma_jitter: float = 0.06
    gain_min: float = 0.92
    gain_max: float = 1.08
    lift_max: float = 0.03

    # --- Codec ---
    jpeg: bool = True
    jpeg_p: float = 0.8
    jpeg_quality_min: int = 55
    jpeg_quality_max: int = 95
    jpeg_grid_jitter: bool = True


def piece_chain_defaults() -> ViewChainConfig:
    """Return the piece view's capture chain (the finer-resolved view).

    The piece is a close-up: sampled essentially at its own lattice and
    sharpened often (ISP oversharpening on a macro shot), because a real piece
    reaches the 128 px model input by being *downsampled* 3x from 382 px native
    — which is crisp, full-MTF-to-Nyquist. Section 5 measured real flat-region
    sigma 0.77 for pieces.

    Returns:
        The piece :class:`ViewChainConfig`.
    """
    return ViewChainConfig(
        scale_min=0.90,
        scale_max=1.00,
        phase_px=0.5,
        optical_blur_p=0.35,
        optical_blur_sigma_max=0.6,
        noise_sigma_min=2.0,
        noise_sigma_max=7.0,
        sharpen_p=0.6,
        jpeg_p=0.8,
    )


def overview_chain_defaults() -> ViewChainConfig:
    """Return the overview view's capture chain (the coarser-resolved view).

    The overview is a wide box shot at ~100-150 px per puzzle cell, reaching the
    256 px model input (64 px/cell) by a 1.6-2.3x downsample — so a *real*
    overview is also crisp at the model input, and the only honest MTF loss to
    model is what the capture itself lost to lens, hand-shake and capture-side
    JPEG before being downsampled. Hence a **mild** sampling range that is
    nevertheless strictly coarser than the piece's, with its own kernel and
    phase: the direction is real, the magnitude is not invented.

    An earlier, far more aggressive range (0.45-0.75, i.e. 29-48 px/cell) was
    both unfaithful — markedly softer than real — and counterproductive on the
    section 8 metric, which is worth recording because it is unintuitive:
    lowpassing the overview *raises* masked NCC, since NCC is dominated by low
    frequencies and the lowpass suppresses exactly the uncorrelated
    high-frequency band. Tightening to 0.62-0.88 moved the gate's GT median
    from 0.764 to 0.737 and the >0.8 tail from 0.431 to 0.381.

    Section 5 also measured real overview flat-region sigma 0.84 where
    ``augment_puzzle()`` leaves 0.07, so noise is on by default here.

    Returns:
        The overview :class:`ViewChainConfig`.
    """
    return ViewChainConfig(
        scale_min=0.62,
        scale_max=0.88,
        phase_px=0.7,
        optical_blur_p=0.6,
        optical_blur_sigma_min=0.3,
        optical_blur_sigma_max=1.1,
        noise_sigma_min=1.5,
        noise_sigma_max=6.0,
        sharpen_p=0.3,
        sharpen_percent_max=90,
        jpeg_p=0.9,
        jpeg_quality_min=45,
        jpeg_quality_max=88,
    )


@dataclass
class CaptureConfig(AugmentConfig):
    """exp26's ``AugmentConfig`` plus the exp31 two-independent-captures model.

    Inherits every exp26 field so the training recipe stays comparable, and
    flips exactly three of them off — each superseded by a measured-better
    exp31 component, each restorable through an ablation flag.
    """

    # --- exp26 overrides, all three for reasons measured in section 5 ---
    # exp26's halo erodes/dilates the alpha, and erosion is what drives the
    # boundary rim ratio to 0.69 (darkening) when the real value is 1.08
    # (brightening). Superseded by ``seg_slop`` (mask error, unbiased) plus
    # ``rim`` (the bright cardboard core).
    halo: bool = False
    # Superseded by the per-view chains, which apply noise at the sampled
    # resolution and JPEG on an independent block grid. Leaving exp26's
    # versions on would add a second, view-symmetric pass.
    noise: bool = False
    jpeg: bool = False

    # --- Master switch for everything exp31 adds ---
    capture: bool = True

    # --- 1. Independent per-view degradation chains ---
    independent_chains: bool = True
    # With this off both views run the *piece* chain from the same RNG state,
    # reproducing the shared-pass failure mode on purpose (diagnostic preset).
    resolution_asymmetry: bool = True
    piece_chain: ViewChainConfig = field(default_factory=piece_chain_defaults)
    overview_chain: ViewChainConfig = field(default_factory=overview_chain_defaults)

    # --- 2. Asymmetric Random Patching (one view only) ---
    arp: bool = True
    arp_p: float = 0.5
    arp_patches_min: int = 2
    arp_patches_max: int = 4
    # Chuah et al. use 50-100 px patches on KITTI stereo pairs, i.e. 13-27%
    # of the 375 px short side. Our views are 128 px (piece) and 256 px
    # (overview), so the range is expressed relative to the view's own short
    # side to keep the same relative occlusion.
    arp_size_min_frac: float = 0.10
    arp_size_max_frac: float = 0.25
    # Cut-Paste-Learn (1708.01642): naive hard pasting is itself a shortcut,
    # so a fraction of patches get a feathered alpha instead.
    arp_feather_p: float = 0.35

    # --- 3. Segmentation-artifact alignment (cheap rembg model) ---
    # The surface the piece is photographed on. Without it every mask
    # dilation reveals the generator's black fill instead of table content
    # (see apply_scene_surface); it also provides the room the rim band and
    # the dilation need outside the tight silhouette.
    scene_surface: bool = True
    surface_pad_frac: float = 0.10
    # Ranges fitted to real u2net output by ``segmentation_validation.py``
    # (60 pieces): the cheap model lands at soft_px 1.06 / alpha_grad 0.434 /
    # area_ratio 1.013 against rembg's 1.13 / 0.494 / 1.011, where the exact
    # lattice cut sits at 0.77 / 0.543 / 1.002. The first tried ranges were
    # 3.4x too soft, which is exactly why this is fitted and not guessed.
    seg_slop: bool = True
    slop_blur_min: float = 0.3
    slop_blur_max: float = 1.0
    slop_threshold_min: float = 0.38
    slop_threshold_max: float = 0.62
    slop_edge_softness_min: float = 0.3
    slop_edge_softness_max: float = 1.2
    slop_boundary_noise_amp: float = 0.18
    slop_boundary_noise_cell_px: int = 8
    slop_morph_max_px: int = 1
    # rembg errs outward in practice — measured area_ratio 1.011, i.e. it
    # keeps about 1% extra area at the rim.
    slop_dilate_p: float = 0.7

    # --- 4. Bright die-cut cardboard rim + cast shadow ---
    rim: bool = True
    # Fractions of the piece bbox short side, so the band scales with the
    # source resolution (4-9 px on today's ~100 px pieces). Wider than the
    # 3 mm core alone, on purpose: what section 5's 3-px boundary band
    # actually measures is the core *plus* the domed face and the specular
    # edge, and a sub-pixel band does not survive the downstream resampling.
    # Fitted (160 pieces) to land the view-level rim ratio at 1.087 against
    # the real 1.08 — synthetic was 0.98 and exp26-augmented 0.69, i.e. the
    # wrong direction. Spread p25 0.93 / p75 1.36, since real lighting varies.
    rim_frac_min: float = 0.030
    rim_frac_max: float = 0.070
    rim_strength_min: float = 0.35
    rim_strength_max: float = 0.75
    rim_luma_min: int = 195
    rim_luma_max: int = 250
    rim_warmth: int = 12
    shadow: bool = True
    shadow_frac_min: float = 0.010
    shadow_frac_max: float = 0.040
    # Capped below ALPHA_THRESHOLD/255 so the shadow can never be mistaken
    # for subject by the largest-component bbox or by the mask probes.
    shadow_alpha_min: float = 0.20
    shadow_alpha_max: float = 0.45

    # --- 5. Box-photo overview realism ---
    box_photo: bool = True
    box_perspective_p: float = 0.6
    # torchvision displaces each corner by up to distortion/2 of the side, so
    # 0.05 shifts a cell centre by <=2.5% of the image = <=10% of a cell.
    # That is bounded label noise matching the real domain's imperfect
    # rectification, not a label bug (README spells the arithmetic out).
    box_perspective_distortion: float = 0.05
    glare_p: float = 0.7
    glare_blobs_min: int = 1
    glare_blobs_max: int = 3
    glare_strength_min: float = 0.12
    glare_strength_max: float = 0.50
    glare_size_min_frac: float = 0.10
    glare_size_max_frac: float = 0.35
    lighting_gradient: bool = True
    lighting_amp: float = 0.15
    vignette: bool = True
    vignette_min: float = 0.05
    vignette_max: float = 0.25

    # --- 5b. The same room light, falling on the piece's close-up ---
    # Independently drawn and milder (the piece fills its frame, so the
    # falloff across it is a smaller fraction of the scene's). Spatially
    # varying illumination is the one photometric effect a zero-mean masked
    # NCC cannot cancel, which is why the piece needs its own.
    piece_lighting: bool = True
    piece_glare_p: float = 0.5
    piece_glare_scale: float = 0.7
    piece_lighting_scale: float = 1.3

    # --- 6. Crop / bbox jitter (section 7 gap (b)) ---
    crop_jitter: bool = True
    # Deliberately asymmetric: outward jitter is 2.5x the inward one so the
    # silhouette can never reach the input border and exp30's border-touch
    # acceptance probe keeps reading ~0. rembg also errs outward in practice.
    crop_jitter_inward: float = 0.02
    crop_jitter_outward: float = 0.05
    crop_margin_min: float = 0.05
    crop_margin_max: float = 0.12

    def ablation_flags(self) -> dict[str, bool]:
        """Return the **effective** on/off state of every augmentation.

        Extends exp26's flag set with the exp31 components so an ablation run
        is self-describing in ``results.json``. The exp31 entries report
        whether the component actually runs, i.e. they are ANDed with
        ``enabled`` and ``capture`` — a preset that switches ``capture`` off
        must not read as "ARP: on".

        Returns:
            Mapping of augmentation name to whether it is enabled.
        """
        flags = super().ablation_flags()
        names = (
            "independent_chains",
            "resolution_asymmetry",
            "arp",
            "scene_surface",
            "seg_slop",
            "rim",
            "shadow",
            "box_photo",
            "piece_lighting",
            "crop_jitter",
        )
        active = self.enabled and self.capture
        flags["capture"] = bool(active)
        flags.update({name: bool(active and getattr(self, name)) for name in names})
        return flags


def _capture_presets() -> dict[str, CaptureConfig]:
    """Build the named presets used by ``--capture-preset``.

    Each disables exactly one exp31 component relative to ``full`` so its
    marginal effect on the NCC headroom (and later on transfer) is
    measurable, plus two whole-family controls.

    Returns:
        Mapping of preset name to a :class:`CaptureConfig`.
    """
    presets: dict[str, CaptureConfig] = {
        "full": CaptureConfig(),
        # exp30 behaviour: exp26's augmentations only, exp31 additions off.
        "exp30": CaptureConfig(capture=False, halo=True, noise=True, jpeg=True),
        "no_chains": CaptureConfig(independent_chains=False),
        "no_asymmetry": CaptureConfig(resolution_asymmetry=False),
        "no_arp": CaptureConfig(arp=False),
        "no_substrate": CaptureConfig(
            piece_chain=replace(piece_chain_defaults(), substrate=False),
            overview_chain=replace(overview_chain_defaults(), substrate=False),
        ),
        "no_seg_slop": CaptureConfig(seg_slop=False),
        "no_rim": CaptureConfig(rim=False, shadow=False),
        "no_box_photo": CaptureConfig(box_photo=False),
        "no_piece_lighting": CaptureConfig(piece_lighting=False),
        "no_crop_jitter": CaptureConfig(crop_jitter=False),
        "chains_only": CaptureConfig(
            arp=False,
            scene_surface=False,
            seg_slop=False,
            rim=False,
            shadow=False,
            box_photo=False,
            piece_lighting=False,
            crop_jitter=False,
        ),
        "arp_only": CaptureConfig(
            independent_chains=False,
            scene_surface=False,
            seg_slop=False,
            rim=False,
            shadow=False,
            box_photo=False,
            piece_lighting=False,
            crop_jitter=False,
        ),
    }
    return presets


CAPTURE_PRESETS = _capture_presets()


def capture_config_to_dict(config: CaptureConfig) -> dict[str, object]:
    """Serialize a :class:`CaptureConfig` (nested chains included) to a dict.

    exp26's ``config_to_dict`` returns the nested ``ViewChainConfig``
    instances as objects, which ``json.dump`` cannot write; this uses
    ``dataclasses.asdict`` so the whole tree comes out JSON-serializable.

    Args:
        config: The config to serialize.

    Returns:
        A JSON-serializable dict of every config field.
    """
    return asdict(config)


# --------------------------------------------------------------------------
# 1. Independent per-view degradation chains
# --------------------------------------------------------------------------


@dataclass
class ChainDraw:
    """Every scalar parameter one capture chain uses, drawn up front.

    Separating the *draw* from the *apply* is what makes the shared-pass
    ablation honest. The obvious implementation — save the RNG state, run the
    chain twice, restore in between — silently does not work here, because the
    two views have different pixel dimensions: the noise array and the
    substrate octaves consume a size-dependent number of ``np.random`` values,
    so every draw after the first desynchronises and the "shared" pass is
    secretly independent again. With the parameters drawn once into this
    struct, ``independent_chains=False`` gives both views provably identical
    camera settings — same phase, blur, sampling factor, kernels, noise sigma,
    substrate amplitudes, sharpening, tone curve and JPEG quality/grid — while
    the pixel-level noise and field *realisations* necessarily differ, because
    they are realisations over differently shaped views. That is the strongest
    available notion of "one shared pipeline instance".
    """

    phase: tuple[float, float]
    blur_sigma: float | None
    sample_factor: float
    down_kernel: str
    up_kernel: str
    noise_sigma: float | None
    substrate: tuple[tuple[float, float], ...]
    sharpen: tuple[float, int] | None
    tone: tuple[tuple[float, float, float], float, float] | None
    jpeg: tuple[int, int, int] | None


def draw_chain(chain: ViewChainConfig) -> ChainDraw:
    """Draw one capture chain's parameters.

    Args:
        chain: The view's chain config.

    Returns:
        The drawn parameters.
    """
    phase = (
        random.uniform(-chain.phase_px, chain.phase_px) if chain.phase_px > 0 else 0.0,
        random.uniform(-chain.phase_px, chain.phase_px) if chain.phase_px > 0 else 0.0,
    )
    blur = (
        random.uniform(chain.optical_blur_sigma_min, chain.optical_blur_sigma_max)
        if random.random() < chain.optical_blur_p
        else None
    )
    factor = random.uniform(chain.scale_min, chain.scale_max)
    down_kernel = random.choice(chain.down_kernels)
    up_kernel = random.choice(chain.up_kernels)
    noise_sigma = random.uniform(chain.noise_sigma_min, chain.noise_sigma_max) if chain.noise else None
    substrate = (
        (
            random.uniform(chain.substrate_amp_min, chain.substrate_amp_max),
            random.uniform(chain.substrate_cell_frac_min, chain.substrate_cell_frac_max),
        ),
        (
            random.uniform(chain.substrate_fine_amp_min, chain.substrate_fine_amp_max),
            random.uniform(chain.substrate_fine_cell_frac_min, chain.substrate_fine_cell_frac_max),
        ),
    )
    sharpen = (
        (
            random.uniform(chain.sharpen_radius_min, chain.sharpen_radius_max),
            random.randint(chain.sharpen_percent_min, chain.sharpen_percent_max),
        )
        if random.random() < chain.sharpen_p
        else None
    )
    tone = None
    if chain.tone:
        base = random.uniform(chain.gamma_min, chain.gamma_max)
        gammas = tuple(
            base * (1.0 + random.uniform(-chain.channel_gamma_jitter, chain.channel_gamma_jitter)) for _ in range(3)
        )
        tone = (gammas, random.uniform(chain.gain_min, chain.gain_max), random.uniform(0.0, chain.lift_max))
    jpeg = None
    if chain.jpeg and random.random() < chain.jpeg_p:
        grid = JPEG_BLOCK - 1 if chain.jpeg_grid_jitter else 0
        jpeg = (
            random.randint(chain.jpeg_quality_min, chain.jpeg_quality_max),
            random.randint(0, grid),
            random.randint(0, grid),
        )
    return ChainDraw(
        phase=phase,
        blur_sigma=blur,
        sample_factor=factor,
        down_kernel=down_kernel,
        up_kernel=up_kernel,
        noise_sigma=noise_sigma,
        substrate=substrate,
        sharpen=sharpen,
        tone=tone,
        jpeg=jpeg,
    )


def _subpixel_shift(rgb: Image.Image, phase: tuple[float, float]) -> Image.Image:
    """Translate an image by a subpixel offset.

    Gives the view its own sampling-grid phase. Sub-pixel offsets make the
    edge fill negligible, so no explicit padding is needed.

    Args:
        rgb: RGB image.
        phase: (dx, dy) offset in pixels.

    Returns:
        The shifted RGB image.
    """
    dx, dy = phase
    if dx == 0.0 and dy == 0.0:
        return rgb
    return rgb.transform(
        rgb.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, dx, 0.0, 1.0, dy),
        resample=Image.Resampling.BICUBIC,
    )


def _sampled_size(size: tuple[int, int], chain: ViewChainConfig, factor: float) -> tuple[int, int]:
    """Return the intermediate size the view is sampled down to.

    Args:
        size: The view's (width, height).
        chain: Active chain config.
        factor: Drawn scale factor, ignored when ``target_px`` is set.

    Returns:
        The sampled (width, height), at least 8x8.
    """
    width, height = size
    if chain.target_px is not None:
        factor = chain.target_px / max(width, height)
    return max(8, round(width * factor)), max(8, round(height * factor))


def _add_noise_array(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise (0-255 units) to a float array.

    Args:
        arr: Float array in [0, 255].
        sigma: Noise standard deviation in 0-255 units.

    Returns:
        The noisy array, clipped to [0, 255].
    """
    if sigma <= 0:
        return arr
    return np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0.0, 255.0)


def _resample_stage(rgb: Image.Image, chain: ViewChainConfig, draw: ChainDraw) -> Image.Image:
    """Sample the view down and back up, with its own kernels and noise.

    The MTF break: the down-kernel, the up-kernel, the scale and the sensor
    noise all come from this view's own draw, and the noise is injected at the
    *sampled* resolution so the upsample correlates the grain the way a real
    low-resolution capture's is correlated.

    Args:
        rgb: RGB image.
        chain: Active chain config.
        draw: This view's drawn parameters.

    Returns:
        The resampled RGB image, back at its original size.
    """
    original = rgb.size
    target = _sampled_size(original, chain, draw.sample_factor)
    sampled = rgb.resize(target, RESAMPLE_KERNELS[draw.down_kernel])
    if draw.noise_sigma is not None:
        arr = _add_noise_array(np.asarray(sampled, dtype=np.float32), draw.noise_sigma)
        sampled = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return sampled.resize(original, RESAMPLE_KERNELS[draw.up_kernel])


def _substrate_stage(rgb: Image.Image, draw: ChainDraw) -> Image.Image:
    """Modulate the view by two multiplicative octaves.

    Coarse octave = uneven illumination across the view; fine octave =
    substrate and gloss texture. A zero-mean masked NCC cancels global affine
    intensity maps — which is why exp26's ColorJitter left the correlation at
    0.937 — and a single smooth full-view ramp is nearly affine in space; two
    octaves at a fraction of the view are neither, which is what makes this
    the strongest single lever on the section 8 headroom.

    Args:
        rgb: RGB image.
        draw: This view's drawn parameters.

    Returns:
        The modulated RGB image.
    """
    shape = (rgb.height, rgb.width)
    short_side = min(shape)
    gain = np.ones(shape, dtype=np.float32)
    for amp, cell_frac in draw.substrate:
        cell_px = max(2, round(short_side * cell_frac))
        gain = gain * (1.0 + amp * _low_frequency_noise(shape, cell_px))
    arr = np.asarray(rgb, dtype=np.float32) * gain[..., None]
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _tone_stage(rgb: Image.Image, draw: ChainDraw) -> Image.Image:
    """Apply a non-linear per-channel tone curve.

    ``out = (in ** gamma_c) * gain + lift`` with a small per-channel gamma
    spread, modelling two different print runs' dot gain and gamut rather than
    a camera's white balance. Deliberately non-affine: a contrast-normalised
    matcher cannot cancel it, which is precisely where exp26's ColorJitter
    failed.

    Args:
        rgb: RGB image.
        draw: This view's drawn parameters.

    Returns:
        The tone-mapped RGB image.
    """
    assert draw.tone is not None
    gammas, gain, lift = draw.tone
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), np.array(gammas, dtype=np.float32)[None, None, :]) * gain + lift
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def _jpeg_stage(rgb: Image.Image, draw: ChainDraw) -> Image.Image:
    """Re-encode as JPEG on this view's own 8x8 block grid.

    The grid offset is the part exp26 was missing: both views were encoded with
    their DCT lattice anchored to pixel (0, 0), so the block artifacts lined up
    and stayed correlated. Padding by (ox, oy) with edge replication before
    encoding and cropping back afterwards moves the lattice to (-ox, -oy).

    Args:
        rgb: RGB image.
        draw: This view's drawn parameters.

    Returns:
        The re-encoded RGB image, at its original size.
    """
    assert draw.jpeg is not None
    quality, ox, oy = draw.jpeg

    padded = rgb
    if ox or oy:
        arr = np.pad(np.asarray(rgb), ((oy, 0), (ox, 0), (0, 0)), mode="edge")
        padded = Image.fromarray(arr, mode="RGB")

    buffer = io.BytesIO()
    padded.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        out = decoded.convert("RGB")
    return out.crop((ox, oy, ox + rgb.width, oy + rgb.height)) if (ox or oy) else out


def apply_chain_draw(rgb: Image.Image, chain: ViewChainConfig, draw: ChainDraw) -> Image.Image:
    """Run one view through a capture chain with already-drawn parameters.

    Physical stage order: subpixel phase -> optical blur -> sample down ->
    sensor noise -> sample up -> substrate/illumination field -> ISP sharpen ->
    tone curve -> JPEG.

    Args:
        rgb: The view as an RGB image.
        chain: The chain config (for the structural flags).
        draw: The drawn parameters.

    Returns:
        The degraded RGB image, same size as the input.
    """
    if not chain.enabled:
        return rgb

    out = _subpixel_shift(rgb, draw.phase)
    if draw.blur_sigma is not None:
        out = out.filter(ImageFilter.GaussianBlur(radius=draw.blur_sigma))
    if chain.resample:
        out = _resample_stage(out, chain, draw)
    if chain.substrate:
        out = _substrate_stage(out, draw)
    if draw.sharpen is not None:
        radius, percent = draw.sharpen
        out = out.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
    if draw.tone is not None:
        out = _tone_stage(out, draw)
    if draw.jpeg is not None:
        out = _jpeg_stage(out, draw)
    return out


def apply_view_chain(rgb: Image.Image, chain: ViewChainConfig) -> Image.Image:
    """Draw one view's capture-chain parameters and apply them.

    The normal entry point: called once per view with that view's own config,
    so every random draw inside is that view's alone.

    Args:
        rgb: The view as an RGB image.
        chain: That view's chain config.

    Returns:
        The degraded RGB image, same size as the input.
    """
    if not chain.enabled:
        return rgb
    return apply_chain_draw(rgb, chain, draw_chain(chain))


# --------------------------------------------------------------------------
# 2. Asymmetric Random Patching
# --------------------------------------------------------------------------


class PatchSource:
    """Supplies ARP patch content that belongs to neither view's true content.

    Patches are drawn from *other* source images (60%), a smooth gradient
    (25%) or uniform noise (15%). Deliberately never a crop of the image
    being patched: self-patching an overview would duplicate one cell's
    content somewhere else in the same overview and weaken the label the
    model is being trained to predict.
    """

    def __init__(self, texture_paths: list[Path] | None = None, cache_size: int = 32) -> None:
        """Initialize the patch source.

        Args:
            texture_paths: Source images to cut patches from (typically the
                training puzzle JPEGs — never val/test, never north_star).
            cache_size: Number of decoded source images to keep in memory.
        """
        self.texture_paths = texture_paths or []
        self.cache_size = cache_size
        self._cache: dict[Path, Image.Image] = {}

    def _load(self, path: Path) -> Image.Image:
        """Load and cache one source image as RGB.

        Args:
            path: Image path.

        Returns:
            The decoded RGB image.
        """
        cached = self._cache.get(path)
        if cached is None:
            with Image.open(path) as img:
                cached = img.convert("RGB")
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = cached
        return cached

    def _texture_patch(self, size: tuple[int, int], exclude_stem: str | None) -> Image.Image | None:
        """Cut a random patch out of a random other source image.

        Args:
            size: Patch (width, height).
            exclude_stem: Source stem to avoid (the puzzle being augmented).

        Returns:
            The patch, or None when no usable source is configured.
        """
        candidates = [p for p in self.texture_paths if p.stem != exclude_stem]
        if not candidates:
            return None
        try:
            src = self._load(random.choice(candidates))
        except (OSError, ValueError):
            return None
        width, height = min(size[0], src.width), min(size[1], src.height)
        left = random.randint(0, src.width - width)
        top = random.randint(0, src.height - height)
        return src.crop((left, top, left + width, top + height)).resize(size, Image.Resampling.BILINEAR)

    def patch(self, size: tuple[int, int], exclude_stem: str | None = None) -> Image.Image:
        """Return one ARP patch of the requested size.

        Args:
            size: Patch (width, height).
            exclude_stem: Source stem to avoid (the puzzle being augmented).

        Returns:
            An RGB patch.
        """
        roll = random.random()
        if roll < 0.60:
            textured = self._texture_patch(size, exclude_stem)
            if textured is not None:
                return textured
        if roll < 0.85:
            colours = np.array([[random.randint(0, 255) for _ in range(3)] for _ in range(2)], dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, size[0], dtype=np.float32)[None, :, None]
            arr = colours[0][None, None, :] * (1.0 - ramp) + colours[1][None, None, :] * ramp
            arr = np.broadcast_to(arr, (size[1], size[0], 3))
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")
        noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(noise, mode="RGB")


def _paste_patch(view: Image.Image, patch: Image.Image, feather: bool) -> None:
    """Paste one patch into a view at a random position, in place.

    Args:
        view: RGB view to modify.
        patch: RGB patch to paste.
        feather: Whether to blend with a feathered alpha instead of pasting
            hard (Cut-Paste-Learn: a single fixed blend mode is itself a
            learnable artifact).
    """
    x = random.randint(0, max(0, view.width - patch.width))
    y = random.randint(0, max(0, view.height - patch.height))
    if not feather:
        view.paste(patch, (x, y))
        return
    mask = Image.new("L", patch.size, 255)
    radius = max(1.0, min(patch.size) * 0.2)
    view.paste(patch, (x, y), mask.filter(ImageFilter.GaussianBlur(radius=radius)))


def apply_arp(
    piece_rgb: Image.Image,
    overview_rgb: Image.Image,
    config: CaptureConfig,
    patch_source: PatchSource | None = None,
    exclude_stem: str | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Apply Asymmetric Random Patching to **exactly one** of the two views.

    The stereo literature's anti-pixel-identity augmentation (Chuah et al.,
    2106.08486: synthetic->real error 28.0% -> 4.0% from ACA + ARP). Content
    that exists in one view and provably not in the other is the one thing a
    pixel-matching shortcut cannot survive, and unlike a photometric map it
    cannot be normalised away.

    Args:
        piece_rgb: The piece view.
        overview_rgb: The overview view.
        config: Active capture config.
        patch_source: Patch content supplier (a default is built if None).
        exclude_stem: Source stem to avoid when cutting texture patches.

    Returns:
        The (possibly patched) piece and overview views. Exactly one of them
        is ever modified.
    """
    if not config.capture or not config.arp or random.random() >= config.arp_p:
        return piece_rgb, overview_rgb

    source = patch_source or PatchSource()
    target = piece_rgb.copy() if random.random() < 0.5 else overview_rgb.copy()
    short_side = min(target.size)
    for _ in range(random.randint(config.arp_patches_min, config.arp_patches_max)):
        frac = random.uniform(config.arp_size_min_frac, config.arp_size_max_frac)
        side = max(3, round(short_side * frac))
        size = (side, max(3, round(side * random.uniform(0.6, 1.6))))
        size = (min(size[0], target.width), min(size[1], target.height))
        _paste_patch(target, source.patch(size, exclude_stem), random.random() < config.arp_feather_p)

    return (target, overview_rgb) if target.size == piece_rgb.size else (piece_rgb, target)


# --------------------------------------------------------------------------
# 3. Segmentation-artifact alignment
# --------------------------------------------------------------------------


def _pad_rgba(piece_rgba: Image.Image, pad: int) -> Image.Image:
    """Pad an RGBA piece with transparent pixels on every side.

    Both the rim band and the dilation stages need room outside the
    silhouette; generator pieces arrive cropped tight, so a canvas-touching
    silhouette would otherwise get no rim (and no room to dilate) on that
    side.

    Args:
        piece_rgba: RGBA piece.
        pad: Pixels of transparent padding per side.

    Returns:
        The padded RGBA piece.
    """
    if pad <= 0:
        return piece_rgba
    canvas = Image.new("RGBA", (piece_rgba.width + 2 * pad, piece_rgba.height + 2 * pad), (0, 0, 0, 0))
    canvas.paste(piece_rgba, (pad, pad))
    return canvas


def apply_scene_surface(
    piece_rgba: Image.Image,
    config: CaptureConfig,
    background_sampler: BackgroundSampler | None = None,
) -> Image.Image:
    """Give the piece room to bleed into, and a plausible surface to bleed from.

    Two things depend on this and both are wrong without it:

    - ``cut_piece`` writes **black** RGB wherever alpha is zero
      (``image_masking.py``: ``Image.new(..., (0, 0, 0, 0))`` then a masked
      paste), and transparent padding is black too. So every mask *dilation*
      downstream would reveal pure black — an artifact of the fill value, not
      of any camera. Measured effect: it put a heavy dark tail on the boundary
      rim ratio (25th percentile 0.57) and dragged the median to 0.83 while
      the real value is 1.08.
    - The rim band and the dilation need room outside the tight silhouette,
      which the generator's crop does not leave.

    Physically this is the surface the piece is photographed on: rembg keeps a
    pixel or two of it at the rim, so those pixels carry table content, not
    black. Note the generator does *not* leave the neighbouring piece's
    content there (it is zeroed), so no cell label can leak through this
    stage — the surface is drawn from unrelated imagery.

    Args:
        piece_rgba: RGBA piece cropped tight to its silhouette.
        config: Active capture config.
        background_sampler: Sampler used for the surface (None uses
            procedural surfaces only).

    Returns:
        A padded RGBA piece whose out-of-silhouette RGB is a plausible
        surface. Alpha is untouched.
    """
    if not config.capture or not config.scene_surface:
        return piece_rgba

    pad = max(2, round(min(piece_rgba.size) * config.surface_pad_frac) + 2)
    padded = _pad_rgba(piece_rgba, pad)

    sampler = background_sampler or BackgroundSampler()
    roll = random.random()
    if roll < 0.35:
        surface = sampler.solid(padded.size)
    elif roll < 0.70:
        surface = sampler.gradient(padded.size)
    else:
        surface = sampler.texture(padded.size)

    alpha = np.asarray(padded.getchannel("A"))
    outside = (alpha <= ALPHA_THRESHOLD)[..., None]
    arr = np.where(outside, np.asarray(surface.convert("RGB")), np.asarray(padded.convert("RGB")))

    out = Image.fromarray(arr.astype(np.uint8), mode="RGB").convert("RGBA")
    out.putalpha(padded.getchannel("A"))
    return out


def _low_frequency_noise(shape: tuple[int, int], cell_px: int) -> np.ndarray:
    """Return a smooth zero-mean noise field in roughly [-1, 1].

    Args:
        shape: Output (height, width).
        cell_px: Approximate correlation length in pixels.

    Returns:
        A float32 field of ``shape``.
    """
    height, width = shape
    small = (max(2, height // max(1, cell_px)), max(2, width // max(1, cell_px)))
    coarse = np.random.uniform(-1.0, 1.0, small).astype(np.float32)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def _short_side(mask: np.ndarray) -> float:
    """Return the short side of a boolean mask's bounding box, in pixels.

    Args:
        mask: Boolean mask.

    Returns:
        The bbox short side, or 0.0 for an empty mask.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0.0
    return float(min(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1))


def apply_segmentation_slop(piece_rgba: Image.Image, config: CaptureConfig) -> Image.Image:
    """Perturb the alpha mask the way rembg perturbs a real piece's mask.

    The real pipeline segments pieces with rembg (u2net), so real masks have
    boundary softness, tab-neck rounding and a couple of pixels of slop;
    synthetic masks are exact Bezier lattice cuts. Running true rembg on
    ~192k pieces every epoch is far too slow, so this is an explicit cheap
    model of the same statistics: low-frequency boundary noise, a Gaussian
    blur, a *soft* re-threshold (which is what keeps the boundary gradient
    rembg-like instead of a step), and a small randomised dilate/erode
    biased outward. ``segmentation_validation.py`` measures this against
    real rembg on the three statistics that matter.

    RGB is untouched — only alpha moves, so the piece's print content is
    unchanged and no label can leak through this stage.

    Args:
        piece_rgba: RGBA piece (alpha = the exact generator mask).
        config: Active capture config.

    Returns:
        The RGBA piece with a slopped alpha channel.
    """
    if not config.capture or not config.seg_slop:
        return piece_rgba

    alpha = np.asarray(piece_rgba.getchannel("A"), dtype=np.float32) / 255.0
    if not (alpha > 0.5).any():
        return piece_rgba

    field_noise = _low_frequency_noise(alpha.shape, config.slop_boundary_noise_cell_px)
    perturbed = alpha + field_noise * config.slop_boundary_noise_amp

    blur = random.uniform(config.slop_blur_min, config.slop_blur_max)
    perturbed = cv2.GaussianBlur(perturbed, (0, 0), sigmaX=blur, sigmaY=blur)

    threshold = random.uniform(config.slop_threshold_min, config.slop_threshold_max)
    softness_px = random.uniform(config.slop_edge_softness_min, config.slop_edge_softness_max)
    # Soft re-threshold: a hard step would give an infinitely sharp boundary,
    # which is exactly the synthetic artifact being removed. The blurred
    # alpha crosses 0->1 over roughly 2*blur pixels, so a target transition
    # of ``softness_px`` pixels means a window of softness_px/(2*blur) in
    # alpha units.
    window = float(np.clip(softness_px / (2.0 * blur), 0.02, 1.0))
    soft = np.clip((perturbed - threshold) / window + 0.5, 0.0, 1.0)

    radius = random.randint(0, max(0, config.slop_morph_max_px))
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        soft = cv2.dilate(soft, kernel) if random.random() < config.slop_dilate_p else cv2.erode(soft, kernel)

    out = piece_rgba.copy()
    out.putalpha(Image.fromarray((soft * 255.0).astype(np.uint8), mode="L"))
    return out


# --------------------------------------------------------------------------
# 4. Bright die-cut cardboard rim + cast shadow
# --------------------------------------------------------------------------


def apply_die_cut_edge(piece_rgba: Image.Image, config: CaptureConfig) -> Image.Image:
    """Brighten the piece's boundary band like an exposed cardboard core.

    A puzzle piece is a ~3 mm die-cut 3D object: the cut edge shows light
    grey-brown board, and a close-up photo of it under any lighting picks
    that up as a bright rim. Section 5 measured real boundary rim ratio 1.08
    against synthetic 0.98 — and found exp26's halo augmentation pushing the
    *wrong* way at 0.69 — which is why exp31 turns ``halo`` off and adds this
    instead.

    Also part of section 7's flagged gap (a): the synthetic crop has razor
    edges with the neighbouring piece's pixels immediately adjacent, where
    the real object has a lit core and a cast shadow.

    Expects the caller to have run :func:`apply_scene_surface` first, so the
    silhouette does not touch the canvas border (a border-touching silhouette
    gets no rim on that side, because the distance transform sees no outside
    there).

    Args:
        piece_rgba: RGBA piece (alpha = the exact generator mask).
        config: Active capture config.

    Returns:
        The RGBA piece with the rim band brightened; alpha untouched.
    """
    if not config.capture or not config.rim:
        return piece_rgba

    padded = piece_rgba
    mask = (np.asarray(padded.getchannel("A")) > ALPHA_THRESHOLD).astype(np.uint8)
    side = _short_side(mask.astype(bool))
    if side <= 0:
        return padded

    band_px = max(1.0, side * random.uniform(config.rim_frac_min, config.rim_frac_max))
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    weight = np.clip(1.0 - distance / band_px, 0.0, 1.0) * mask
    weight = weight * random.uniform(config.rim_strength_min, config.rim_strength_max)

    luma = random.uniform(config.rim_luma_min, config.rim_luma_max)
    warmth = config.rim_warmth
    core = np.array([luma + warmth, luma, luma - warmth], dtype=np.float32)

    arr = np.asarray(padded.convert("RGB"), dtype=np.float32)
    arr = arr * (1.0 - weight[..., None]) + core[None, None, :] * weight[..., None]

    out = Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB").convert("RGBA")
    out.putalpha(padded.getchannel("A"))
    return out


def composite_with_shadow(
    piece_rgba: Image.Image,
    background: Image.Image | None,
    config: CaptureConfig,
) -> Image.Image:
    """Composite the piece over its background, casting a directional shadow.

    The shadow is derived from the *final* silhouette (after slop, framing
    and geometry) because that is the silhouette the light actually meets.
    Its alpha is capped below ``ALPHA_THRESHOLD`` so it can never be counted
    as subject by the largest-component bbox or by the mask probes, and on a
    black background it is invisible — which is exactly what happens in
    deployment, where rembg removes the surface the shadow fell on.

    Args:
        piece_rgba: RGBA piece.
        background: RGB background of the same size, or None for black.
        config: Active capture config.

    Returns:
        The composited RGB image.
    """
    if not config.capture or not config.shadow or background is None:
        return _composite(piece_rgba, background)

    mask = (np.asarray(piece_rgba.getchannel("A")) > ALPHA_THRESHOLD).astype(np.uint8)
    side = _short_side(mask.astype(bool))
    if side <= 0:
        return _composite(piece_rgba, background)

    band = max(1, round(side * random.uniform(config.shadow_frac_min, config.shadow_frac_max)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    grown = cv2.dilate(mask, kernel)
    offset = np.roll(np.roll(grown, random.randint(-band, band), axis=0), random.randint(-band, band), axis=1)
    ring = cv2.GaussianBlur((offset * (1 - mask)).astype(np.float32), (0, 0), sigmaX=max(0.5, band * 0.6))
    ring = ring * random.uniform(config.shadow_alpha_min, config.shadow_alpha_max)

    if background.size != piece_rgba.size:
        background = background.resize(piece_rgba.size, Image.Resampling.BILINEAR)
    shaded = np.asarray(background.convert("RGB"), dtype=np.float32) * (1.0 - ring[..., None])
    darkened = Image.fromarray(np.clip(shaded, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return _composite(piece_rgba, darkened)


# --------------------------------------------------------------------------
# 5. Box-photo overview realism
# --------------------------------------------------------------------------


def _residual_perspective(rgb: Image.Image, distortion: float) -> Image.Image:
    """Apply a mild perspective warp without introducing empty corners.

    The iOS pipeline rectifies the box photo but not perfectly, so a few
    percent of residual perspective is part of the real overview domain. The
    image is edge-padded first and centre-cropped afterwards so the warp's
    empty wedges land in the padding rather than in the view.

    Args:
        rgb: RGB overview.
        distortion: torchvision ``distortion_scale``.

    Returns:
        The warped RGB overview, same size as the input.
    """
    pad = max(2, round(max(rgb.size) * distortion))
    padded = Image.fromarray(np.pad(np.asarray(rgb), ((pad, pad), (pad, pad), (0, 0)), mode="edge"), mode="RGB")
    start, end = RandomPerspective.get_params(padded.width, padded.height, distortion)
    warped = TF.perspective(padded, start, end, interpolation=TF.InterpolationMode.BILINEAR)
    assert isinstance(warped, Image.Image)
    return warped.crop((pad, pad, pad + rgb.width, pad + rgb.height))


def _glare_blobs(shape: tuple[int, int], config: CaptureConfig, scale: float = 1.0) -> np.ndarray:
    """Build an additive specular-glare field for a glossy printed surface.

    Args:
        shape: (height, width) of the view.
        config: Active capture config.
        scale: Multiplier on blob strength and size (the piece view uses a
            milder scale than the wide box shot).

    Returns:
        A float32 field in [0, 1] giving the blend weight toward white.
    """
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    field = np.zeros((height, width), dtype=np.float32)
    short_side = min(height, width)
    for _ in range(random.randint(config.glare_blobs_min, config.glare_blobs_max)):
        radius = short_side * random.uniform(config.glare_size_min_frac, config.glare_size_max_frac) * scale
        cx = random.uniform(0.0, width)
        cy = random.uniform(0.0, height)
        aspect = random.uniform(0.5, 2.0)
        dist2 = ((xx - cx) / (max(1e-3, radius * aspect))) ** 2 + ((yy - cy) / max(1e-3, radius)) ** 2
        strength = random.uniform(config.glare_strength_min, config.glare_strength_max) * scale
        field += strength * np.exp(-2.0 * dist2)
    return np.clip(field, 0.0, 1.0)


def _lighting_and_vignette(arr: np.ndarray, config: CaptureConfig, scale: float = 1.0) -> np.ndarray:
    """Apply a directional lighting gradient and a radial vignette.

    Args:
        arr: HxWx3 float array in [0, 255].
        config: Active capture config.
        scale: Multiplier on both amplitudes.

    Returns:
        The modulated array (not yet clipped).
    """
    height, width = arr.shape[:2]
    if config.lighting_gradient and config.lighting_amp > 0:
        angle = random.uniform(0.0, 2.0 * np.pi)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        ramp = (np.cos(angle) * xx / width + np.sin(angle) * yy / height) - 0.5
        arr = arr * (1.0 + config.lighting_amp * scale * ramp)[..., None]
    if config.vignette:
        strength = random.uniform(config.vignette_min, config.vignette_max) * scale
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        radius = np.sqrt(((xx / width - 0.5) * 2.0) ** 2 + ((yy / height - 0.5) * 2.0) ** 2) / np.sqrt(2.0)
        arr = arr * (1.0 - strength * radius**2)[..., None]
    return arr


def apply_piece_lighting(rgb: Image.Image, config: CaptureConfig) -> Image.Image:
    """Light the piece's close-up like a real macro shot under room lighting.

    The piece view was the only one of the two with no spatially varying
    illumination at all, which is both physically wrong — a phone macro shot of
    a laminated, slightly domed piece has a light falloff and usually a
    specular highlight — and the single most effective remaining lever on the
    section 4.3 shortcut. A masked *zero-mean* NCC cancels any global affine
    intensity map, which is exactly why exp26's ColorJitter left the
    correlation at 0.937; a spatially varying multiplicative field is not
    affine and does not cancel.

    Drawn independently of the overview's lighting, and milder: the piece fills
    the frame, so the falloff across it is a smaller fraction of the scene's.

    Args:
        rgb: The composited RGB piece view.
        config: Active capture config.

    Returns:
        The relit RGB piece view.
    """
    if not config.capture or not config.piece_lighting:
        return rgb

    arr = np.asarray(rgb, dtype=np.float32)
    if random.random() < config.piece_glare_p:
        glare = _glare_blobs(arr.shape[:2], config, scale=config.piece_glare_scale)[..., None]
        arr = arr * (1.0 - glare) + 255.0 * glare
    arr = _lighting_and_vignette(arr, config, scale=config.piece_lighting_scale)
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB")


def apply_box_photo(rgb: Image.Image, config: CaptureConfig) -> Image.Image:
    """Make an overview look like a photograph of a glossy puzzle box.

    Section 2: in production the user photographs the box — printed motif on
    glossy cardboard — so the overview's realism target is "print-and-
    photograph of box art" under real lighting, mildly perspective-corrected
    by the iOS pipeline. Pure numpy/PIL/torchvision; no Augraphy dependency
    (CPU-only, document-oriented, and a new dep needs a root ``uv lock``).

    Args:
        rgb: RGB overview.
        config: Active capture config.

    Returns:
        The RGB overview with residual perspective, specular glare, a
        lighting gradient and vignetting applied.
    """
    if not config.capture or not config.box_photo:
        return rgb

    out = rgb
    if random.random() < config.box_perspective_p and config.box_perspective_distortion > 0:
        out = _residual_perspective(out, config.box_perspective_distortion)

    arr = np.asarray(out, dtype=np.float32)
    if random.random() < config.glare_p:
        glare = _glare_blobs(arr.shape[:2], config)[..., None]
        arr = arr * (1.0 - glare) + 255.0 * glare
    arr = _lighting_and_vignette(arr, config)
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB")


# --------------------------------------------------------------------------
# 6. Crop / bbox jitter
# --------------------------------------------------------------------------


def frame_rgba_jittered(piece_rgba: Image.Image, config: CaptureConfig) -> Image.Image:
    """Frame the piece like the real path does, with segmentation-grade slop.

    exp30's ``frame_rgba`` reproduces exp24's deployment geometry exactly:
    largest opaque component -> 8% margin -> pad to square. Real crops are
    not that tidy — the bbox comes from an imperfect segmentation, so both
    the box and the margin wobble (section 7, gap (b): classical print-scan
    analysis found mild cropping is the distortion that hits every frequency
    band).

    The jitter is deliberately asymmetric — up to 5% outward but only 2%
    inward, against a margin of at least 5% — so the silhouette still cannot
    reach the input border and exp30's border-touch acceptance probe keeps
    reading ~0. It is also drawn independently of the rotation label, so it
    cannot become a second 4.1-style leak.

    Args:
        piece_rgba: RGBA piece (alpha = mask).
        config: Active capture config.

    Returns:
        A square RGBA piece, framed with jittered crop and margin.
    """
    piece = piece_rgba if piece_rgba.mode == "RGBA" else piece_rgba.convert("RGBA")
    bbox = alpha_bbox(piece)
    if bbox is None:
        crop = piece
    else:
        crop = piece.crop(_jittered_box(piece.size, bbox, config))

    margin = (
        random.uniform(config.crop_margin_min, config.crop_margin_max)
        if config.capture and config.crop_jitter
        else 0.08
    )
    pad_x = round(crop.width * margin)
    pad_y = round(crop.height * margin)
    side = max(crop.width + 2 * pad_x, crop.height + 2 * pad_y)

    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


def _jittered_box(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    config: CaptureConfig,
) -> tuple[int, int, int, int]:
    """Jitter a bounding box's four sides independently.

    Args:
        size: The source image's (width, height).
        bbox: Bounding box (x, y, w, h).
        config: Active capture config.

    Returns:
        Crop box (left, top, right, bottom), clamped to the image.
    """
    x, y, width, height = bbox
    if not config.capture or not config.crop_jitter:
        return x, y, x + width, y + height

    def offset(extent: int) -> int:
        """Draw one side's jitter in pixels (positive = outward)."""
        return round(extent * random.uniform(-config.crop_jitter_inward, config.crop_jitter_outward))

    left = max(0, x - offset(width))
    top = max(0, y - offset(height))
    right = min(size[0], x + width + offset(width))
    bottom = min(size[1], y + height + offset(height))
    return left, top, max(left + 1, right), max(top + 1, bottom)


# --------------------------------------------------------------------------
# The two view pipelines
# --------------------------------------------------------------------------


def resolve_chain(config: CaptureConfig, view: str) -> ViewChainConfig:
    """Return the chain config a view should run.

    Args:
        config: Active capture config.
        view: ``"piece"`` or ``"overview"``.

    Returns:
        That view's chain config. With ``independent_chains`` off both views
        get the piece's chain (the shared-pass diagnostic); with
        ``resolution_asymmetry`` off the overview keeps its own chain but
        borrows the piece's sampling budget, so the two views are degraded to
        the same MTF from independent draws.

    Raises:
        ValueError: If ``view`` is not a known view name.
    """
    if view not in ("piece", "overview"):
        raise ValueError(f"view must be 'piece' or 'overview', got {view!r}")
    if view == "piece" or not config.independent_chains:
        return config.piece_chain
    if config.resolution_asymmetry:
        return config.overview_chain
    return replace(
        config.overview_chain,
        scale_min=config.piece_chain.scale_min,
        scale_max=config.piece_chain.scale_max,
        target_px=config.piece_chain.target_px,
    )


def augment_piece_capture(
    piece_rgba: Image.Image,
    config: CaptureConfig,
    background_sampler: BackgroundSampler | None = None,
    apply_chain: bool = True,
) -> tuple[Image.Image, Image.Image]:
    """Render one piece as an independent close-up capture of a die-cut piece.

    Stage order, with the exp31 additions marked:

    1. scene surface: pad, and fill the out-of-silhouette RGB with table
       content instead of the generator's black (**exp31**)
    2. bright die-cut rim (**exp31**)
    3. segmentation slop on the alpha (**exp31**)
    4. jittered real-path framing to a square canvas (**exp31**, on top of
       exp30's validated geometry)
    5. exp26 geometry: rotation jitter -> perspective -> scale (``halo`` off)
    6. composite on a sampled background, casting a shadow (**exp31**)
    7. the room light falling on the close-up: gradient, vignette, specular
       highlight (**exp31**)
    8. exp26 photometric jitter (independent draw, as in exp26)
    9. the piece's own capture chain (**exp31**)

    With ``capture`` off, steps 1-3 and 5-9 collapse to exp26's own
    ``_augment_appearance`` after the composite, which is what makes
    ``--capture-preset exp30`` a faithful control rather than a weaker one.

    Args:
        piece_rgba: RGBA piece carrying its discrete 90-degree label rotation.
        config: Active capture config.
        background_sampler: Background sampler (None uses procedural/black).
        apply_chain: Whether to run step 9 here. :func:`augment_view_pair`
            passes False for the shared-pass diagnostic, where it has to run
            the identical chain over both views itself.

    Returns:
        Tuple of the RGB piece view and its L-mode alpha mask at the same
        size — the mask is what makes a *masked* NCC probe possible.
    """
    piece = piece_rgba if piece_rgba.mode == "RGBA" else piece_rgba.convert("RGBA")
    if not config.enabled:
        framed = frame_rgba_jittered(piece, replace(config, capture=False))
        return _composite(framed, None), framed.getchannel("A")

    piece = apply_scene_surface(piece, config, background_sampler)
    piece = apply_die_cut_edge(piece, config)
    piece = apply_segmentation_slop(piece, config)
    piece = frame_rgba_jittered(piece, config)
    piece = _augment_geometry(piece, config)

    background: Image.Image | None = None
    if config.background:
        sampler = background_sampler or BackgroundSampler()
        background = sampler.sample(piece.size, config)

    mask = piece.getchannel("A")
    rgb = composite_with_shadow(piece, background, config)

    if not config.capture:
        # The exp30 control: hand the composite to exp26's own appearance stage
        # so its photometric jitter, noise, blur and JPEG all still apply. Doing
        # only the colour jitter here would make ``--capture-preset exp30`` a
        # silently *weaker* baseline than real exp30, and every exp31 number
        # would be measured against the wrong control.
        return _augment_appearance(rgb, config), mask

    rgb = apply_piece_lighting(rgb, config)
    if config.photometric:
        rgb = _color_jitter_image(rgb, config, scale=1.0)
    if apply_chain:
        rgb = apply_view_chain(rgb, resolve_chain(config, "piece"))
    return rgb, mask


def augment_overview_capture(
    puzzle_rgb: Image.Image,
    config: CaptureConfig,
    apply_chain: bool = True,
) -> Image.Image:
    """Render one overview as an independent wide capture of a glossy box.

    exp26's ``augment_puzzle`` (mild, independently drawn photometric
    jitter) then the box-photo model then the overview's *own* capture chain
    — the coarser one, so the two views share neither a modulation transfer
    function nor a pixel lattice.

    Args:
        puzzle_rgb: The RGB source puzzle image (the "box art").
        config: Active capture config.
        apply_chain: Whether to run the overview's capture chain here.
            :func:`augment_view_pair` passes False for the shared-pass
            diagnostic.

    Returns:
        The RGB overview view, same size as the input.
    """
    rgb = augment_puzzle(puzzle_rgb, config)
    if not config.enabled or not config.capture:
        return rgb
    rgb = apply_box_photo(rgb, config)
    return apply_view_chain(rgb, resolve_chain(config, "overview")) if apply_chain else rgb


def augment_view_pair(
    piece_rgba: Image.Image,
    puzzle_rgb: Image.Image,
    config: CaptureConfig,
    background_sampler: BackgroundSampler | None = None,
    patch_source: PatchSource | None = None,
    exclude_stem: str | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Render both views as two independent captures, then apply ARP.

    The one function that must be used to produce a training pair: it is the
    only place that guarantees the two chains are run separately and that
    ARP touches exactly one view.

    When ``independent_chains`` is off, both views are instead run through
    **one** chain config with **one** drawn parameter set (:class:`ChainDraw`)
    — same phase, scale, kernels, noise sigma, substrate amplitudes, tone curve
    and JPEG quality/grid. See :class:`ChainDraw` for why the parameters have
    to be drawn up front rather than by replaying the RNG state.

    Args:
        piece_rgba: RGBA piece carrying its label rotation.
        puzzle_rgb: The RGB source puzzle image.
        config: Active capture config.
        background_sampler: Background sampler for the piece composite.
        patch_source: ARP patch content supplier.
        exclude_stem: Source stem ARP must not cut patches from.

    Returns:
        Tuple of the piece view (RGB), its mask (L) and the overview view
        (RGB).
    """
    shared = config.enabled and config.capture and not config.independent_chains
    piece_rgb, mask = augment_piece_capture(piece_rgba, config, background_sampler, apply_chain=not shared)
    overview_rgb = augment_overview_capture(puzzle_rgb, config, apply_chain=not shared)

    if shared:
        chain = resolve_chain(config, "piece")
        draw = draw_chain(chain)
        piece_rgb = apply_chain_draw(piece_rgb, chain, draw)
        overview_rgb = apply_chain_draw(overview_rgb, chain, draw)

    piece_rgb, overview_rgb = apply_arp(piece_rgb, overview_rgb, config, patch_source, exclude_stem)
    return piece_rgb, mask, overview_rgb


__all__ = [
    "CAPTURE_PRESETS",
    "JPEG_BLOCK",
    "RESAMPLE_KERNELS",
    "CaptureConfig",
    "ChainDraw",
    "PatchSource",
    "ViewChainConfig",
    "apply_arp",
    "apply_chain_draw",
    "apply_box_photo",
    "apply_die_cut_edge",
    "apply_piece_lighting",
    "apply_segmentation_slop",
    "apply_view_chain",
    "augment_overview_capture",
    "augment_piece_capture",
    "augment_view_pair",
    "capture_config_to_dict",
    "composite_with_shadow",
    "draw_chain",
    "frame_rgba_jittered",
    "overview_chain_defaults",
    "piece_chain_defaults",
    "resolve_chain",
]
