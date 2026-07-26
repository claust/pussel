"""Unit tests for the exp31 two-independent-captures augmentation.

These lock in the properties that make exp31 different from exp30, using tiny
in-memory fixtures only (no generated dataset required). The invariants that
matter are all of the form "the two views must not share X", plus the
label-safety guarantees that come with adding this much machinery:

1. The two views never share a chain draw — kernel, phase, scale, noise,
   tone or JPEG block grid (the whole point of section 4.3).
2. ARP touches **exactly one** view, never both, never neither when it fires.
3. The segmentation slop only ever moves alpha, never the piece's RGB, so no
   content (and therefore no label) can leak through it.
4. The bright rim really brightens: the boundary rim ratio goes *up* relative
   to the raw piece, unlike exp26's halo which drove it down to 0.69.
5. The crop jitter keeps exp30's border-touch acceptance probe passing —
   opaque content never reaches the input border, at any rotation.
6. The shared-pass diagnostic really is shared: with
   ``independent_chains=False`` the chain's parameters are drawn exactly once
   for both views (counted, not inferred).

Run from the network/ directory:
    uv run pytest experiments/exp31_pixel_identity/test_exp31.py -q
"""

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from ..exp24_piece_classifier.data_prep import ALPHA_THRESHOLD
from .capture import (
    CAPTURE_PRESETS,
    JPEG_BLOCK,
    RESAMPLE_KERNELS,
    CaptureConfig,
    PatchSource,
    ViewChainConfig,
    apply_arp,
    apply_die_cut_edge,
    apply_piece_lighting,
    apply_scene_surface,
    apply_segmentation_slop,
    apply_view_chain,
    augment_view_pair,
    capture_config_to_dict,
    draw_chain,
    frame_rgba_jittered,
    resolve_chain,
)
from .capture_dataset import ViewPair, deterministic_seed, view_pair

ALL_ROTATIONS = [0, 1, 2, 3]


def make_piece(width: int = 61, height: int = 79) -> Image.Image:
    """Build a small asymmetric RGBA "piece" cropped tight to its silhouette.

    Mirrors exp30's fixture: a rounded blob with one protruding tab so no
    rotation can be confused with another, over an RGB gradient so pixel
    comparisons are meaningful. The transparent region is filled with **black**
    RGB, exactly as ``cut_piece`` leaves it.

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
    draw.rectangle((0, 8, width - 1, height - 1), fill=255)
    draw.ellipse((3, 0, 22, 20), fill=255)
    piece.paste(rgb, mask=mask)
    return piece.crop(piece.getbbox() or (0, 0, width, height))


def make_overview(size: int = 128) -> Image.Image:
    """Build a textured RGB "overview" with content at every frequency.

    Args:
        size: Square side length.

    Returns:
        An RGB image.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    arr = np.stack(
        [
            128 + 90 * np.sin(xx / 3.0),
            128 + 90 * np.sin(yy / 7.0),
            128 + 60 * np.sin((xx + yy) / 11.0),
        ],
        axis=-1,
    )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def border_alpha_max(image: Image.Image) -> int:
    """Return the maximum alpha on the one-pixel border of an RGBA image.

    Args:
        image: RGBA image.

    Returns:
        The largest alpha found on the outermost row/column ring.
    """
    alpha = np.asarray(image.getchannel("A"))
    ring = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
    return int(ring.max())


def rim_ratio(rgb: Image.Image, mask: Image.Image, band: int = 3, inner: int = 6) -> float | None:
    """Return boundary luminance divided by interior luminance.

    The section 5 metric: real pieces measure 1.08 (bright cardboard rim),
    raw synthetic 0.98, exp26-augmented 0.69 (the wrong direction).

    Args:
        rgb: The RGB view.
        mask: The matching L-mode alpha mask.
        band: Boundary band width in pixels.
        inner: Distance beyond which pixels count as interior.

    Returns:
        The ratio, or None when the mask is too small to measure.
    """
    import cv2

    binary = (np.asarray(mask) > ALPHA_THRESHOLD).astype(np.uint8)
    if binary.sum() < 100:
        return None
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    luma = np.asarray(rgb.convert("L"), dtype=np.float32)
    edge = (distance > 0) & (distance <= band)
    core = distance > inner
    if edge.sum() < 20 or core.sum() < 20:
        return None
    return float(luma[edge].mean() / max(1e-6, luma[core].mean()))


def test_chain_draws_are_never_shared_between_views() -> None:
    """Two runs of the chain over the same input give different outputs.

    Not a statistical claim about realism — a structural one. If the chain
    ever reused a draw the two views would land on the same kernel, phase,
    scale, noise field, tone curve and JPEG grid, which is the exp26 failure
    mode restated.
    """
    source = make_overview()
    config = CaptureConfig()
    random.seed(0)
    np.random.seed(0)
    first = np.asarray(apply_view_chain(source, config.piece_chain), dtype=np.int16)
    second = np.asarray(apply_view_chain(source, config.piece_chain), dtype=np.int16)
    assert not np.array_equal(first, second)
    # And meaningfully different, not one stray pixel.
    assert np.abs(first - second).mean() > 1.0


def test_overview_chain_is_coarser_than_the_piece_chain() -> None:
    """The asymmetry direction: the overview is resolved worse than the piece.

    Absolute resolution cannot be fixed on a 256 px corpus, but the two views
    must not share a modulation transfer function. Disjoint sampling ranges
    guarantee the direction by construction.
    """
    config = CaptureConfig()
    assert config.overview_chain.scale_max < config.piece_chain.scale_min
    assert resolve_chain(config, "piece") is config.piece_chain
    assert resolve_chain(config, "overview") is config.overview_chain


def test_resolution_asymmetry_off_matches_the_sampling_budget() -> None:
    """With the asymmetry off the overview borrows the piece's budget."""
    config = CaptureConfig(resolution_asymmetry=False)
    overview = resolve_chain(config, "overview")
    assert overview.scale_min == config.piece_chain.scale_min
    assert overview.scale_max == config.piece_chain.scale_max
    # Still the overview's own chain in every other respect.
    assert overview.sharpen_p == config.overview_chain.sharpen_p


def test_resolve_chain_rejects_unknown_views() -> None:
    """A typo'd view name is a programming error, not a silent piece chain."""
    with pytest.raises(ValueError, match="piece.*overview"):
        resolve_chain(CaptureConfig(), "puzzle")


def test_jpeg_grid_offset_shifts_the_block_lattice() -> None:
    """The JPEG stage really does move the 8x8 grid, not just the quality.

    Encoding the same image with two different grid offsets must give two
    different results; a shared block lattice is a correlated artifact both
    views would carry.
    """
    source = make_overview(size=64)
    chain = ViewChainConfig(resample=False, sharpen_p=0.0, tone=False, noise=False, optical_blur_p=0.0, jpeg_p=1.0)
    outputs = set()
    for seed in range(12):
        random.seed(seed)
        np.random.seed(seed)
        outputs.add(np.asarray(apply_view_chain(source, chain)).tobytes())
    # 12 draws over 8x8 offsets x quality: several distinct realizations.
    assert len(outputs) > 4
    assert JPEG_BLOCK == 8


def test_substrate_field_is_multiplicative_and_spatially_varying() -> None:
    """The substrate stage is a *field*, not a global gain.

    This is the lever that carried the §8 headroom (GT median 0.847 → 0.743): a
    zero-mean masked NCC cancels global affine intensity maps, so only a
    spatially varying multiplicative field bites. If a refactor ever collapsed
    it to a scalar gain the gate would silently regress, hence the explicit
    ratio-variance assertion.
    """
    source = Image.new("RGB", (128, 128), (120, 120, 120))
    chain = ViewChainConfig(resample=False, sharpen_p=0.0, tone=False, noise=False, optical_blur_p=0.0, jpeg_p=0.0)
    random.seed(4)
    np.random.seed(4)
    out = np.asarray(apply_view_chain(source, chain), dtype=np.float32)

    ratio = out / 120.0
    assert ratio.std() > 0.01, ratio.std()
    # Two octaves: variance survives both a heavy blur (coarse) and its removal
    # (fine), so the field is not single-scale.
    coarse = np.asarray(Image.fromarray(out.astype(np.uint8)).filter(ImageFilter.GaussianBlur(8)), dtype=np.float32)
    assert (coarse / 120.0).std() > 0.005
    assert (out - coarse).std() > 0.3


def test_substrate_fields_are_independent_per_view() -> None:
    """Two chain runs give two different fields, never a shared one."""
    source = Image.new("RGB", (96, 96), (140, 130, 120))
    chain = ViewChainConfig(resample=False, sharpen_p=0.0, tone=False, noise=False, optical_blur_p=0.0, jpeg_p=0.0)
    random.seed(0)
    np.random.seed(0)
    first = np.asarray(apply_view_chain(source, chain), dtype=np.int16)
    second = np.asarray(apply_view_chain(source, chain), dtype=np.int16)
    assert np.abs(first - second).mean() > 1.0


def test_piece_lighting_varies_spatially_and_is_optional() -> None:
    """The piece gets its own illumination, and it can be switched off."""
    flat = Image.new("RGB", (128, 128), (128, 128, 128))
    random.seed(6)
    np.random.seed(6)
    lit = np.asarray(apply_piece_lighting(flat, CaptureConfig()), dtype=np.float32)
    assert lit.std() > 1.0

    unlit = apply_piece_lighting(flat, CaptureConfig(piece_lighting=False))
    assert unlit is flat
    assert apply_piece_lighting(flat, CaptureConfig(capture=False)) is flat


def test_arp_touches_exactly_one_view() -> None:
    """ARP modifies one view or neither — never both.

    Patching both views would paste *shared* content, which is the shortcut
    ARP exists to destroy (Chuah et al., 2106.08486).
    """
    config = CaptureConfig(arp_p=1.0)
    piece = make_overview(size=96)
    overview = make_overview(size=192)
    source = PatchSource()

    changed_counts = []
    for seed in range(20):
        random.seed(seed)
        np.random.seed(seed)
        out_piece, out_overview = apply_arp(piece, overview, config, source)
        changed = int(not np.array_equal(np.asarray(out_piece), np.asarray(piece))) + int(
            not np.array_equal(np.asarray(out_overview), np.asarray(overview))
        )
        changed_counts.append(changed)
    assert set(changed_counts) == {1}, changed_counts


def test_arp_can_patch_either_view() -> None:
    """Over many draws ARP hits the piece sometimes and the overview others."""
    config = CaptureConfig(arp_p=1.0)
    piece = make_overview(size=96)
    overview = make_overview(size=192)
    source = PatchSource()

    piece_hits = 0
    for seed in range(40):
        random.seed(seed)
        np.random.seed(seed)
        out_piece, _ = apply_arp(piece, overview, config, source)
        piece_hits += int(not np.array_equal(np.asarray(out_piece), np.asarray(piece)))
    assert 5 < piece_hits < 35, piece_hits


def test_arp_respects_its_probability() -> None:
    """With ``arp=False`` (or p=0) both views come back untouched."""
    piece = make_overview(size=96)
    overview = make_overview(size=192)
    for config in (CaptureConfig(arp=False), CaptureConfig(arp_p=0.0), CaptureConfig(capture=False)):
        out_piece, out_overview = apply_arp(piece, overview, config, PatchSource())
        assert out_piece is piece
        assert out_overview is overview


def test_arp_patch_size_scales_with_the_view() -> None:
    """Patch size is a fraction of the view's short side, not a fixed px range.

    The doc's 50-100 px comes from KITTI stereo (13-27% of a 375 px short
    side); on a 128 px piece a fixed 50-100 px patch would cover the whole
    view.
    """
    config = CaptureConfig()
    for side in (128, 256):
        smallest = round(side * config.arp_size_min_frac)
        largest = round(side * config.arp_size_max_frac)
        assert 3 <= smallest < largest < side // 2


def test_segmentation_slop_moves_alpha_only() -> None:
    """The slop model never touches RGB, so no content can leak through it."""
    piece = apply_scene_surface(make_piece(), CaptureConfig())
    random.seed(3)
    np.random.seed(3)
    slopped = apply_segmentation_slop(piece, CaptureConfig())

    assert np.array_equal(np.asarray(piece.convert("RGB")), np.asarray(slopped.convert("RGB")))
    assert not np.array_equal(np.asarray(piece.getchannel("A")), np.asarray(slopped.getchannel("A")))


def test_segmentation_slop_softens_the_boundary() -> None:
    """The exact lattice cut gains a rembg-like transition band.

    ``segmentation_validation.py`` is where this is quantified against real
    u2net output; here we only assert the direction, so a future config change
    that removes the softness entirely fails loudly.
    """
    piece = apply_scene_surface(make_piece(), CaptureConfig())
    exact = np.asarray(piece.getchannel("A"), dtype=np.float32) / 255.0
    random.seed(5)
    np.random.seed(5)
    slopped = np.asarray(apply_segmentation_slop(piece, CaptureConfig()).getchannel("A"), dtype=np.float32) / 255.0

    def partial(alpha: np.ndarray) -> int:
        return int(((alpha > 0.05) & (alpha < 0.95)).sum())

    assert partial(slopped) > partial(exact)


def test_scene_surface_replaces_the_generator_black_fill() -> None:
    """Out-of-silhouette RGB stops being black, so dilation cannot reveal it.

    ``cut_piece`` writes black RGB wherever alpha is zero, so without this
    stage every mask dilation would show pure black — measured as a heavy dark
    tail on the rim ratio (25th percentile 0.57).
    """
    piece = make_piece()
    outside_before = np.asarray(piece.convert("RGB"))[np.asarray(piece.getchannel("A")) <= ALPHA_THRESHOLD]
    assert outside_before.size > 0
    assert outside_before.max() == 0

    random.seed(7)
    np.random.seed(7)
    surfaced = apply_scene_surface(piece, CaptureConfig())
    outside_after = np.asarray(surfaced.convert("RGB"))[np.asarray(surfaced.getchannel("A")) <= ALPHA_THRESHOLD]
    assert outside_after.mean() > 10.0
    # Larger canvas: the rim band and the dilation need room.
    assert surfaced.width > piece.width and surfaced.height > piece.height
    # Alpha inside the silhouette is untouched.
    assert int((np.asarray(surfaced.getchannel("A")) > ALPHA_THRESHOLD).sum()) == int(
        (np.asarray(piece.getchannel("A")) > ALPHA_THRESHOLD).sum()
    )


def test_die_cut_rim_brightens_the_boundary() -> None:
    """The rim pushes the boundary ratio UP, the opposite of exp26's halo.

    Section 5: real 1.08, synthetic 0.98, exp26's halo augmentation 0.69.
    exp31 must not inherit the wrong-direction halo.
    """
    piece = apply_scene_surface(make_piece(), CaptureConfig(scene_surface=True))
    mask = piece.getchannel("A")
    before = rim_ratio(piece.convert("RGB"), mask)
    random.seed(11)
    np.random.seed(11)
    after = rim_ratio(apply_die_cut_edge(piece, CaptureConfig()).convert("RGB"), mask)

    assert before is not None and after is not None
    assert after > before


def test_exp26_halo_is_off_by_default() -> None:
    """exp26's mask erode/dilate is disabled, along with its noise and JPEG.

    All three are superseded: halo by ``rim`` + ``seg_slop`` (it pushed the rim
    ratio the wrong way), noise and JPEG by the per-view chains (which apply
    noise at the sampled resolution and JPEG on an independent grid).
    """
    config = CaptureConfig()
    assert config.halo is False
    assert config.noise is False
    assert config.jpeg is False
    # ...but the exp26 stages that stay are still on.
    assert config.photometric and config.background and config.scale_jitter and config.perspective


@pytest.mark.parametrize("rotation_idx", ALL_ROTATIONS)
def test_crop_jitter_never_touches_the_border(rotation_idx: int) -> None:
    """exp30's section 4.1 acceptance probe keeps passing under crop jitter.

    The jitter is biased outward (2% in, 5% out) against a margin of at least
    5%, so the silhouette cannot reach the border for any rotation or draw.

    Args:
        rotation_idx: Rotation index under test.
    """
    from ..exp30_generator_fixes.framing import rotate_lossless

    piece = make_piece()
    config = CaptureConfig()
    for seed in range(15):
        random.seed(seed * 4 + rotation_idx)
        np.random.seed(seed * 4 + rotation_idx)
        rotated = rotate_lossless(piece, rotation_idx * 90)
        framed = frame_rgba_jittered(apply_scene_surface(rotated, config), config)
        assert framed.width == framed.height
        assert border_alpha_max(framed) == 0, f"seed {seed}"


def test_crop_jitter_is_actually_random() -> None:
    """Two draws give two different framings (otherwise gap (b) is unmodelled)."""
    piece = apply_scene_surface(make_piece(), CaptureConfig())
    config = CaptureConfig()
    sizes = set()
    for seed in range(10):
        random.seed(seed)
        np.random.seed(seed)
        sizes.add(frame_rgba_jittered(piece, config).size)
    assert len(sizes) > 1


def test_crop_jitter_off_reproduces_exp30_framing() -> None:
    """With the jitter off the geometry is exp30's 8%-margin square pad."""
    from ..exp30_generator_fixes.framing import frame_rgba

    piece = make_piece()
    jittered = frame_rgba_jittered(piece, CaptureConfig(crop_jitter=False))
    assert jittered.size == frame_rgba(piece).size
    assert np.array_equal(np.asarray(jittered), np.asarray(frame_rgba(piece)))


def test_shared_chain_diagnostic_draws_parameters_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``independent_chains=False`` draws ONE parameter set for both views.

    The ablation that has to fail, and the property is counted rather than
    inferred. Sharing only the chain *config* would leave the control secretly
    independent; so would the obvious "save the RNG state, run twice, restore"
    trick, because the two views have different pixel dimensions and the
    size-dependent noise/substrate draws desynchronise everything after them
    (see :class:`capture.ChainDraw`).
    """
    from . import capture as capture_module

    calls: list[int] = []
    real_draw = capture_module.draw_chain

    def counting_draw(chain: ViewChainConfig) -> object:
        calls.append(1)
        return real_draw(chain)

    monkeypatch.setattr(capture_module, "draw_chain", counting_draw)

    piece = make_piece()
    overview = make_overview(size=256)

    random.seed(1)
    np.random.seed(1)
    augment_view_pair(piece, overview, CaptureConfig(independent_chains=False))
    assert len(calls) == 1, calls

    calls.clear()
    random.seed(1)
    np.random.seed(1)
    augment_view_pair(piece, overview, CaptureConfig())
    assert len(calls) == 2, calls


def test_chain_draw_carries_every_parameter() -> None:
    """A drawn chain is fully specified, so replaying it needs no RNG state."""
    draw = draw_chain(ViewChainConfig(optical_blur_p=1.0, sharpen_p=1.0, jpeg_p=1.0))
    assert len(draw.phase) == 2
    assert draw.blur_sigma is not None
    assert 0.0 < draw.sample_factor <= 1.0
    assert draw.down_kernel in RESAMPLE_KERNELS and draw.up_kernel in RESAMPLE_KERNELS
    assert draw.noise_sigma is not None
    assert len(draw.substrate) == 2
    assert draw.sharpen is not None
    assert draw.tone is not None and len(draw.tone[0]) == 3
    assert draw.jpeg is not None and 0 <= draw.jpeg[1] < JPEG_BLOCK and 0 <= draw.jpeg[2] < JPEG_BLOCK


def test_augment_view_pair_returns_aligned_mask() -> None:
    """The returned mask matches the piece view's size, for masked NCC."""
    random.seed(1)
    np.random.seed(1)
    piece_rgb, mask, overview_rgb = augment_view_pair(make_piece(), make_overview(size=256), CaptureConfig())
    assert piece_rgb.size == mask.size
    assert piece_rgb.width == piece_rgb.height
    assert mask.mode == "L"
    assert overview_rgb.size == (256, 256)


def test_every_preset_builds_and_runs() -> None:
    """All ablation presets are constructible, serializable and runnable."""
    piece = make_piece()
    overview = make_overview(size=128)
    for name, config in CAPTURE_PRESETS.items():
        flags = config.ablation_flags()
        assert "arp" in flags and "capture" in flags, name
        payload = capture_config_to_dict(config)
        assert isinstance(payload["piece_chain"], dict), name
        random.seed(2)
        np.random.seed(2)
        piece_rgb, mask, overview_rgb = augment_view_pair(piece, overview, config)
        assert piece_rgb.size == mask.size, name
        assert overview_rgb.size == (128, 128), name


def test_exp30_preset_reinstates_the_exp26_stages() -> None:
    """``--capture-preset exp30`` is a real control, not a relabelled full run."""
    config = CAPTURE_PRESETS["exp30"]
    assert config.capture is False
    assert config.halo and config.noise and config.jpeg
    flags = config.ablation_flags()
    assert not any(flags[name] for name in ("capture", "arp", "seg_slop", "rim", "box_photo", "crop_jitter"))


def _write_piece(tmp_path: Path) -> tuple[Path, Path]:
    """Write a fixture piece and overview to disk in the dataset layout.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Tuple of (piece path, puzzle path).
    """
    puzzle_dir = tmp_path / "puzzle_test"
    puzzle_dir.mkdir(parents=True, exist_ok=True)
    piece_path = puzzle_dir / "puzzle_test_x0.375_y0.625_rot90.png"
    make_piece().save(piece_path, "PNG")
    puzzle_path = tmp_path / "puzzle_test.jpg"
    make_overview(size=256).save(puzzle_path, "JPEG", quality=92)
    return piece_path, puzzle_path


def test_view_pair_is_the_probe_contract(tmp_path: Path) -> None:
    """``view_pair`` returns both views at both scales, plus masks and labels.

    This is the interface the section 8 acceptance probes consume, so its
    shape is a contract: breaking it breaks the gate.
    """
    piece_path, puzzle_path = _write_piece(tmp_path)
    pair = view_pair(piece_path, puzzle_path, CaptureConfig(), applied_rotation_idx=1, seed=17)

    assert isinstance(pair, ViewPair)
    assert pair.piece_input.size == (128, 128)
    assert pair.piece_mask_input.size == (128, 128)
    assert pair.overview_input.size == (256, 256)
    assert pair.piece_native.size == pair.piece_mask_native.size
    assert pair.piece_raw_native.size == pair.piece_raw_mask_native.size
    assert pair.overview_raw_native.size == (256, 256)
    assert pair.puzzle_id == "puzzle_test"
    assert (pair.cx, pair.cy) == (0.375, 0.625)
    # base 90 + applied 90 = 180 -> class 2
    assert pair.rotation_idx == 2


def test_view_pair_is_reproducible_from_a_seed(tmp_path: Path) -> None:
    """The same seed gives the same realization; a different seed does not."""
    piece_path, puzzle_path = _write_piece(tmp_path)
    first = view_pair(piece_path, puzzle_path, CaptureConfig(), seed=5)
    again = view_pair(piece_path, puzzle_path, CaptureConfig(), seed=5)
    other = view_pair(piece_path, puzzle_path, CaptureConfig(), seed=6)

    assert np.array_equal(np.asarray(first.piece_input), np.asarray(again.piece_input))
    assert np.array_equal(np.asarray(first.overview_input), np.asarray(again.overview_input))
    assert not np.array_equal(np.asarray(first.piece_input), np.asarray(other.piece_input))


def test_view_pair_raw_baseline_is_undegraded(tmp_path: Path) -> None:
    """The ``*_raw_*`` fields carry exp30's clean views, not exp31's.

    The probe needs the "synthetic raw" 0.99 baseline and the exp31 number in
    the same framing; if the raw fields were degraded too there would be
    nothing to compare against.
    """
    piece_path, puzzle_path = _write_piece(tmp_path)
    pair = view_pair(piece_path, puzzle_path, CaptureConfig(), seed=9)

    with Image.open(puzzle_path) as source:
        assert np.array_equal(np.asarray(pair.overview_raw_native), np.asarray(source.convert("RGB")))
    assert not np.array_equal(
        np.asarray(pair.overview_native.resize(pair.overview_raw_native.size)),
        np.asarray(pair.overview_raw_native),
    )


def test_deterministic_seed_is_stable_and_rotation_aware() -> None:
    """Seeds depend on the filename and the rotation, not on iteration order."""
    first = deterministic_seed("datasets/x/puzzle_007_x0.125_y0.125_rot0.png", 0)
    same = deterministic_seed(Path("elsewhere/puzzle_007_x0.125_y0.125_rot0.png"), 0)
    rotated = deterministic_seed("datasets/x/puzzle_007_x0.125_y0.125_rot0.png", 1)
    other = deterministic_seed("datasets/x/puzzle_008_x0.125_y0.125_rot0.png", 0)

    assert first == same
    assert first != rotated
    assert first != other
    assert 0 <= first < 2**31
