"""Synthetic self-tests for exp29: `score_stitch.py` metrics and `stitch.py` end to end.

Everything is generated in-process with numpy/cv2 -- no fixtures on disk:

1. A random, richly textured "puzzle" scene (grid of colored rectangles/circles over
   blurred noise) for SIFT to key on.
2. A reference view and 4 corner views, each a small random perspective warp of the
   scene (simulating a different camera pose), with a soft glare disc painted onto
   each view at a DIFFERENT position -- mirroring the real 5-shot technique, where
   darkest-pixel-wins across shots removes glare because no two shots glare at the
   same spot.
3. A perfectly aligned composite, built by warping each corner view back onto the
   reference frame with the KNOWN inverse homography, and a deliberately misaligned
   composite, where one frame is additionally shifted ~8px before compositing.

`score_stitch.py`'s metrics must separate the two (misaligned has higher ghosting p95
and edge-doubling; aligned scores near-clean). `stitch.py`'s own SIFT-based
registration -- independent of the known homographies used to build the fixtures
above -- must still reduce glare relative to the raw reference when run on the raw
corner shots.

Also covered, with their own smaller, purpose-built fixtures (not the 5-shot one above):
the glare-healing (darkening-map) metric on two constructed cases -- glare that persists
identically in the composite (nothing healed) vs. a non-saturating matte sheen the
composite heals (which `GlareReductionMetrics`'s near-saturation check is blind to); the
healed-patch exclusion mechanism in `compute_local_ghosting`, which must prefer a
genuinely misaligned patch over a more-darkened one with a larger raw phase-correlation
shift; and the `--quad` region-restriction plumbing.

Run: `uv run pytest experiments/exp29_stitch_quality/test_exp29.py` (from `network/`).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import score_stitch  # noqa: E402
import stitch  # noqa: E402
from common import (  # noqa: E402
    DARKENED_PIXEL_THRESHOLD,
    HEALED_PATCH_DARKENING_THRESHOLD,
    CaptureDump,
    compute_darkening_map,
    near_saturated_fraction,
    phase_correlation_grid,
)

SCENE_WIDTH, SCENE_HEIGHT = 640, 480
CORNER_JITTER_PX = 18.0
MISALIGN_SHIFT_PX = 8.0
GLARE_RADIUS_PX = 55
SEED = 42


def _make_scene(rng: np.random.Generator) -> np.ndarray:
    """Render a colorful, richly textured synthetic "puzzle" scene for SIFT to key on.

    Args:
        rng: Numpy random generator.

    Returns:
        BGR image (SCENE_HEIGHT, SCENE_WIDTH, 3).
    """
    scene = rng.integers(0, 255, size=(SCENE_HEIGHT, SCENE_WIDTH, 3), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (0, 0), sigmaX=1.5)  # smooth pure noise into blobs, not salt-and-pepper
    grid = 8
    for row in range(grid):
        for col in range(grid):
            y0, y1 = row * SCENE_HEIGHT // grid, (row + 1) * SCENE_HEIGHT // grid
            x0, x1 = col * SCENE_WIDTH // grid, (col + 1) * SCENE_WIDTH // grid
            color = tuple(int(c) for c in rng.integers(40, 220, size=3))
            cv2.rectangle(scene, (x0, y0), (x1, y1), color, thickness=-1 if rng.random() < 0.5 else 3)
            if rng.random() < 0.5:
                circle_color = tuple(int(c) for c in rng.integers(0, 255, size=3))
                cv2.circle(scene, ((x0 + x1) // 2, (y0 + y1) // 2), min(x1 - x0, y1 - y0) // 3, circle_color, -1)
    return scene


def _random_homography(size: Tuple[int, int], jitter_px: float, rng: np.random.Generator) -> np.ndarray:
    """A small random perspective homography mapping the full-frame corners to jittered corners.

    Args:
        size: (width, height) of the frame.
        jitter_px: Max per-corner jitter, in pixels.
        rng: Numpy random generator.

    Returns:
        3x3 homography.
    """
    width, height = size
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    dst = src + rng.uniform(-jitter_px, jitter_px, size=src.shape).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _add_glare(image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    """Paint a soft near-white glare disc onto a copy of `image`.

    Args:
        image: Source BGR image.
        center: Disc center, (x, y).
        radius: Disc radius in pixels.

    Returns:
        A new BGR image with the glare disc composited in.
    """
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, thickness=-1)
    # A small blur fraction keeps the disc's center genuinely saturated (soft only at the rim) --
    # too large a sigma relative to the radius washes out the center too, defeating the fixture.
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.2)
    alpha = mask[..., None]
    white = np.full_like(image, 255)
    return (image.astype(np.float32) * (1 - alpha) + white.astype(np.float32) * alpha).astype(np.uint8)


def _translation_matrix(dx: float, dy: float) -> np.ndarray:
    """A 3x3 homography representing a pure pixel translation.

    Args:
        dx: X translation in pixels.
        dy: Y translation in pixels.

    Returns:
        3x3 homogeneous translation matrix.
    """
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


class SyntheticFixture:
    """A synthetic 5-shot glare-free capture, plus ground-truth aligned/misaligned composites.

    Attributes:
        reference: Centered reference shot (identity pose), with a glare disc.
        corner_shots: Raw corner shots (as "captured", each its own camera pose + glare disc),
            keyed by 1-based index -- this is what `stitch.py` registers from scratch.
        aligned_composite: Ground-truth-registered min-composite (near-zero ghosting by construction).
        misaligned_composite: Same, but corner 1 is additionally shifted ~8px before compositing.
    """

    def __init__(self) -> None:
        """Build the scene, the 5 shots, and the aligned/misaligned ground-truth composites."""
        rng = np.random.default_rng(SEED)
        size = (SCENE_WIDTH, SCENE_HEIGHT)
        scene = _make_scene(rng)

        # Reference: identity pose, glare centered.
        self.reference = _add_glare(scene, (SCENE_WIDTH // 2, SCENE_HEIGHT // 2), GLARE_RADIUS_PX)

        # One glare position per corner, in a different quadrant from the reference and each other.
        glare_centers = {
            1: (SCENE_WIDTH // 5, SCENE_HEIGHT // 5),
            2: (4 * SCENE_WIDTH // 5, SCENE_HEIGHT // 5),
            3: (SCENE_WIDTH // 5, 4 * SCENE_HEIGHT // 5),
            4: (4 * SCENE_WIDTH // 5, 4 * SCENE_HEIGHT // 5),
        }

        homographies: Dict[int, np.ndarray] = {}
        self.corner_shots: Dict[int, np.ndarray] = {}
        aligned_frames: Dict[int, np.ndarray] = {}
        for index, center in glare_centers.items():
            homography = _random_homography(size, CORNER_JITTER_PX, rng)
            homographies[index] = homography
            corner_raw = cv2.warpPerspective(scene, homography, size, borderValue=(255, 255, 255))
            self.corner_shots[index] = _add_glare(corner_raw, center, GLARE_RADIUS_PX)
            inverse = np.linalg.inv(homography)
            aligned_frames[index] = cv2.warpPerspective(
                self.corner_shots[index], inverse, size, borderValue=(255, 255, 255)
            )

        self.aligned_composite = self.reference.copy()
        for frame in aligned_frames.values():
            self.aligned_composite = np.minimum(self.aligned_composite, frame)

        misaligned_frame_1 = cv2.warpPerspective(
            self.corner_shots[1],
            _translation_matrix(MISALIGN_SHIFT_PX, MISALIGN_SHIFT_PX) @ np.linalg.inv(homographies[1]),
            size,
            borderValue=(255, 255, 255),
        )
        self.misaligned_composite = self.reference.copy()
        for index, frame in aligned_frames.items():
            source = misaligned_frame_1 if index == 1 else frame
            self.misaligned_composite = np.minimum(self.misaligned_composite, source)

    def write_dump(self, dump_dir: Path, composite: np.ndarray, with_metadata: bool = True) -> None:
        """Write this fixture to disk as a capture-dump directory.

        Args:
            dump_dir: Destination directory (created if missing).
            composite: Which composite to write as `composite.jpg`.
            with_metadata: Whether to also write a `metadata.json`.
        """
        dump_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dump_dir / "reference.jpg"), self.reference)
        cv2.imwrite(str(dump_dir / "composite.jpg"), composite)
        for index, corner in self.corner_shots.items():
            cv2.imwrite(str(dump_dir / f"corner_{index}.jpg"), corner)
        if with_metadata:
            (dump_dir / "metadata.json").write_text(
                '{"timestamp": "2026-07-24T00:00:00Z", "expectedShifts": [], "alignedFrameCount": 4}'
            )


@pytest.fixture(scope="module")
def fixture() -> SyntheticFixture:
    """Build the synthetic 5-shot capture once per test module (deterministic, no I/O)."""
    return SyntheticFixture()


def _report(
    fixture: SyntheticFixture, composite: np.ndarray, quad_unit: Optional[np.ndarray] = None
) -> score_stitch.StitchQualityReport:
    """Score a synthetic composite against the fixture's reference, skipping disk I/O."""
    dump = CaptureDump(
        dump_dir=Path("<synthetic>"),
        reference=fixture.reference,
        composite=composite,
        corners=fixture.corner_shots,
        metadata=None,
        scale=1.0,
    )
    report, _patches = score_stitch.score_dump(dump, quad_unit=quad_unit)
    return report


def test_aligned_composite_scores_near_clean(fixture: SyntheticFixture) -> None:
    """A perfectly (ground-truth) registered composite should show minimal ghosting."""
    report = _report(fixture, fixture.aligned_composite)

    assert report.local_ghosting.p95_shift_px is not None
    assert report.local_ghosting.p95_shift_px < 2.0

    assert report.global_geometry.identity_deviation_rms_px is not None
    assert report.global_geometry.identity_deviation_rms_px < 2.0

    assert 0.85 < report.edge_doubling.canny_edge_ratio < 1.2


def test_misaligned_composite_shows_more_ghosting_and_edge_doubling(fixture: SyntheticFixture) -> None:
    """A deliberately ~8px-misaligned frame should push ghosting and edge metrics up."""
    aligned_report = _report(fixture, fixture.aligned_composite)
    misaligned_report = _report(fixture, fixture.misaligned_composite)

    assert aligned_report.local_ghosting.p95_shift_px is not None
    assert misaligned_report.local_ghosting.p95_shift_px is not None
    assert misaligned_report.local_ghosting.p95_shift_px > aligned_report.local_ghosting.p95_shift_px

    # The shifted frame introduces a genuine edge-doubling signal relative to the clean case.
    assert misaligned_report.edge_doubling.canny_edge_ratio > aligned_report.edge_doubling.canny_edge_ratio


def test_glare_reduction_favors_composite_over_reference(fixture: SyntheticFixture) -> None:
    """Both composites should remove far more glare than the raw reference has -- the benefit metric."""
    for composite in (fixture.aligned_composite, fixture.misaligned_composite):
        report = _report(fixture, composite)
        assert report.glare_reduction.reference_saturated_fraction > 0.0
        assert report.glare_reduction.composite_saturated_fraction < report.glare_reduction.reference_saturated_fraction
        assert report.glare_reduction.reduction_factor > 2.0


def test_glare_healing_zero_when_glare_persists_in_composite() -> None:
    """Glare-that-darkens-nothing case: composite darkens NOTHING and healing metrics reflect that.

    If every candidate frame had glare at the same spot (nothing for darkest-pixel-wins to
    select instead), the composite is identical to the reference there -- the healing metrics
    must report ~0 darkening, not just "some glare exists".
    """
    rng = np.random.default_rng(100)
    scene = _make_scene(rng)
    reference = _add_glare(scene, (SCENE_WIDTH // 2, SCENE_HEIGHT // 2), GLARE_RADIUS_PX)
    composite = reference.copy()  # composite couldn't remove the glare -- identical to reference

    gray_composite = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    darkening_map = compute_darkening_map(gray_composite, gray_reference)
    healing = score_stitch.compute_glare_healing(darkening_map)

    assert healing.darkened_fraction == pytest.approx(0.0, abs=1e-6)
    assert healing.mean_darkening_over_darkened == 0.0
    assert healing.p95_darkening == pytest.approx(0.0, abs=1e-6)


def test_glare_healing_detects_matte_sheen_saturation_misses() -> None:
    """Sheen-style-glare-the-composite-darkens case: a broad, non-saturating haze the composite heals.

    The haze is capped well below 250/255 gray, so `GlareReductionMetrics` (near-saturation)
    is blind to it on BOTH images; `GlareHealingMetrics` must still detect the composite's
    healing via the darkening map, since it doesn't require either image to saturate.
    """
    rng = np.random.default_rng(101)
    scene = _make_scene(rng)

    # A broad, moderate-alpha wash (not full-strength _add_glare) -- stays under gray 250.
    mask = np.zeros((SCENE_HEIGHT, SCENE_WIDTH), dtype=np.float32)
    cv2.circle(mask, (SCENE_WIDTH // 2, SCENE_HEIGHT // 2), 140, 0.5, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=40)
    white = np.full_like(scene, 255)
    alpha = mask[..., None]
    reference = (scene.astype(np.float32) * (1 - alpha) + white.astype(np.float32) * alpha).astype(np.uint8)
    composite = scene.copy()  # this frame healed the sheen entirely

    gray_composite = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    assert gray_reference.max() < 250, "fixture must stay below saturation to test the matte case"
    assert near_saturated_fraction(gray_reference) == 0.0
    assert near_saturated_fraction(gray_composite) == 0.0

    darkening_map = compute_darkening_map(gray_composite, gray_reference)
    healing = score_stitch.compute_glare_healing(darkening_map)

    assert healing.darkened_fraction > 0.05
    assert healing.mean_darkening_over_darkened > DARKENED_PIXEL_THRESHOLD
    assert healing.p95_darkening > 0.0


def test_healed_patch_excluded_from_worst_patch_selection() -> None:
    """A more-darkened, larger-raw-shift patch must lose to a genuinely misaligned one.

    Constructs a scene with two touched patches: patch A gets a real ~8px content shift
    (genuine ghosting, low darkening); patch B's composite content is swapped for an
    unrelated crop (a large, organic phase-correlation shift reading) AND its reference is
    washed toward white (genuine high darkening) -- simulating a region so glare-washed that
    even phase correlation partly loses the plot. Without darkening-aware exclusion, patch B
    would incorrectly win as "worst"; with it, patch A must win.
    """
    rng = np.random.default_rng(11)
    scene = _make_scene(rng)
    patch_size = 64

    reference = scene.copy()
    composite = scene.copy()

    shift_row, shift_col = 1, 1
    sx, sy = shift_col * patch_size, shift_row * patch_size
    composite[sy : sy + patch_size, sx : sx + patch_size] = scene[
        sy - 6 : sy + patch_size - 6, sx - 6 : sx + patch_size - 6
    ]

    heal_row, heal_col = 3, 5
    hx, hy = heal_col * patch_size, heal_row * patch_size
    composite[hy : hy + patch_size, hx : hx + patch_size] = scene[10 : 10 + patch_size, 500 : 500 + patch_size]
    mask = np.zeros(scene.shape[:2], dtype=np.float32)
    cv2.circle(mask, (hx + patch_size // 2, hy + patch_size // 2), 50, 1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10)
    white = np.full_like(reference, 255)
    alpha = mask[..., None]
    reference = (reference.astype(np.float32) * (1 - alpha) + white.astype(np.float32) * alpha).astype(np.uint8)

    gray_composite = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    darkening_map = compute_darkening_map(gray_composite, gray_reference)

    # Sanity-check the fixture actually exercises the interesting case: patch B's raw shift
    # must beat patch A's, and patch B's darkening must clear the healed-exclusion threshold.
    raw_patches = phase_correlation_grid(gray_composite, gray_reference, patch_size)
    by_cell = {(p.row, p.col): p for p in raw_patches}
    patch_a_raw = by_cell[(shift_row, shift_col)]
    patch_b_raw = by_cell[(heal_row, heal_col)]
    assert patch_b_raw.shift_px > patch_a_raw.shift_px
    patch_b_darkening = float(darkening_map[hy : hy + patch_size, hx : hx + patch_size].mean())
    assert patch_b_darkening > HEALED_PATCH_DARKENING_THRESHOLD

    ghosting, patches = score_stitch.compute_local_ghosting(gray_composite, gray_reference, darkening_map, patch_size)

    assert ghosting.healed_patches >= 1
    annotated = {(p.row, p.col): p for p in patches}
    assert annotated[(heal_row, heal_col)].healed is True
    assert annotated[(shift_row, shift_col)].healed is False

    assert ghosting.worst_patch is not None
    assert (ghosting.worst_patch.row, ghosting.worst_patch.col) == (shift_row, shift_col)


def test_quad_region_restricts_stats_and_reports_alongside_full_frame(fixture: SyntheticFixture) -> None:
    """`--quad` restricts stats to a region while the full-frame report stays unchanged."""
    quad_unit = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    report = _report(fixture, fixture.misaligned_composite, quad_unit=quad_unit)

    assert report.quad is not None
    assert report.region is not None
    assert 0 < report.region.n_pixels < SCENE_WIDTH * SCENE_HEIGHT

    # Full-frame numbers are unaffected by the region restriction.
    full_frame_only = _report(fixture, fixture.misaligned_composite)
    assert report.local_ghosting.p95_shift_px == full_frame_only.local_ghosting.p95_shift_px

    # The region's own grid is a strict subset of the full frame's.
    assert report.region.local_ghosting.n_patches <= report.local_ghosting.n_patches


def _make_starfield_scene(
    rng: np.random.Generator, n_dots: int = 40
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Build a small, dark, mildly textured "sky" scene sprinkled with tiny bright dots.

    Purpose-built for the bright-detail-preservation metric rather than reusing `_make_scene`
    (whose loud grid of colored rectangles/circles would itself trip the top-hat speck
    detector everywhere) -- mirrors a real starry-artwork capture at a much smaller scale:
    small, isolated bright specks against a locally darker, low-contrast background.

    Args:
        rng: Numpy random generator.
        n_dots: Number of small bright dots to sprinkle in.

    Returns:
        Tuple of (background-only BGR image, BGR image with dots painted in, list of
        (x, y) dot centers).
    """
    height, width = 300, 400
    base = rng.integers(20, 40, size=(height, width, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(base, (0, 0), sigmaX=2.0)
    scene = background.copy()
    margin = 20
    centers = [
        (int(rng.integers(margin, width - margin)), int(rng.integers(margin, height - margin))) for _ in range(n_dots)
    ]
    for x, y in centers:
        cv2.circle(scene, (x, y), 2, (235, 235, 235), -1, lineType=cv2.LINE_8)
    return background, scene, centers


def test_bright_detail_preservation_and_glare_healing_speck_exclusion() -> None:
    """Bright-detail retention must separate "dots kept" from "dots erased".

    Mirrors the real-dump finding that motivated this metric: a min-composite can delete a
    small bright detail (e.g. a star on a dark background) wholesale when a 1-3px
    misregistration lets a neighboring frame's darker background pixel win the
    darkest-pixel-wins comparison at that exact location. A composite that keeps the dots
    should retain ~all of them; one that erases them (painted over with the local background
    color, simulating darkest-pixel-wins erasure) should retain far fewer -- and the erasure
    must not itself register as glare healing "benefit" once the reference specks are
    subtracted from the darkening map (see `score_stitch._darkening_map_excluding_specks`).
    """
    rng = np.random.default_rng(202)
    background, reference, centers = _make_starfield_scene(rng)

    composite_preserved = reference.copy()

    composite_erased = reference.copy()
    for x, y in centers:
        fill = tuple(int(v) for v in background[y, x])
        cv2.circle(composite_erased, (x, y), 2, fill, -1, lineType=cv2.LINE_8)

    def _dump(composite: np.ndarray) -> CaptureDump:
        return CaptureDump(
            dump_dir=Path("<synthetic>"), reference=reference, composite=composite, corners={}, metadata=None, scale=1.0
        )

    report_preserved, _ = score_stitch.score_dump(_dump(composite_preserved))
    report_erased, _ = score_stitch.score_dump(_dump(composite_erased))

    # Sanity: the fixture actually put detectable specks in the reference.
    assert report_preserved.bright_detail.reference_speck_count >= n_dots_detected_floor(len(centers))

    assert report_preserved.bright_detail.retention_ratio == pytest.approx(1.0, abs=0.05)
    assert report_erased.bright_detail.retention_ratio < 0.2

    # Sanity: without speck exclusion, the erased dots WOULD read as glare healing -- this is
    # what makes the assertion below meaningful rather than a no-op.
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_erased = cv2.cvtColor(composite_erased, cv2.COLOR_BGR2GRAY)
    raw_darkening_map = compute_darkening_map(gray_erased, gray_reference)
    raw_healing = score_stitch.compute_glare_healing(raw_darkening_map)
    assert raw_healing.darkened_fraction > 0.0

    # With speck exclusion (the actual `score_dump` pipeline), erasing the dots must not
    # increase the reported darkened fraction versus the preserved (identical-to-reference,
    # ~0-darkening) composite.
    assert report_erased.glare_healing.darkened_fraction <= report_preserved.glare_healing.darkened_fraction + 1e-6


def n_dots_detected_floor(n_dots: int) -> int:
    """A conservative lower bound on detected specks for a fixture with `n_dots` painted in.

    Args:
        n_dots: Number of dots painted into the fixture.

    Returns:
        A floor well below `n_dots` to tolerate occasional overlapping dots merging into one
        connected component, without making the sanity check toothless.
    """
    return round(n_dots * 0.7)


def test_score_stitch_handles_missing_metadata(fixture: SyntheticFixture, tmp_path: Path) -> None:
    """`load_dump` + `score_dump` must work on 6 loose images with no metadata.json."""
    dump_dir = tmp_path / "loose_dump"
    fixture.write_dump(dump_dir, fixture.aligned_composite, with_metadata=False)

    from common import load_dump

    dump = load_dump(dump_dir, long_side=480)
    assert isinstance(dump, CaptureDump)
    assert dump.metadata is None
    assert len(dump.corners) == 4

    report, patches = score_stitch.score_dump(dump)
    assert report.metadata is None
    assert patches  # non-empty patch grid


def test_score_stitch_cli_writes_metrics_json(
    fixture: SyntheticFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI writes metrics.json and the diagnostic images to --out."""
    dump_dir = tmp_path / "dump"
    fixture.write_dump(dump_dir, fixture.misaligned_composite, with_metadata=True)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", ["score_stitch.py", str(dump_dir), "--out", str(out_dir)])
    score_stitch.main()

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "ghost_heatmap.png").exists()
    assert (out_dir / "absdiff_heatmap.png").exists()

    import json

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["metadata"]["alignedFrameCount"] == 4
    assert metrics["corner_frame_count"] == 4


def test_stitch_reduces_glare_versus_raw_reference(fixture: SyntheticFixture, tmp_path: Path) -> None:
    """`stitch.py`'s own SIFT-based registration (no ground-truth homographies) reduces glare."""
    from common import load_dump

    dump_dir = tmp_path / "raw_dump"
    fixture.write_dump(dump_dir, fixture.aligned_composite, with_metadata=True)  # composite.jpg unused by stitch()
    dump = load_dump(dump_dir, long_side=480)

    composite, frame_reports = stitch.stitch(dump)

    assert len(frame_reports) == 4
    assert sum(1 for r in frame_reports if r.status in ("verified", "unverified")) >= 3

    gray_composite = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(dump.reference, cv2.COLOR_BGR2GRAY)
    assert near_saturated_fraction(gray_composite) < near_saturated_fraction(gray_reference)


def test_stitch_cli_writes_output_image(
    fixture: SyntheticFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `stitch.py` CLI writes a valid composite JPEG to --out."""
    dump_dir = tmp_path / "cli_dump"
    fixture.write_dump(dump_dir, fixture.aligned_composite, with_metadata=True)
    out_path = tmp_path / "restitched.jpg"

    monkeypatch.setattr(sys, "argv", ["stitch.py", str(dump_dir), "--out", str(out_path), "--skip-unverified"])
    stitch.main()

    assert out_path.exists()
    written = cv2.imread(str(out_path))
    assert written is not None
    # main() loads the dump at the full WORKING_LONG_SIDE (2048), not the fixture's native
    # 640x480 -- just check the aspect ratio round-trips rather than an exact pixel size.
    fixture_aspect = fixture.reference.shape[1] / fixture.reference.shape[0]
    written_aspect = written.shape[1] / written.shape[0]
    assert written_aspect == pytest.approx(fixture_aspect, rel=0.01)
