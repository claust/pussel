"""Experiment 31: break piece<->overview pixel identity structurally.

Test 2 of ``docs/synthetic-dataset-realism.html`` section 7. exp26 tried to
break the pixel-identity shortcut with photometric and geometric jitter and
did not move the correlation (masked NCC at the ground-truth location: raw
synthetic 0.990, exp26-augmented 0.937, real 0.730) — photometric jitter is a
near-affine map any contrast-normalised matcher cancels, and geometric jitter
is absorbed by a scale/rotation search. exp30 then removed both label-leaking
generator bugs and real transfer still did not move (13.2% both, against
exp26's 12.7% and the classical SIFT->NCC bar of 76.7%).

That leaves section 4.3: the synthetic "overview" is the byte-identical source
JPEG the pieces were cut from — shared demosaic, white balance, JPEG block
grid, noise fingerprint and sharpening. In production the box photo and the
piece photo are **two independent captures of two different physical prints**.

exp31 models exactly that, on top of exp30's validated geometry, with six
components that are each drawn independently per sample *and* per view:

1. independent per-view degradation chains (resample kernel + subpixel phase,
   blur/sharpen, sensor noise, a two-octave substrate/illumination field, a
   non-linear tone curve, and JPEG on an independent block grid);
2. Asymmetric Random Patching into exactly one view (Chuah et al.,
   2106.08486: synthetic->real stereo error 28.0% -> 4.0%);
3. a segmentation-slop model fitted to real rembg output;
4. a bright die-cut cardboard rim with a cast shadow (real rim ratio 1.08,
   synthetic 0.98, exp26's halo 0.69 — the wrong direction);
5. box-photo overview realism (residual perspective, specular glare, lighting
   gradient, vignetting, and the overview's own sensor noise, which
   ``augment_puzzle()`` left at 0.07 against a real 0.84), plus the same room
   light falling independently on the piece's close-up;
6. explicit crop/bbox jitter.

Everything else — model, optimizer, frozen exp20 split, harness,
checkpoint-selection discipline, checkpoint artifact names — is exp26/exp30
unchanged, so the comparison is clean.

Gate before retraining: the section 8 NCC-headroom probe (``ncc_probe.py``).
It **passes**: masked NCC at the ground-truth cell falls from exp30's 0.937 to
0.737 (real 0.679) with the true-vs-decoy margin preserved at 0.300. See the
README for the per-component ablation and for why 0.737 is close to this
corpus's floor.
"""

from .capture import (
    CAPTURE_PRESETS,
    CaptureConfig,
    ChainDraw,
    PatchSource,
    ViewChainConfig,
    apply_arp,
    apply_box_photo,
    apply_chain_draw,
    apply_die_cut_edge,
    apply_piece_lighting,
    apply_scene_surface,
    apply_segmentation_slop,
    apply_view_chain,
    augment_overview_capture,
    augment_piece_capture,
    augment_view_pair,
    capture_config_to_dict,
    composite_with_shadow,
    draw_chain,
    frame_rgba_jittered,
    resolve_chain,
)
from .capture_dataset import (
    CapturePieceDataset,
    ViewPair,
    create_datasets_from_split,
    deterministic_seed,
    piece_to_model_input,
    view_pair,
)

__all__ = [
    "CAPTURE_PRESETS",
    "CaptureConfig",
    "CapturePieceDataset",
    "ChainDraw",
    "PatchSource",
    "ViewChainConfig",
    "ViewPair",
    "apply_arp",
    "apply_box_photo",
    "apply_chain_draw",
    "apply_die_cut_edge",
    "apply_piece_lighting",
    "apply_scene_surface",
    "apply_segmentation_slop",
    "apply_view_chain",
    "augment_overview_capture",
    "augment_piece_capture",
    "augment_view_pair",
    "capture_config_to_dict",
    "composite_with_shadow",
    "create_datasets_from_split",
    "deterministic_seed",
    "draw_chain",
    "frame_rgba_jittered",
    "piece_to_model_input",
    "resolve_chain",
    "view_pair",
]
