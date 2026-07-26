"""Experiment 30: fix the two label-leaking generator bugs, retrain exp26.

The 2026-07-26 realism investigation (``docs/synthetic-dataset-realism.html``)
found that the sim-to-real collapse is **shortcut learning**, not a realism
deficit, and named two outright label leaks in the synthetic piece pipeline:

- **4.1 border-touch rotation shortcut** — pieces are cropped tight to the
  silhouette and rotated with ``expand=False``, so 0deg/180deg fill the
  canvas while 90deg/270deg are clipped with black bands. "Alpha touches
  all four borders => rotation in {0, 180}" is 100%-precise in synthetic
  train *and* test, and never fires on real crops.
- **4.2 half-pixel blur cue** — bilinear ``expand=False`` rotation on a
  non-square canvas makes 90deg/270deg ~32% blurrier than the pixel-exact
  0deg/180deg, so sharpness alone leaks the rotation label.

Plus a framing mismatch: training squashed the tight crop to 128x128 while
deployment pads to square with an 8% margin first.

exp30 is the **exp26 recipe unchanged** with those three things fixed — the
base rotation and the load-time rotation both go through
``Image.transpose`` (exact, no clipping, no blur), and every piece is
framed with exp24's real-path geometry before augmentation.

Prediction: north_star rotation accuracy jumps well above 33.5% and the
90deg/270deg prediction bias disappears.
"""

from .framed_dataset import (
    FramedAugmentedPieceDataset,
    FramedBlackCompositeTestDataset,
    create_datasets_from_split,
    piece_to_model_input,
)
from .framing import alpha_bbox, frame_and_black_composite, frame_rgba, rotate_lossless

__all__ = [
    "FramedAugmentedPieceDataset",
    "FramedBlackCompositeTestDataset",
    "alpha_bbox",
    "create_datasets_from_split",
    "frame_and_black_composite",
    "frame_rgba",
    "piece_to_model_input",
    "rotate_lossless",
]
