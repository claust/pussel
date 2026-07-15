"""Experiment 26: Domain randomization to survive real photos.

exp25 showed the exp20 CNN collapses on the north-star real-photo
benchmark (72.2% -> 14.8% both-correct) because it learned pixel-identical
template lookup, while a zero-training SIFT->NCC hybrid holds 76.7%. exp26
retrains the same architecture with realism augmentations so the pixel
shortcut no longer works:

- independent photometric jitter on piece and puzzle (the key lever),
- random scale, mild perspective and sub-90-degree rotation jitter,
- realistic backgrounds composited behind RGBA pieces (with a
  black-composite fraction matching the deployed rembg output),
- sensor noise and JPEG artifacts.

Every augmentation is individually toggleable (see ``augment.AugmentConfig``
and the ``--aug-preset`` ablation entry point). The exp20 frozen split,
harness and model are reused unchanged; val selects the checkpoint and the
synthetic test set is touched once. Target: beat 76.7% both-correct on
north_star v1.
"""

from .aug_dataset import AugmentedPieceDataset, BlackCompositeTestDataset, create_datasets_from_split
from .augment import AUG_PRESETS, AugmentConfig, augment_piece, augment_puzzle, black_composite

__all__ = [
    "AugmentConfig",
    "AUG_PRESETS",
    "augment_piece",
    "augment_puzzle",
    "black_composite",
    "AugmentedPieceDataset",
    "BlackCompositeTestDataset",
    "create_datasets_from_split",
]
