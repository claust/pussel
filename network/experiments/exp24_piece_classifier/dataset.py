"""Dataset and splits for the binary piece classifier.

Samples are pre-built square crops on black backgrounds (see
``build_positives.py`` / ``build_negatives.py``):

- ``positives/synthetic/<puzzle>__<piece>.png`` — exp20 generator pieces
- ``positives/real/puzzleNN__IMG_xxxx.png`` — rembg crops of real photos
- ``negatives/<source>/<category>/<name>.png`` — rembg crops of faces and
  household objects

Splits are made at group level to avoid near-duplicate leakage: positives
group by puzzle, negatives by small chunks of consecutive files within a
category (Caltech-101 orders images of the same subject consecutively).
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_DATA_ROOT = Path(__file__).parent.parent.parent / "datasets" / "piece_classifier"

INPUT_SIZE = 128

# Chunk size for grouping consecutive negative files (leakage control)
NEGATIVE_CHUNK = 10


@dataclass(frozen=True)
class Sample:
    """A single labeled crop on disk."""

    path: Path
    label: int  # 1 = puzzle piece, 0 = not a piece
    source: str  # e.g. "synthetic", "real", "caltech101", "coco128"
    category: str  # e.g. "synthetic", "real", "Faces", "cup", "scenes"
    group: str  # leakage-control group used for splitting


def collect_samples(data_root: Path = DEFAULT_DATA_ROOT) -> list[Sample]:
    """Scan the dataset root and return all labeled samples.

    Args:
        data_root: Dataset root with positives/ and negatives/ trees.

    Returns:
        All samples found, sorted by path.

    Raises:
        FileNotFoundError: When the dataset root has no samples.
    """
    samples: list[Sample] = []

    for source in ("synthetic", "real"):
        source_dir = data_root / "positives" / source
        for path in sorted(source_dir.glob("*.png")):
            group = f"{source}:{path.stem.split('__')[0]}"
            samples.append(Sample(path=path, label=1, source=source, category=source, group=group))

    negatives_root = data_root / "negatives"
    if negatives_root.exists():
        for source_dir in sorted(p for p in negatives_root.iterdir() if p.is_dir()):
            for category_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
                files = sorted(category_dir.glob("*.png"))
                for i, path in enumerate(files):
                    group = f"{source_dir.name}:{category_dir.name}:{i // NEGATIVE_CHUNK}"
                    samples.append(
                        Sample(path=path, label=0, source=source_dir.name, category=category_dir.name, group=group)
                    )

    if not samples:
        raise FileNotFoundError(f"No samples under {data_root}; run the build_* scripts first")
    return samples


def make_splits(
    samples: list[Sample],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Split samples into train/val/test at group level, stratified by source.

    Args:
        samples: All samples.
        val_fraction: Fraction of each source's groups assigned to val.
        test_fraction: Fraction of each source's groups assigned to test.
        seed: Shuffle seed.

    Returns:
        Dict with "train", "val" and "test" sample lists.
    """
    rng = random.Random(seed)
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    sources = sorted({s.source for s in samples})
    for source in sources:
        source_samples = [s for s in samples if s.source == source]
        groups = sorted({s.group for s in source_samples})
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * test_fraction))
        n_val = max(1, round(len(groups) * val_fraction))
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        for sample in source_samples:
            if sample.group in test_groups:
                splits["test"].append(sample)
            elif sample.group in val_groups:
                splits["val"].append(sample)
            else:
                splits["train"].append(sample)
    return splits


class RandomDownscale:
    """Randomly downscale then restore, simulating low-resolution preview crops."""

    def __init__(self, min_size: int = 48, max_size: int = 112, probability: float = 0.5):
        """Initialize the augmentation.

        Args:
            min_size: Smallest intermediate size.
            max_size: Largest intermediate size.
            probability: Chance of applying the downscale.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply the augmentation to one image.

        Args:
            image: Input image.

        Returns:
            Possibly downscaled-and-restored image.
        """
        if random.random() > self.probability:
            return image
        size = random.randint(self.min_size, self.max_size)
        small = image.resize((size, size), Image.Resampling.BILINEAR)
        return small.resize((image.width, image.height), Image.Resampling.BILINEAR)


def train_transform() -> Callable[[Image.Image], torch.Tensor]:
    """Build the training augmentation pipeline.

    Pieces appear at any angle in a live frame, so rotation covers the full
    circle; color jitter covers lighting differences; the downscale step
    covers the low-resolution crops the 320px preview pipeline produces.

    Returns:
        A transform mapping a PIL image to a [3, 128, 128] tensor.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, expand=True, fill=0),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.8),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            RandomDownscale(),
            transforms.ToTensor(),
        ]
    )


def eval_transform() -> Callable[[Image.Image], torch.Tensor]:
    """Build the deterministic evaluation pipeline.

    Returns:
        A transform mapping a PIL image to a [3, 128, 128] tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    )


class PieceClassifierDataset(Dataset):
    """Torch dataset over labeled classifier crops."""

    def __init__(self, samples: list[Sample], train: bool = False):
        """Initialize the dataset.

        Args:
            samples: The samples to serve.
            train: Whether to apply training augmentation.
        """
        self.samples = samples
        self.transform = train_transform() if train else eval_transform()

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image tensor [3, 128, 128], label tensor scalar float).
        """
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), torch.tensor(float(sample.label))


def summarize(samples: list[Sample], name: Optional[str] = None) -> str:
    """Produce a one-line per-source count summary of a sample list.

    Args:
        samples: The samples to summarize.
        name: Optional prefix (e.g. split name).

    Returns:
        Human-readable summary line.
    """
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.source] = counts.get(sample.source, 0) + 1
    parts = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
    prefix = f"{name}: " if name else ""
    n_pos = sum(1 for s in samples if s.label == 1)
    return f"{prefix}{len(samples)} samples ({n_pos} pos / {len(samples) - n_pos} neg; {parts})"
