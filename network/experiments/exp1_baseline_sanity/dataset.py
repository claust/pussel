"""On-the-fly square image generator for baseline sanity check.

Generates 64x64 RGB images with:
- Solid gray background (random shade 100-180)
- One colored square, fixed size 16x16, random color
- Square placed randomly (fully within bounds)
- Target: normalized center coordinates (cx, cy) in [0, 1]
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SquareDataset(Dataset):
    """Generates random images with a single colored square.

    Returns (image, target) where target is normalized (cx, cy).
    """

    def __init__(
        self,
        size: int = 1000,
        image_size: int = 64,
        square_size: int = 16,
        seed: int | None = None,
    ):
        """Initialize the dataset.

        Args:
            size: Number of samples in the dataset.
            image_size: Size of the square image (64x64).
            square_size: Size of the colored square (16x16).
            seed: Random seed for reproducibility (None for random each time).
        """
        self.size = size
        self.image_size = image_size
        self.square_size = square_size
        self.seed = seed

        # Pre-generate all samples for consistency during training
        if seed is not None:
            np.random.seed(seed)

        self.samples = [self._generate_sample() for _ in range(size)]

    def _generate_sample(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Generate a single sample with a colored square on gray background."""
        # Random gray background (100-180)
        bg_gray = np.random.randint(100, 181)
        image = np.full((self.image_size, self.image_size, 3), bg_gray, dtype=np.uint8)

        # Random position for the square (fully within bounds)
        max_pos = self.image_size - self.square_size
        x = np.random.randint(0, max_pos + 1)
        y = np.random.randint(0, max_pos + 1)

        # Random color for the square (make it distinct from background)
        # Either very dark (<50) or very bright (>200) to ensure contrast
        color = np.random.randint(0, 256, size=3)
        # Ensure at least one channel is far from the background gray
        while np.all(np.abs(color.astype(int) - bg_gray) < 50):
            color = np.random.randint(0, 256, size=3)

        # Draw the square
        image[y : y + self.square_size, x : x + self.square_size] = color

        # Compute normalized center coordinates
        cx = (x + self.square_size / 2) / self.image_size
        cy = (y + self.square_size / 2) / self.image_size

        return image, (cx, cy)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the image and target tensors for the given index."""
        image, (cx, cy) = self.samples[idx]

        # Convert to tensor: (H, W, C) -> (C, H, W), normalize to [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Target as tensor
        target_tensor = torch.tensor([cx, cy], dtype=torch.float32)

        return image_tensor, target_tensor


class InfiniteSquareDataset(Dataset):
    """Generates samples on-the-fly without pre-storing.

    Useful for training with fresh random samples each epoch.
    """

    def __init__(self, size: int = 1000, image_size: int = 64, square_size: int = 16):
        """Initialize the dataset with given parameters."""
        self.size = size
        self.image_size = image_size
        self.square_size = square_size

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate and return a random sample on-the-fly."""
        # Random gray background (100-180)
        bg_gray = np.random.randint(100, 181)
        image = np.full((self.image_size, self.image_size, 3), bg_gray, dtype=np.uint8)

        # Random position for the square (fully within bounds)
        max_pos = self.image_size - self.square_size
        x = np.random.randint(0, max_pos + 1)
        y = np.random.randint(0, max_pos + 1)

        # Random color for the square
        color = np.random.randint(0, 256, size=3)
        while np.all(np.abs(color.astype(int) - bg_gray) < 50):
            color = np.random.randint(0, 256, size=3)

        # Draw the square
        image[y : y + self.square_size, x : x + self.square_size] = color

        # Compute normalized center coordinates
        cx = (x + self.square_size / 2) / self.image_size
        cy = (y + self.square_size / 2) / self.image_size

        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.tensor([cx, cy], dtype=torch.float32)

        return image_tensor, target_tensor


if __name__ == "__main__":
    # Quick test
    dataset = SquareDataset(size=5, seed=42)
    for i in range(len(dataset)):
        img, target = dataset[i]
        print(f"Sample {i}: image shape = {img.shape}, target = {target}")
