"""Service running the binary piece classifier on rembg-segmented crops.

Loads the exp24 checkpoint the same way ``image_processor`` loads the
position model. When the checkpoint is absent (e.g. in CI, where model
checkpoints are not committed) the service reports itself unavailable and
the piece detector falls back to its heuristic confidence.

The input protocol mirrors ``network/experiments/exp24_piece_classifier/
data_prep.py`` exactly: largest opaque component of the rembg RGBA output,
composited on black, cropped to its bounding box with a small margin,
padded to a square and resized to 128x128. Change both places together.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from app.models.piece_classifier_model import INPUT_SIZE, PieceClassifier

DEFAULT_CHECKPOINT_PATH = str(
    Path(__file__).resolve().parents[3]
    / "network"
    / "experiments"
    / "exp24_piece_classifier"
    / "outputs"
    / "checkpoint_best.pt"
)
CHECKPOINT_PATH = os.environ.get("PIECE_CLASSIFIER_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)

# Margin around the subject bounding box, as a fraction of its size
CROP_MARGIN = 0.08

# Alpha value above which a pixel counts as opaque
ALPHA_THRESHOLD = 128


def prepare_classifier_input(rgba: Image.Image, margin: float = CROP_MARGIN) -> Optional[Image.Image]:
    """Convert a rembg RGBA output into the classifier's square black-backed crop.

    Args:
        rgba: RGBA image where alpha marks the segmented subject.
        margin: Extra margin around the subject bounding box.

    Returns:
        A square RGB image with the subject composited on black, or None when
        the image has no opaque region.
    """
    if rgba.mode != "RGBA":
        return None
    alpha_channel = rgba.split()[3]
    alpha = np.asarray(alpha_channel)
    mask = (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    black = Image.new("RGB", rgba.size, (0, 0, 0))
    black.paste(rgba, mask=alpha_channel)

    pad_x = round(w * margin)
    pad_y = round(h * margin)
    crop = black.crop(
        (max(0, x - pad_x), max(0, y - pad_y), min(black.width, x + w + pad_x), min(black.height, y + h + pad_y))
    )

    side = max(crop.width, crop.height)
    if crop.width != side or crop.height != side:
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        crop = canvas
    return crop


class PieceClassifierService:
    """Runs the trained piece/not-piece classifier, degrading gracefully."""

    def __init__(self) -> None:
        """Load the classifier checkpoint if one is available."""
        self.device = torch.device("cpu")  # preview frames are small; CPU keeps latency predictable
        self.model = self._load_model()

    def _load_model(self) -> Optional[PieceClassifier]:
        """Load the trained classifier from checkpoint.

        Returns:
            The model in evaluation mode, or None when the checkpoint is
            missing or unloadable (the detector then uses its heuristic).
        """
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Piece classifier checkpoint not found at {CHECKPOINT_PATH}; using heuristic confidence")
            return None
        try:
            # weights_only=True: the checkpoint is a plain state_dict container,
            # so the restricted loader suffices and avoids arbitrary pickle code.
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            model = PieceClassifier(pretrained=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            print(f"Failed to load piece classifier checkpoint: {exc}; using heuristic confidence")
            return None

    @property
    def available(self) -> bool:
        """Whether a trained classifier is loaded."""
        return self.model is not None

    def score(self, rgba: Image.Image) -> Optional[float]:
        """Compute the probability that the segmented subject is a puzzle piece.

        Args:
            rgba: The rembg RGBA output for the frame.

        Returns:
            Probability in [0, 1], or None when the classifier is unavailable
            or no usable crop could be prepared.
        """
        if self.model is None:
            return None
        crop = prepare_classifier_input(rgba)
        if crop is None:
            return None
        try:
            resized = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                probability = torch.sigmoid(self.model(tensor.to(self.device)))
            return float(probability.item())
        except Exception as exc:
            print(f"Piece classifier inference failed: {exc}")
            return None


# Singleton instance
_piece_classifier: Optional[PieceClassifierService] = None


def get_piece_classifier() -> PieceClassifierService:
    """Get or create the singleton PieceClassifierService instance.

    Returns:
        The shared PieceClassifierService instance.
    """
    global _piece_classifier
    if _piece_classifier is None:
        _piece_classifier = PieceClassifierService()
    return _piece_classifier
