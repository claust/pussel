"""Service module for processing puzzle pieces and matching them to puzzles.

Uses FastBackboneModel that compares piece features with puzzle features
to predict position (3x3 grid) and rotation.
"""

import base64
import io
import os
from pathlib import Path
from typing import Optional, cast

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from fastapi import UploadFile
from PIL import Image
from torch import Tensor

from app.config import settings
from app.models.model import FastBackboneModel
from app.models.puzzle_model import PieceResponse, Position
from app.services.background_remover import get_background_remover

# Path to checkpoint (3x3 grid model with 82% cell accuracy, 95% rotation accuracy)
DEFAULT_CHECKPOINT_PATH = str(
    Path(__file__).resolve().parents[3]
    / "network"
    / "experiments"
    / "exp18_3x3_20k_puzzles"
    / "outputs"
    / "checkpoint_best.pt"
)
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)

# Image sizes expected by the model
PIECE_SIZE = 128
PUZZLE_SIZE = 256


class ImageProcessor:
    """Image processing service for puzzle piece detection."""

    def __init__(self) -> None:
        """Initialize the image processor with the trained model."""
        self.device = self._get_device()
        self.model = self._load_model()

        # Separate transforms for piece and puzzle (no ImageNet normalization!)
        self.piece_transform = transforms.Compose(
            [
                transforms.Resize((PIECE_SIZE, PIECE_SIZE)),
                transforms.ToTensor(),
            ]
        )
        self.puzzle_transform = transforms.Compose(
            [
                transforms.Resize((PUZZLE_SIZE, PUZZLE_SIZE)),
                transforms.ToTensor(),
            ]
        )

        # Cache for puzzle tensors to avoid reloading
        self._puzzle_cache: dict[str, Tensor] = {}

    def _get_device(self) -> torch.device:
        """Get the appropriate device (CPU, CUDA, or MPS)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> FastBackboneModel:
        """Load the trained model from checkpoint.

        Returns:
            The loaded model in evaluation mode.
        """
        # Initialize model with same config as training
        model = FastBackboneModel(
            pretrained=False,  # We'll load weights from checkpoint
            correlation_dim=128,
            rotation_hidden_dim=128,
            freeze_backbone=False,
            dropout=0.1,
            rotation_dropout=0.2,
        )

        # Load checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(self.device)
        model.eval()
        return model

    def _load_puzzle_tensor(self, puzzle_id: str) -> Tensor:
        """Load and cache puzzle image tensor.

        Args:
            puzzle_id: The ID of the puzzle to load.

        Returns:
            The puzzle image as a tensor.

        Raises:
            FileNotFoundError: If the puzzle image doesn't exist.
        """
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = os.path.join(settings.UPLOAD_DIR, f"{puzzle_id}.jpg")
            # Normalize and ensure the puzzle path stays within the upload directory
            base_dir = os.path.realpath(settings.UPLOAD_DIR)
            normalized_path = os.path.realpath(puzzle_path)
            # Ensure the resolved puzzle path is contained within the upload directory
            if os.path.commonpath([base_dir, normalized_path]) != base_dir:
                raise FileNotFoundError(f"Puzzle image not found: {normalized_path}")
            if not os.path.exists(normalized_path):
                raise FileNotFoundError(f"Puzzle image not found: {normalized_path}")

            puzzle_img = Image.open(normalized_path).convert("RGB")
            puzzle_tensor = cast(Tensor, self.puzzle_transform(puzzle_img))
            self._puzzle_cache[puzzle_id] = puzzle_tensor

        return self._puzzle_cache[puzzle_id]

    def clear_puzzle_cache(self, puzzle_id: Optional[str] = None) -> None:
        """Clear cached puzzle tensors.

        Args:
            puzzle_id: If provided, only clear this puzzle. Otherwise clear all.
        """
        if puzzle_id is not None:
            self._puzzle_cache.pop(puzzle_id, None)
        else:
            self._puzzle_cache.clear()

    async def process_piece(
        self,
        piece_file: UploadFile,
        puzzle_id: str,
        remove_background: bool = True,
    ) -> PieceResponse:
        """Process a puzzle piece image and predict its position and rotation.

        Args:
            piece_file: The puzzle piece image file.
            puzzle_id: The ID of the puzzle to match against.
            remove_background: Whether to remove background from piece image.

        Returns:
            PieceResponse with predicted position, confidence, rotation, and optionally cleaned image.
        """
        try:
            # Read piece image bytes
            contents = await piece_file.read()

            # Optionally remove background
            cleaned_image_b64: Optional[str] = None
            if remove_background and settings.ENABLE_BACKGROUND_REMOVAL:
                remover = get_background_remover(settings.REMBG_MODEL)

                # Get RGBA image with transparent background for frontend display
                rgba_image = remover.remove_background(contents)

                # Encode as base64 PNG for frontend
                buffer = io.BytesIO()
                rgba_image.save(buffer, format="PNG")
                cleaned_image_b64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

                # Convert to RGB with white background for model inference
                piece_img = Image.new("RGB", rgba_image.size, (255, 255, 255))
                if rgba_image.mode == "RGBA":
                    piece_img.paste(rgba_image, mask=rgba_image.split()[3])
                else:
                    piece_img.paste(rgba_image)
            else:
                piece_img = Image.open(io.BytesIO(contents)).convert("RGB")

            piece_tensor = cast(Tensor, self.piece_transform(piece_img)).unsqueeze(0)
            piece_tensor = piece_tensor.to(self.device)

            # Load puzzle tensor
            puzzle_tensor = self._load_puzzle_tensor(puzzle_id).unsqueeze(0)
            puzzle_tensor = puzzle_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                position, rotation_logits, attention_map = self.model(piece_tensor, puzzle_tensor)

            # Position is already (cx, cy) in [0, 1] - use directly
            x_center = float(position[0, 0].item())
            y_center = float(position[0, 1].item())

            # Get position confidence from attention map (max attention value)
            position_confidence = float(attention_map.max().item())

            # Get rotation class and confidence
            rotation_probs = F.softmax(rotation_logits, dim=1)
            rotation_class = int(rotation_logits.argmax(dim=1).item())
            rotation_degrees = rotation_class * 90  # 0, 90, 180, 270
            rotation_confidence = float(rotation_probs[0, rotation_class].item())

            return PieceResponse(
                position=Position(x=x_center, y=y_center),
                position_confidence=position_confidence,
                rotation=rotation_degrees,
                rotation_confidence=rotation_confidence,
                cleaned_image=cleaned_image_b64,
            )

        except FileNotFoundError:
            # Re-raise file not found errors
            raise
        except Exception as e:
            # Log error and return fallback response
            print(f"Error processing image: {str(e)}")
            return PieceResponse(
                position=Position(x=0.5, y=0.5),
                position_confidence=0.0,
                rotation=0,
                rotation_confidence=0.0,
                cleaned_image=None,
            )


# Singleton instance
_processor: Optional[ImageProcessor] = None


def get_image_processor() -> ImageProcessor:
    """Get the singleton ImageProcessor instance.

    Returns:
        The shared ImageProcessor instance.
    """
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor
