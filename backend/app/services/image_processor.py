"""Service module for processing puzzle pieces and matching them to puzzles.

Uses FastBackboneModel that compares piece features with puzzle features
to predict position (4x4 grid) and rotation.
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional, cast
from uuid import UUID

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
from app.services.piece_detector import crop_to_alpha_region, harden_alpha

# Path to checkpoint (exp20 4x4 grid model, realistic pieces, ~72.9% cell accuracy)
DEFAULT_CHECKPOINT_PATH = str(
    Path(__file__).resolve().parents[3]
    / "network"
    / "experiments"
    / "exp20_realistic_pieces"
    / "outputs"
    / "checkpoint_best.pt"
)
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)

# Opt-in escape hatch for legacy checkpoints that pickle non-tensor objects and
# therefore cannot be loaded with ``weights_only=True``. Off by default because
# a full pickle load executes arbitrary code, and ``MODEL_CHECKPOINT_PATH`` is
# operator-configurable. Set ``ALLOW_UNSAFE_CHECKPOINT_LOAD=1`` only for a
# trusted checkpoint.
ALLOW_UNSAFE_CHECKPOINT_LOAD = os.environ.get("ALLOW_UNSAFE_CHECKPOINT_LOAD", "").lower() in ("1", "true", "yes")

# Image sizes expected by the model
PIECE_SIZE = 128
PUZZLE_SIZE = 256


def _composite_on_black(image: Image.Image) -> Image.Image:
    """Composite an RGBA cutout onto a black background.

    The exp20 training pieces (and the exp25 north-star eval that validated
    this pipeline) put segmented pieces on black, so model inputs must match.

    Args:
        image: The piece image, typically RGBA from background removal.

    Returns:
        An RGB image with transparent regions rendered black.
    """
    rgb = Image.new("RGB", image.size, (0, 0, 0))
    if image.mode == "RGBA":
        rgb.paste(image, mask=image.split()[3])
    else:
        rgb.paste(image)
    return rgb


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an RGB image to a centered square on black.

    The model resizes inputs to a fixed square, so non-square crops must be
    padded (as in training and the exp25 eval) rather than squashed, which
    would distort the piece's aspect ratio.

    Args:
        image: The RGB image to pad.

    Returns:
        A square RGB image with the input centered on a black canvas, or the
        input unchanged when it is already square.
    """
    side = max(image.width, image.height)
    if image.width == side and image.height == side:
        return image
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def _extract_state_dict(checkpoint: Any) -> Any:
    """Return the model weights from a loaded checkpoint.

    Supports both a raw ``state_dict`` (exp20's ``checkpoint_best.pt``) and a
    ``{"model_state_dict": ...}`` wrapper (exp18-style checkpoints).

    Args:
        checkpoint: The object returned by ``torch.load``.

    Returns:
        The model ``state_dict`` mapping.
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _load_checkpoint(path: str, device: torch.device) -> Any:
    """Load a checkpoint file using the safe ``weights_only`` path.

    Modern checkpoints (including exp20's raw tensor ``state_dict``) load with
    ``weights_only=True``, which avoids arbitrary pickle execution. Legacy
    checkpoints that pickle non-tensor objects can only be read with a full
    (unsafe) load; that path is gated behind ``ALLOW_UNSAFE_CHECKPOINT_LOAD`` so
    it is never taken implicitly. When the safe load fails and the escape hatch
    is off, the error propagates and the caller degrades to neutral predictions.

    Args:
        path: Filesystem path to the checkpoint.
        device: Device to map tensors onto.

    Returns:
        The loaded checkpoint object.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001 - decide whether the unsafe fallback is permitted
        if not ALLOW_UNSAFE_CHECKPOINT_LOAD:
            raise
        print(f"weights_only load failed for {path}; retrying with full pickle load (ALLOW_UNSAFE_CHECKPOINT_LOAD set)")
        return torch.load(path, map_location=device, weights_only=False)


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

    def _load_model(self) -> Optional[FastBackboneModel]:
        """Load the trained model from checkpoint.

        Returns:
            The loaded model in evaluation mode, or None when no checkpoint is
            available or fails to load. Without a model, ``process_piece``
            returns a neutral fallback prediction so the app still runs (e.g. in
            CI, where the checkpoint is not committed).
        """
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Model checkpoint not found at {CHECKPOINT_PATH}; predictions will use a neutral fallback")
            return None

        # Initialize model with same config as exp20 training
        model = FastBackboneModel(
            backbone_name="shufflenet_v2_x0_5",
            pretrained=False,  # We'll load weights from checkpoint
            correlation_dim=128,
            rotation_hidden_dim=128,
            freeze_backbone=False,
            dropout=0.1,
            rotation_dropout=0.2,
        )

        # Load onto CPU first, then move the whole model to the target device in
        # one transfer. A corrupt or incompatible checkpoint falls back to the
        # neutral prediction path rather than crashing ImageProcessor init.
        try:
            checkpoint = _load_checkpoint(CHECKPOINT_PATH, torch.device("cpu"))
            model.load_state_dict(_extract_state_dict(checkpoint))
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any load failure
            print(
                f"Failed to load model checkpoint at {CHECKPOINT_PATH} ({exc}); predictions will use a neutral fallback"
            )
            return None

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
            try:
                puzzle_uuid = UUID(puzzle_id)
                if str(puzzle_uuid) != puzzle_id:
                    raise ValueError
            except ValueError as exc:
                raise FileNotFoundError(f"Puzzle image not found: {puzzle_id}") from exc

            # Build the filename from the parsed UUID rather than the raw request
            # string, so the path component can only ever be a canonical UUID.
            puzzle_path = os.path.join(settings.UPLOAD_DIR, f"{puzzle_uuid}.jpg")
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

                # Drop the matte's faint background ghost before cropping so
                # neither the client nor the crop sees it.
                rgba_image = harden_alpha(rgba_image)

                # Crop to the segmented subject so the piece fills the frame for the
                # model (training pieces fill the image) instead of floating in a
                # large mostly-white canvas
                rgba_image = crop_to_alpha_region(rgba_image)

                # Encode as base64 PNG for frontend
                buffer = io.BytesIO()
                rgba_image.save(buffer, format="PNG")
                cleaned_image_b64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

                # Model input: composite on black and pad to square, matching
                # the exp20 training pieces and the exp25 north-star eval prep
                piece_img = _pad_to_square(_composite_on_black(rgba_image))
            else:
                piece_img = _pad_to_square(Image.open(io.BytesIO(contents)).convert("RGB"))

            # No model available (e.g. checkpoint not present): return a neutral
            # prediction but still hand back the cleaned cutout for display.
            if self.model is None:
                return PieceResponse(
                    position=Position(x=0.5, y=0.5),
                    position_confidence=0.0,
                    rotation=0,
                    rotation_confidence=0.0,
                    cleaned_image=cleaned_image_b64,
                )

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
