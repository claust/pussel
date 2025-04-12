"""Service module for processing puzzle pieces and matching them to puzzles."""

import io
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from fastapi import UploadFile
from PIL import Image

from app.models.puzzle_model import PieceResponse, Position

# Path to checkpoint can be configured or discovered dynamically
CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    str(
        Path(__file__).resolve().parents[3] / "network" / "checkpoints" / "last-v2.ckpt"
    ),
)


class ImageProcessor:
    """Image processing service for puzzle piece detection."""

    def __init__(self):
        """Initialize the image processor with the trained model."""
        # Load the model when the class is instantiated
        self.device = self._get_device()
        self.model = self._load_model()

        # Define image transformation pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _get_device(self) -> torch.device:
        """Get the appropriate device (CPU, CUDA, or MPS)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> torch.nn.Module:
        """Load the trained model from checkpoint.

        Returns:
            The loaded model in evaluation mode.
        """
        # Load checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        model = checkpoint["hyper_parameters"]["_model_class"](
            **checkpoint["hyper_parameters"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    async def process_piece(self, piece_file: UploadFile) -> PieceResponse:
        """Process a puzzle piece image and predict its position.

        Args:
            piece_file: The puzzle piece image file.

        Returns:
            PieceResponse: Predicted position and confidence.
        """
        try:
            # Read image from UploadFile
            contents = await piece_file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(self.device)

            # Get model predictions
            with torch.no_grad():
                position_pred, rotation_class, rotation_probs = (
                    self.model.predict_piece(img_tensor)
                )

            # Convert position from (x1, y1, x2, y2) to center (x, y)
            # Assuming position_pred is in format (x1, y1, x2, y2) as per model output
            x_center = (position_pred[0].item() + position_pred[2].item()) / 2
            y_center = (position_pred[1].item() + position_pred[3].item()) / 2

            # Get rotation in degrees and confidence
            # Map class 0,1,2,3 to 0,90,180,270 degrees
            rotation_degrees = rotation_class * 90
            confidence = rotation_probs[rotation_class].item()

            return PieceResponse(
                position=Position(x=x_center, y=y_center),
                confidence=confidence,
                rotation=rotation_degrees,
            )

        except Exception as e:
            # For production, you would want to log this error
            print(f"Error processing image: {str(e)}")
            # Return a fallback response or raise an appropriate exception
            # For now, let's just return a default response
            position = Position(x=0.5, y=0.5)
            return PieceResponse(position=position, confidence=0.0, rotation=0)
