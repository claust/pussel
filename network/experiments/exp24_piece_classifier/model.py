"""Binary puzzle-piece classifier model.

A single MobileNetV3-Small backbone with a small binary head. The model is
mobile-friendly by design (~1.0M parameters, 128px input, few-ms CPU
inference) because it will eventually run on-device in the live camera
preview loop.

Input protocol (must match the backend's ``piece_classifier`` service):
the rembg-segmented subject, composited on black, cropped to its bounding
box with a small margin, padded to a square, and resized to 128x128.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

INPUT_SIZE = 128


class PieceClassifier(nn.Module):
    """MobileNetV3-Small binary classifier: puzzle piece vs. not a piece."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2, head_dim: int = 128):
        """Initialize the classifier.

        Args:
            pretrained: Whether to initialize the backbone with ImageNet weights.
            dropout: Dropout rate in the classification head.
            head_dim: Hidden dimension of the classification head.
        """
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.mobilenet_v3_small(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(576, head_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute piece-vs-not logits.

        Args:
            x: Image batch [B, 3, H, W] with values in [0, 1].

        Returns:
            Logits tensor of shape [B]; positive means "puzzle piece".
        """
        feat = self.pool(self.features(x)).flatten(1)
        return self.head(feat).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the probability that each image is a puzzle piece.

        Args:
            x: Image batch [B, 3, H, W] with values in [0, 1].

        Returns:
            Probabilities tensor of shape [B] in [0, 1].
        """
        return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module) -> int:
    """Count all parameters in a model.

    Args:
        model: The model to count.

    Returns:
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = PieceClassifier(pretrained=False)
    dummy = torch.rand(2, 3, INPUT_SIZE, INPUT_SIZE)
    logits = net(dummy)
    print(f"Parameters: {count_parameters(net):,}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Probabilities: {net.predict_proba(dummy).tolist()}")
