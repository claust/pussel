"""Binary puzzle-piece classifier network.

Backend copy of the exp24 model architecture
(network/experiments/exp24_piece_classifier/model.py); the two definitions
must stay in sync so checkpoints load. A single MobileNetV3-Small backbone
(~1.0M parameters, 128px input) with a small binary head, chosen to be cheap
enough for the polled preview endpoint and a later on-device port.
"""

import torch
import torch.nn as nn
from torchvision import models  # type: ignore[import-untyped]
from torchvision.models import MobileNet_V3_Small_Weights  # type: ignore[import-untyped]

INPUT_SIZE = 128


class PieceClassifier(nn.Module):
    """MobileNetV3-Small binary classifier: puzzle piece vs. not a piece."""

    def __init__(self, pretrained: bool = False, dropout: float = 0.2, head_dim: int = 128):
        """Initialize the classifier.

        Args:
            pretrained: Whether to initialize the backbone with ImageNet weights
                (False in the backend; weights come from the checkpoint).
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
