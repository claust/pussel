"""Dual-input regressor for coarse position prediction with spatial correlation.

This model uses MobileNetV3-Small as the backbone for both piece and puzzle
encoders. It predicts the center coordinates (cx, cy) of where a piece
belongs in the puzzle, normalized to [0, 1].

Key insight: We must preserve SPATIAL information from the puzzle to enable
template matching. Global average pooling discards where things are located.

Architecture:
- Piece Encoder: MobileNetV3-Small -> global feature vector (576-dim)
- Puzzle Encoder: MobileNetV3-Small -> spatial feature map (16x16x576 for 512px input)
- Spatial Correlation: piece features correlated with each puzzle location
- Position Head: correlation map -> predicted (cx, cy)

For Phase 1 (frozen backbone), only the correlation and position layers
are trained.

Note: With 512x512 puzzle input, the spatial feature map is 16x16 (vs 8x8 for 256x256),
providing finer-grained spatial resolution for template matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


class SpatialCorrelationModule(nn.Module):
    """Computes spatial correlation between piece features and puzzle feature map.

    This module finds WHERE in the puzzle the piece features best match,
    producing a correlation/attention map over puzzle locations.
    """

    def __init__(self, feature_dim: int = 576, correlation_dim: int = 128):
        """Initialize the spatial correlation module.

        Args:
            feature_dim: Feature dimension from backbones.
            correlation_dim: Reduced dimension for correlation computation.
        """
        super().__init__()

        # Project features to lower dimension for efficient correlation
        self.piece_proj = nn.Sequential(
            nn.Linear(feature_dim, correlation_dim),
            nn.ReLU(),
        )
        self.puzzle_proj = nn.Sequential(
            nn.Conv2d(feature_dim, correlation_dim, 1),
            nn.ReLU(),
        )

        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1) * (correlation_dim**0.5))

    def forward(self, piece_feat: torch.Tensor, puzzle_feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial correlation between piece and puzzle.

        Args:
            piece_feat: Piece feature vector (B, feature_dim).
            puzzle_feat_map: Puzzle spatial features (B, feature_dim, H, W).

        Returns:
            Tuple of:
                - correlation_map: Softmax attention over puzzle locations (B, 1, H, W)
                - expected_position: Weighted average position (B, 2)
        """
        batch_size = piece_feat.shape[0]
        _, _, h, w = puzzle_feat_map.shape

        # Project to correlation dimension
        piece_proj = self.piece_proj(piece_feat)  # (B, correlation_dim)
        puzzle_proj = self.puzzle_proj(puzzle_feat_map)  # (B, correlation_dim, H, W)

        # Compute correlation: dot product at each spatial location
        # piece_proj: (B, C) -> (B, C, 1, 1)
        piece_proj = piece_proj.unsqueeze(-1).unsqueeze(-1)
        # correlation: (B, 1, H, W)
        correlation = (piece_proj * puzzle_proj).sum(dim=1, keepdim=True)
        correlation = correlation / self.temperature

        # Softmax to get attention weights
        correlation_flat = correlation.view(batch_size, -1)
        attention_flat = F.softmax(correlation_flat, dim=-1)
        attention_map = attention_flat.view(batch_size, 1, h, w)

        # Compute expected position as weighted average of grid coordinates
        # Create coordinate grids
        device = piece_feat.device
        y_coords = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device)
        x_coords = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Weighted average of coordinates
        attention_squeezed = attention_map.squeeze(1)  # (B, H, W)
        expected_x = (attention_squeezed * xx).sum(dim=[1, 2])
        expected_y = (attention_squeezed * yy).sum(dim=[1, 2])
        expected_position = torch.stack([expected_x, expected_y], dim=1)  # (B, 2)

        return attention_map, expected_position


class DualInputRegressorWithCorrelation(nn.Module):
    """Dual-input model with spatial correlation for position prediction.

    Uses spatial feature maps from the puzzle to enable proper template matching.
    """

    def __init__(
        self,
        correlation_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        """Initialize the model.

        Args:
            correlation_dim: Dimension for correlation computation.
            freeze_backbone: If True, freeze MobileNetV3 weights.
        """
        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Load pretrained MobileNetV3-Small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        piece_model = models.mobilenet_v3_small(weights=weights)
        puzzle_model = models.mobilenet_v3_small(weights=weights)

        # Extract feature extractors (before pooling)
        self.piece_features = piece_model.features
        self.puzzle_features = puzzle_model.features

        # MobileNetV3-Small feature dimension
        feature_dim = 576

        # Global pooling for piece
        self.piece_pool = nn.AdaptiveAvgPool2d(1)

        # Freeze backbones if requested
        if freeze_backbone:
            for param in self.piece_features.parameters():
                param.requires_grad = False
            for param in self.puzzle_features.parameters():
                param.requires_grad = False

        # Spatial correlation module
        self.spatial_correlation = SpatialCorrelationModule(
            feature_dim=feature_dim,
            correlation_dim=correlation_dim,
        )

        # Optional refinement head (learns to adjust the correlation-based prediction)
        self.refinement = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        # Whether to use refinement (can be disabled for pure correlation)
        self.use_refinement = True

    def forward(self, piece: torch.Tensor, puzzle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: (piece, puzzle) -> (cx, cy).

        Args:
            piece: Piece image tensor of shape (batch, 3, H, W).
            puzzle: Puzzle image tensor of shape (batch, 3, H, W).

        Returns:
            Tuple of:
                - position: Predicted (cx, cy) coordinates (batch, 2)
                - attention_map: Correlation map over puzzle (batch, 1, H, W)
        """
        # Extract spatial features from puzzle (keep spatial dimensions)
        puzzle_feat_map = self.puzzle_features(puzzle)  # (B, 576, H, W)

        # Extract features from piece and pool to vector
        piece_feat_map = self.piece_features(piece)  # (B, 576, h, w)
        piece_feat = self.piece_pool(piece_feat_map).flatten(1)  # (B, 576)

        # Compute spatial correlation
        attention_map, expected_pos = self.spatial_correlation(piece_feat, puzzle_feat_map)

        # Optional refinement
        if self.use_refinement:
            refinement = self.refinement(expected_pos)
            position = expected_pos + 0.1 * refinement  # Small adjustment
            position = torch.clamp(position, 0, 1)
        else:
            position = expected_pos

        return position, attention_map

    def predict_quadrant(self, piece: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
        """Predict which quadrant the piece belongs to."""
        position, _ = self.forward(piece, puzzle)
        cx, cy = position[:, 0], position[:, 1]
        quadrant = (cx >= 0.5).long() + 2 * (cy >= 0.5).long()
        return quadrant

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        for param in self.piece_features.parameters():
            param.requires_grad = True
        for param in self.puzzle_features.parameters():
            param.requires_grad = True
        self.freeze_backbone = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("Model Architecture Test - DualInputRegressorWithCorrelation")
    print("Experiment 8: High Resolution (512x512 puzzle)")
    print("=" * 60)

    batch_size = 4
    piece_size = 128
    puzzle_size = 512  # High resolution

    dummy_piece = torch.randn(batch_size, 3, piece_size, piece_size)
    dummy_puzzle = torch.randn(batch_size, 3, puzzle_size, puzzle_size)

    # Test new model with spatial correlation
    print("\n--- Model: With Spatial Correlation ---")
    model = DualInputRegressorWithCorrelation(freeze_backbone=True)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Forward pass
    position, attention_map = model(dummy_piece, dummy_puzzle)
    print(f"\nPiece input shape: {dummy_piece.shape}")
    print(f"Puzzle input shape: {dummy_puzzle.shape}")
    print(f"Position output shape: {position.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Position values: {position[0].tolist()}")

    # Check attention map sums to 1
    attn_sums = attention_map.sum(dim=[2, 3]).squeeze()
    print(f"Attention map sums: {attn_sums.tolist()} (should be ~1.0)")

    # Test quadrant prediction
    quadrants = model.predict_quadrant(dummy_piece, dummy_puzzle)
    print(f"Quadrant predictions: {quadrants.tolist()}")

    # Note the larger feature map size with 512px input
    print(f"\nNote: With 512x512 puzzle input, attention map is {attention_map.shape[2]}x{attention_map.shape[3]}")
    print("This provides finer spatial resolution compared to 8x8 with 256x256 input.")
