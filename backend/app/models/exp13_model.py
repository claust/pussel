"""Dual-input regressor with rotation correlation for puzzle piece matching.

This model predicts both position and rotation of a puzzle piece by comparing
the piece features with puzzle features. Key insight: Rotation is not an
intrinsic property of the piece - it's a relationship between piece and puzzle.

Adapted from network/experiments/exp13_rotation_correlation_5k/model.py for
backend inference.
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

    def __init__(
        self,
        feature_dim: int = 576,
        correlation_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize the spatial correlation module.

        Args:
            feature_dim: Feature dimension from backbones.
            correlation_dim: Reduced dimension for correlation computation.
            dropout: Dropout rate for regularization.
        """
        super().__init__()

        # Project features to lower dimension for efficient correlation
        self.piece_proj = nn.Sequential(
            nn.Linear(feature_dim, correlation_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.puzzle_proj = nn.Sequential(
            nn.Conv2d(feature_dim, correlation_dim, 1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
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
        piece_proj = piece_proj.unsqueeze(-1).unsqueeze(-1)
        correlation = (piece_proj * puzzle_proj).sum(dim=1, keepdim=True)
        correlation = correlation / self.temperature

        # Softmax to get attention weights
        correlation_flat = correlation.view(batch_size, -1)
        attention_flat = F.softmax(correlation_flat, dim=-1)
        attention_map = attention_flat.view(batch_size, 1, h, w)

        # Compute expected position as weighted average of grid coordinates
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


class RotationCorrelationModule(nn.Module):
    """Rotation prediction via correlation between piece and puzzle.

    For each rotation r in [0, 90, 180, 270]:
        1. Rotate the piece feature map by r degrees
        2. Extract the puzzle region at the predicted position
        3. Compute similarity between rotated piece and puzzle region
        4. The rotation with highest similarity is the prediction
    """

    def __init__(
        self,
        feature_dim: int = 576,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """Initialize the rotation correlation module.

        Args:
            feature_dim: Feature dimension from backbones.
            hidden_dim: Hidden dimension for comparison network.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable projection for piece and puzzle features before comparison
        self.piece_proj = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.puzzle_proj = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        # Comparison network: takes concatenated piece and puzzle features
        # and produces a similarity score
        self.comparison_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global pool to (B, 64, 1, 1)
            nn.Flatten(),  # (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),  # Single similarity score
        )

    def _rotate_feature_map(self, feat_map: torch.Tensor, rotation_idx: int) -> torch.Tensor:
        """Rotate a feature map by 0, 90, 180, or 270 degrees.

        Args:
            feat_map: Feature map of shape (B, C, H, W).
            rotation_idx: Rotation index (0=0deg, 1=90deg, 2=180deg, 3=270deg).

        Returns:
            Rotated feature map of shape (B, C, H, W).
        """
        if rotation_idx == 0:
            return feat_map
        elif rotation_idx == 1:  # 90 degrees clockwise
            return torch.rot90(feat_map, k=-1, dims=[2, 3])
        elif rotation_idx == 2:  # 180 degrees
            return torch.rot90(feat_map, k=2, dims=[2, 3])
        else:  # 270 degrees clockwise = 90 degrees counter-clockwise
            return torch.rot90(feat_map, k=1, dims=[2, 3])

    def _extract_region(
        self,
        puzzle_feat_map: torch.Tensor,
        position: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Extract puzzle region at the predicted position using quadrant indexing.

        For 2x2 quadrant prediction, this uses simple slicing instead of grid_sample
        to avoid MPS compatibility issues with grid_sampler backward pass.

        Args:
            puzzle_feat_map: Puzzle features (B, C, H_puzzle, W_puzzle).
            position: Predicted positions (B, 2) in [0, 1] normalized coords.
            target_size: Desired output size (H_piece, W_piece).

        Returns:
            Extracted regions (B, C, H_piece, W_piece).
        """
        batch_size = puzzle_feat_map.shape[0]
        _, c, h_puzzle, w_puzzle = puzzle_feat_map.shape
        h_piece, w_piece = target_size

        # Determine quadrant indices from position
        x_idx = (position[:, 0] >= 0.5).long()  # 0 for left, 1 for right
        y_idx = (position[:, 1] >= 0.5).long()  # 0 for top, 1 for bottom

        # Calculate start indices for slicing
        half_h = h_puzzle // 2
        half_w = w_puzzle // 2

        # Create output tensor
        device = puzzle_feat_map.device
        extracted = torch.zeros(batch_size, c, h_piece, w_piece, device=device)

        # Extract regions for each sample
        for b in range(batch_size):
            h_start = int(y_idx[b].item() * half_h)
            w_start = int(x_idx[b].item() * half_w)
            h_end = h_start + half_h
            w_end = w_start + half_w

            region = puzzle_feat_map[b, :, h_start:h_end, w_start:w_end]

            # Resize to target size if needed
            if region.shape[1:] != (h_piece, w_piece):
                region = F.interpolate(
                    region.unsqueeze(0),
                    size=(h_piece, w_piece),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)

            extracted[b] = region

        return extracted

    def forward(
        self,
        piece_feat_map: torch.Tensor,
        puzzle_feat_map: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rotation scores by comparing piece to puzzle.

        Args:
            piece_feat_map: Piece spatial features (B, C, H_piece, W_piece).
            puzzle_feat_map: Puzzle spatial features (B, C, H_puzzle, W_puzzle).
            position: Predicted position (B, 2) in [0, 1] coords.

        Returns:
            Rotation logits (B, 4) for rotations [0, 90, 180, 270].
        """
        _, _, h_piece, w_piece = piece_feat_map.shape

        # Extract puzzle region at predicted position
        puzzle_region = self._extract_region(puzzle_feat_map, position, (h_piece, w_piece))

        # Project features
        puzzle_proj = self.puzzle_proj(puzzle_region)  # (B, hidden_dim, H, W)

        # Compute similarity score for each rotation
        rotation_scores = []
        for rot_idx in range(4):
            # Rotate piece features
            rotated_piece = self._rotate_feature_map(piece_feat_map, rot_idx)
            piece_proj = self.piece_proj(rotated_piece)  # (B, hidden_dim, H, W)

            # Concatenate and compare
            combined = torch.cat([piece_proj, puzzle_proj], dim=1)  # (B, 2*hidden, H, W)
            score = self.comparison_net(combined)  # (B, 1)
            rotation_scores.append(score)

        # Stack to get (B, 4) rotation logits
        rotation_logits = torch.cat(rotation_scores, dim=1)

        return rotation_logits


class DualInputRegressorWithRotationCorrelation(nn.Module):
    """Dual-input model with position AND rotation correlation prediction.

    Uses spatial correlation for position prediction and rotation correlation
    that compares piece to puzzle at each rotation angle.
    """

    def __init__(
        self,
        correlation_dim: int = 128,
        rotation_hidden_dim: int = 128,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        rotation_dropout: float = 0.2,
    ):
        """Initialize the model.

        Args:
            correlation_dim: Dimension for position correlation computation.
            rotation_hidden_dim: Hidden dimension for rotation comparison.
            freeze_backbone: If True, freeze MobileNetV3 weights.
            dropout: Dropout rate for position head.
            rotation_dropout: Dropout rate for rotation head.
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

        # Global pooling for piece (only used for position correlation)
        self.piece_pool = nn.AdaptiveAvgPool2d(1)

        # Freeze backbones if requested
        if freeze_backbone:
            self._freeze_all_backbone_layers()

        # Spatial correlation module for position (with dropout)
        self.spatial_correlation = SpatialCorrelationModule(
            feature_dim=feature_dim,
            correlation_dim=correlation_dim,
            dropout=dropout,
        )

        # Position refinement head with dropout
        self.refinement = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        # Rotation correlation module
        self.rotation_correlation = RotationCorrelationModule(
            feature_dim=feature_dim,
            hidden_dim=rotation_hidden_dim,
            dropout=rotation_dropout,
        )

        # Whether to use refinement (can be disabled for pure correlation)
        self.use_refinement = True

        # Track unfrozen layers for gradual unfreezing
        self._unfrozen_layers: set[int] = set()

    def _freeze_all_backbone_layers(self) -> None:
        """Freeze all layers in both backbones."""
        for param in self.piece_features.parameters():
            param.requires_grad = False
        for param in self.puzzle_features.parameters():
            param.requires_grad = False
        self._unfrozen_layers.clear()

    def forward(self, piece: torch.Tensor, puzzle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: (piece, puzzle) -> (position, rotation_logits, attention_map).

        Args:
            piece: Piece image tensor of shape (batch, 3, H, W).
            puzzle: Puzzle image tensor of shape (batch, 3, H, W).

        Returns:
            Tuple of:
                - position: Predicted (cx, cy) coordinates (batch, 2)
                - rotation_logits: Rotation class logits (batch, 4)
                - attention_map: Correlation map over puzzle (batch, 1, H, W)
        """
        # Extract spatial features from puzzle (keep spatial dimensions)
        puzzle_feat_map = self.puzzle_features(puzzle)  # (B, 576, H, W)

        # Extract SPATIAL features from piece (keep spatial dimensions!)
        piece_feat_map = self.piece_features(piece)  # (B, 576, 4, 4)

        # Pool piece features for position correlation only
        piece_feat = self.piece_pool(piece_feat_map).flatten(1)  # (B, 576)

        # Compute spatial correlation for position
        attention_map, expected_pos = self.spatial_correlation(piece_feat, puzzle_feat_map)

        # Optional position refinement
        if self.use_refinement:
            refinement = self.refinement(expected_pos)
            position = expected_pos + 0.1 * refinement  # Small adjustment
            position = torch.clamp(position, 0, 1)
        else:
            position = expected_pos

        # Rotation prediction by comparing piece to puzzle
        rotation_logits = self.rotation_correlation(piece_feat_map, puzzle_feat_map, position)

        return position, rotation_logits, attention_map
