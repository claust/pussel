"""Neural network model for multi-puzzle high-resolution experiment.

This model uses a dual-input architecture optimized for higher resolution
puzzle images (512x512) to provide more detail for piece-puzzle matching.

Key changes from exp5:
1. Additional conv layer to handle 512x512 puzzle images
2. Separate backbones for piece (64x64) and puzzle (512x512) with different depths
3. Feature map alignment for spatial correlation

Architecture:
- Piece backbone: 5 conv layers, 64x64 -> 2x2 feature map
- Puzzle backbone: 6 conv layers, 512x512 -> 16x16 feature map
- Spatial correlation module for matching
- Feature fusion and classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default grid dimensions
DEFAULT_NUM_CELLS = 950


class PieceBackbone(nn.Module):
    """CNN backbone for piece images (64x64)."""

    def __init__(self, feature_dim: int = 256):
        """Initialize the piece backbone.

        Args:
            feature_dim: Output feature dimension.
        """
        super().__init__()
        self.feature_dim = feature_dim

        # 64 -> 32 -> 16 -> 8 -> 4 -> 2
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, 64, 64).

        Returns:
            Feature map (B, feature_dim, 2, 2).
        """
        return self.layers(x)


class PuzzleBackbone(nn.Module):
    """CNN backbone for puzzle images (512x512)."""

    def __init__(self, feature_dim: int = 256):
        """Initialize the puzzle backbone.

        Args:
            feature_dim: Output feature dimension.
        """
        super().__init__()
        self.feature_dim = feature_dim

        # 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=1, padding=1),  # Keep 16x16
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, 512, 512).

        Returns:
            Feature map (B, feature_dim, 16, 16).
        """
        return self.layers(x)


class SpatialCorrelationModule(nn.Module):
    """Computes spatial correlation between piece and puzzle feature maps.

    This module computes a correlation map showing where piece features
    best match locations in the puzzle.
    """

    def __init__(self, feature_dim: int = 256, correlation_dim: int = 64):
        """Initialize the SpatialCorrelationModule.

        Args:
            feature_dim: Feature dimension from backbones.
            correlation_dim: Reduced dimension for correlation computation.
        """
        super().__init__()

        # Project features to lower dimension for efficient correlation
        self.piece_proj = nn.Sequential(
            nn.Conv2d(feature_dim, correlation_dim, 1),
            nn.BatchNorm2d(correlation_dim),
            nn.ReLU(),
        )
        self.puzzle_proj = nn.Sequential(
            nn.Conv2d(feature_dim, correlation_dim, 1),
            nn.BatchNorm2d(correlation_dim),
            nn.ReLU(),
        )

        # Learnable temperature for attention
        self.temperature = nn.Parameter(torch.ones(1) * (correlation_dim**0.5))

        # Process correlation map to extract position features
        # Input: 16x16 correlation map
        self.correlation_processor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.output_dim = 128

    def forward(self, piece_feat_map: torch.Tensor, puzzle_feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial correlation between piece and puzzle.

        Args:
            piece_feat_map: Piece feature map (B, C, Hp, Wp).
            puzzle_feat_map: Puzzle feature map (B, C, H, W).

        Returns:
            Tuple of:
                - correlation_features: Processed correlation features (B, 128).
                - correlation_map: Raw correlation map (B, 1, H, W).
        """
        batch_size = piece_feat_map.shape[0]

        # Project to lower dimension
        piece_proj = self.piece_proj(piece_feat_map)
        puzzle_proj = self.puzzle_proj(puzzle_feat_map)

        # Global average pool piece features to get a single descriptor
        piece_vec = piece_proj.mean(dim=[2, 3])

        # Compute correlation: how well does each puzzle location match the piece?
        _, d, h, w = puzzle_proj.shape
        puzzle_flat = puzzle_proj.view(batch_size, d, h * w)

        # Correlation scores
        correlation = torch.bmm(piece_vec.unsqueeze(1), puzzle_flat)
        correlation = correlation / self.temperature

        # Reshape to spatial map
        correlation_map = correlation.view(batch_size, 1, h, w)

        # Apply softmax to get attention-like weights
        correlation_softmax = F.softmax(correlation_map.view(batch_size, -1), dim=-1)
        correlation_softmax = correlation_softmax.view(batch_size, 1, h, w)

        # Process correlation map to extract position-aware features
        correlation_features = self.correlation_processor(correlation_softmax)
        correlation_features = correlation_features.view(batch_size, -1)

        return correlation_features, correlation_map


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for combining piece and puzzle features."""

    def __init__(self, dim: int = 256, output_dim: int = 256):
        """Initialize the CrossAttentionFusion module.

        Args:
            dim: Feature dimension from backbones.
            output_dim: Desired output dimension.
        """
        super().__init__()

        # Attention computation
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

        # Feature projections
        self.piece_proj = nn.Linear(dim, dim)
        self.puzzle_proj = nn.Linear(dim, dim)

        # Layer normalization
        self.norm_piece = nn.LayerNorm(dim)
        self.norm_puzzle = nn.LayerNorm(dim)

        # Output fusion network
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, piece_feat: torch.Tensor, puzzle_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention-based fusion.

        Args:
            piece_feat: Piece features (batch_size, dim).
            puzzle_feat: Puzzle features (batch_size, dim).

        Returns:
            Fused features (batch_size, output_dim).
        """
        # Normalize inputs
        piece_feat = self.norm_piece(piece_feat)
        puzzle_feat = self.norm_puzzle(puzzle_feat)

        # Project features
        piece_proj = self.piece_proj(piece_feat)
        puzzle_proj = self.puzzle_proj(puzzle_feat)

        # Compute attention weights
        combined = torch.cat([piece_proj, puzzle_proj], dim=1)
        attn_logits = self.attention(combined)
        attn_weights = torch.sigmoid(attn_logits)

        # Apply attention to puzzle features
        attended_puzzle = attn_weights * puzzle_proj

        # Concatenate piece with attended puzzle
        fused = torch.cat([piece_proj, attended_puzzle], dim=1)

        return self.fusion(fused)


class HighResDualInputClassifier(nn.Module):
    """Cell classifier with dual-input architecture for high-resolution puzzles.

    This model uses separate backbones for piece (64x64) and puzzle (512x512)
    images, optimized for the different input resolutions.
    """

    def __init__(
        self,
        num_cells: int = DEFAULT_NUM_CELLS,
        feature_dim: int = 256,
        fusion_dim: int = 256,
    ):
        """Initialize the network.

        Args:
            num_cells: Number of output classes (grid cells).
            feature_dim: Dimension of backbone features.
            fusion_dim: Dimension of fused features.
        """
        super().__init__()
        self.num_cells = num_cells
        self.feature_dim = feature_dim

        # Separate backbones for different input sizes
        self.piece_backbone = PieceBackbone(feature_dim=feature_dim)
        self.puzzle_backbone = PuzzleBackbone(feature_dim=feature_dim)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Spatial correlation module
        self.spatial_correlation = SpatialCorrelationModule(
            feature_dim=feature_dim,
            correlation_dim=64,
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            dim=feature_dim,
            output_dim=fusion_dim,
        )

        # Combined feature dimension: fusion + spatial correlation
        self.combined_dim = fusion_dim + self.spatial_correlation.output_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_cells),
        )

    def forward(self, piece: torch.Tensor, puzzle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: (piece, puzzle) -> cell logits.

        Args:
            piece: Piece image tensor of shape (batch, 3, 64, 64).
            puzzle: Puzzle image tensor of shape (batch, 3, 512, 512).

        Returns:
            Tuple of (logits, correlation_map).
            - logits: Cell classification logits of shape (batch, num_cells).
            - correlation_map: Attention map showing where piece matches puzzle.
        """
        # Extract spatial feature maps
        piece_feat_map = self.piece_backbone(piece)  # (B, 256, 2, 2)
        puzzle_feat_map = self.puzzle_backbone(puzzle)  # (B, 256, 16, 16)

        # Global features for fusion
        piece_feat = self.gap(piece_feat_map).view(piece_feat_map.size(0), -1)
        puzzle_feat = self.gap(puzzle_feat_map).view(puzzle_feat_map.size(0), -1)

        # Spatial correlation features
        corr_feat, corr_map = self.spatial_correlation(piece_feat_map, puzzle_feat_map)

        # Cross-attention fusion
        fused_feat = self.fusion(piece_feat, puzzle_feat)

        # Combine fusion and correlation features
        combined = torch.cat([fused_feat, corr_feat], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits, corr_map

    def predict_proba(self, piece: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over cells.

        Args:
            piece: Piece image tensor.
            puzzle: Puzzle image tensor.

        Returns:
            Probability tensor of shape (batch, num_cells).
        """
        logits, _ = self.forward(piece, puzzle)
        return F.softmax(logits, dim=1)

    def predict(self, piece: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
        """Get predicted cell index.

        Args:
            piece: Piece image tensor.
            puzzle: Puzzle image tensor.

        Returns:
            Predicted cell indices of shape (batch,).
        """
        logits, _ = self.forward(piece, puzzle)
        return torch.argmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_backbone_parameters(model: HighResDualInputClassifier) -> int:
    """Count parameters in both backbones."""
    piece_params = sum(p.numel() for p in model.piece_backbone.parameters())
    puzzle_params = sum(p.numel() for p in model.puzzle_backbone.parameters())
    return piece_params + puzzle_params


if __name__ == "__main__":
    print("=" * 60)
    print("Model Architecture Test - High-Res Dual-Input Classifier")
    print("=" * 60)

    batch_size = 2
    dummy_piece = torch.randn(batch_size, 3, 64, 64)
    dummy_puzzle = torch.randn(batch_size, 3, 512, 512)

    model = HighResDualInputClassifier(num_cells=950)
    total_params = count_parameters(model)
    backbone_params = count_backbone_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")

    # Test forward pass
    logits, corr_map = model(dummy_piece, dummy_puzzle)
    print(f"\nPiece input shape: {dummy_piece.shape}")
    print(f"Puzzle input shape: {dummy_puzzle.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Correlation map shape: {corr_map.shape}")

    # Test prediction methods
    probs = model.predict_proba(dummy_piece, dummy_puzzle)
    preds = model.predict(dummy_piece, dummy_puzzle)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Sample prediction: cell {preds[0].item()}")

    # Verify feature map sizes
    print("\n--- Feature Map Sizes ---")
    piece_feat = model.piece_backbone(dummy_piece)
    puzzle_feat = model.puzzle_backbone(dummy_puzzle)
    print(f"Piece feature map: {piece_feat.shape}")
    print(f"Puzzle feature map: {puzzle_feat.shape}")
