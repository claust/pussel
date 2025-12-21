"""Dual-input regressor with SPATIAL rotation prediction.

This model fixes the critical architectural flaw from exp10:
- Exp10: Global pooling destroyed spatial info before rotation head
- Exp11: Rotation head receives spatial feature maps directly

The key insight: To predict rotation, the model must know WHERE textures
are located in the piece (e.g., "sky at top" vs "sky at bottom").
Global pooling makes these indistinguishable. This model preserves
the spatial structure for rotation prediction.
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


class SpatialRotationHead(nn.Module):
    """Rotation head that PRESERVES spatial information.

    Unlike exp10's rotation head (which used globally-pooled features),
    this head processes the spatial feature map directly with convolutions
    before flattening. This allows it to distinguish between orientations
    like "sky at top" vs "sky at bottom" which are identical after pooling.

    Architecture:
        Input: (B, 576, 4, 4) - spatial features from backbone
        Conv2d(576, 128) -> BatchNorm -> ReLU
        Conv2d(128, 64) -> BatchNorm -> ReLU
        Flatten -> (B, 1024)
        Linear(1024, 256) -> ReLU -> Dropout
        Linear(256, 4) -> rotation logits
    """

    def __init__(
        self,
        in_channels: int = 576,
        spatial_size: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """Initialize the spatial rotation head.

        Args:
            in_channels: Number of input channels from backbone.
            spatial_size: Spatial dimension of feature map (assumes square).
            hidden_dim: Hidden dimension for the MLP.
            dropout: Dropout rate for regularization.
        """
        super().__init__()

        # Convolutional layers to process spatial features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Calculate flattened size
        flat_size = 64 * spatial_size * spatial_size  # 64 * 4 * 4 = 1024

        # MLP for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # 4 rotation classes
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            spatial_features: Spatial feature map (B, C, H, W).

        Returns:
            Rotation logits (B, 4).
        """
        x = self.conv_layers(spatial_features)
        logits = self.classifier(x)
        return logits


class DualInputRegressorWithSpatialRotation(nn.Module):
    """Dual-input model with position AND spatial rotation prediction.

    This is the key fix for exp10's failure:
    - Position prediction uses spatial correlation (same as exp9/10)
    - Rotation prediction uses SPATIAL features, not pooled features

    The rotation head receives the raw (B, 576, 4, 4) feature map from
    the piece backbone, allowing it to learn texture orientation cues
    that were destroyed by pooling in exp10.
    """

    def __init__(
        self,
        correlation_dim: int = 128,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        rotation_dropout: float = 0.2,
    ):
        """Initialize the model.

        Args:
            correlation_dim: Dimension for correlation computation.
            freeze_backbone: If True, freeze MobileNetV3 weights.
            dropout: Dropout rate for position head.
            rotation_dropout: Dropout rate for rotation head (higher by default).
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

        # NEW: SPATIAL Rotation classification head
        # Takes spatial features (before pooling) to preserve orientation info
        self.rotation_head = SpatialRotationHead(
            in_channels=feature_dim,
            spatial_size=4,  # 128x128 input -> 4x4 feature map
            hidden_dim=256,
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

        # KEY FIX: Rotation prediction from SPATIAL features (not pooled!)
        # This preserves the "sky at top" vs "sky at bottom" distinction
        rotation_logits = self.rotation_head(piece_feat_map)

        return position, rotation_logits, attention_map

    def predict_quadrant(self, piece: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
        """Predict which quadrant the piece belongs to."""
        position, _, _ = self.forward(piece, puzzle)
        cx, cy = position[:, 0], position[:, 1]
        quadrant = (cx >= 0.5).long() + 2 * (cy >= 0.5).long()
        return quadrant

    def predict_rotation(self, piece: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
        """Predict the rotation class (0, 1, 2, 3 for 0, 90, 180, 270 degrees)."""
        _, rotation_logits, _ = self.forward(piece, puzzle)
        return rotation_logits.argmax(dim=1)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers for full fine-tuning."""
        for param in self.piece_features.parameters():
            param.requires_grad = True
        for param in self.puzzle_features.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        self._unfrozen_layers = set(range(len(self.piece_features)))

    def unfreeze_layers(self, layer_indices: list[int]) -> None:
        """Unfreeze specific layers by index for gradual unfreezing.

        Args:
            layer_indices: List of layer indices to unfreeze.
        """
        for idx in layer_indices:
            if idx < len(self.piece_features):
                for param in self.piece_features[idx].parameters():
                    param.requires_grad = True
            if idx < len(self.puzzle_features):
                for param in self.puzzle_features[idx].parameters():
                    param.requires_grad = True
            self._unfrozen_layers.add(idx)
        self.freeze_backbone = False

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """Get parameter groups with differential learning rates.

        Args:
            backbone_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for heads (position, rotation, correlation).
            weight_decay: Weight decay for regularization.

        Returns:
            List of parameter group dictionaries for optimizer.
        """
        # Backbone parameters (both piece and puzzle encoders)
        backbone_params = [p for p in self.piece_features.parameters() if p.requires_grad]
        backbone_params += [p for p in self.puzzle_features.parameters() if p.requires_grad]

        # Head parameters (correlation module, refinement, and rotation head)
        head_params = [p for p in self.spatial_correlation.parameters() if p.requires_grad]
        head_params += [p for p in self.refinement.parameters() if p.requires_grad]
        head_params += [p for p in self.rotation_head.parameters() if p.requires_grad]

        param_groups = []
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                    "name": "backbone",
                }
            )
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                    "name": "heads",
                }
            )

        return param_groups

    def get_num_backbone_layers(self) -> int:
        """Get the number of layers in the backbone."""
        return len(self.piece_features)

    def get_layer_info(self) -> list[dict]:
        """Get information about each backbone layer.

        Returns:
            List of dicts with layer info (index, name, param count, unfrozen).
        """
        info = []
        for idx, layer in enumerate(self.piece_features):
            param_count = sum(p.numel() for p in layer.parameters())
            trainable = any(p.requires_grad for p in layer.parameters())
            info.append(
                {
                    "index": idx,
                    "name": layer.__class__.__name__,
                    "params": param_count,
                    "trainable": trainable,
                }
            )
        return info


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("Model Architecture Test - Spatial Rotation Head")
    print("=" * 60)

    batch_size = 4
    piece_size = 128
    puzzle_size = 256

    dummy_piece = torch.randn(batch_size, 3, piece_size, piece_size)
    dummy_puzzle = torch.randn(batch_size, 3, puzzle_size, puzzle_size)

    # Test unfrozen backbone
    print("\n--- Unfrozen Backbone (Full Fine-tuning) ---")
    model = DualInputRegressorWithSpatialRotation(freeze_backbone=False)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Count rotation head parameters specifically
    rotation_params = sum(p.numel() for p in model.rotation_head.parameters())
    print(f"Spatial rotation head parameters: {rotation_params:,}")

    # Compare with exp10's rotation head size
    exp10_rotation_params = 576 * 256 + 256 + 256 * 64 + 64 + 64 * 4 + 4
    print(f"Exp10 rotation head parameters: {exp10_rotation_params:,}")
    print(f"Parameter increase: {rotation_params - exp10_rotation_params:,}")

    # Test parameter groups
    print("\n--- Parameter Groups for Differential LR ---")
    param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        print(f"  {group['name']}: {param_count:,} params, LR={group['lr']}")

    # Forward pass test
    print("\n--- Forward Pass ---")
    position, rotation_logits, attention_map = model(dummy_piece, dummy_puzzle)
    print(f"Position output shape: {position.shape}")
    print(f"Rotation logits shape: {rotation_logits.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Position values: {position[0].tolist()}")
    print(f"Rotation logits: {rotation_logits[0].tolist()}")
    print(f"Predicted rotation: {rotation_logits.argmax(dim=1).tolist()}")

    # Test SpatialRotationHead in isolation
    print("\n--- SpatialRotationHead Isolation Test ---")
    spatial_head = SpatialRotationHead(in_channels=576, spatial_size=4)
    dummy_features = torch.randn(batch_size, 576, 4, 4)
    rotation_out = spatial_head(dummy_features)
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {rotation_out.shape}")
