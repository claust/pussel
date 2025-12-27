"""Dual-input regressor with backbone fine-tuning support.

This model builds on exp7's DualInputRegressorWithCorrelation architecture,
adding support for:
- Differential learning rates (lower LR for backbone, higher for heads)
- Gradual unfreezing (progressively unfreeze layers during training)
- Additional regularization (dropout) to prevent overfitting

Key insight from exp7: Frozen ImageNet features achieved 67% test accuracy.
By fine-tuning with task-specific features, we aim to push past 70%.

Architecture:
- Piece Encoder: MobileNetV3-Small (unfrozen with lower LR)
- Puzzle Encoder: MobileNetV3-Small (unfrozen with lower LR)
- Spatial Correlation: Piece features correlated with puzzle spatial map
- Position Head: Correlation map -> predicted (cx, cy)
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

    Supports backbone fine-tuning with differential learning rates.
    """

    def __init__(
        self,
        correlation_dim: int = 128,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        """Initialize the model.

        Args:
            correlation_dim: Dimension for correlation computation.
            freeze_backbone: If True, freeze MobileNetV3 weights.
            dropout: Dropout rate for regularization.
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
            self._freeze_all_backbone_layers()

        # Spatial correlation module (with dropout)
        self.spatial_correlation = SpatialCorrelationModule(
            feature_dim=feature_dim,
            correlation_dim=correlation_dim,
            dropout=dropout,
        )

        # Refinement head with dropout
        self.refinement = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
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
        """Unfreeze all backbone layers for full fine-tuning."""
        for param in self.piece_features.parameters():
            param.requires_grad = True
        for param in self.puzzle_features.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        # Mark all layers as unfrozen
        self._unfrozen_layers = set(range(len(self.piece_features)))

    def unfreeze_layers(self, layer_indices: list[int]) -> None:
        """Unfreeze specific layers by index for gradual unfreezing.

        MobileNetV3-Small has layers 0-12 in its features block.
        Later layers (higher indices) are more task-specific.

        Recommended unfreezing order (later layers first):
        - Phase 1: layers [10, 11, 12] (final conv blocks)
        - Phase 2: layers [7, 8, 9] (middle blocks)
        - Phase 3: layers [4, 5, 6] (early-middle blocks)
        - Phase 4: layers [0, 1, 2, 3] (early blocks) - usually keep frozen

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
            head_lr: Learning rate for correlation and refinement heads.
            weight_decay: Weight decay for regularization.

        Returns:
            List of parameter group dictionaries for optimizer.
        """
        # Backbone parameters (both piece and puzzle encoders)
        backbone_params = [p for p in self.piece_features.parameters() if p.requires_grad]
        backbone_params += [p for p in self.puzzle_features.parameters() if p.requires_grad]

        # Head parameters (correlation module and refinement)
        head_params = [p for p in self.spatial_correlation.parameters() if p.requires_grad]
        head_params += [p for p in self.refinement.parameters() if p.requires_grad]

        param_groups = []
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"}
            )
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "heads"})

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
    print("Model Architecture Test - Fine-tunable Backbone")
    print("=" * 60)

    batch_size = 4
    piece_size = 128
    puzzle_size = 256

    dummy_piece = torch.randn(batch_size, 3, piece_size, piece_size)
    dummy_puzzle = torch.randn(batch_size, 3, puzzle_size, puzzle_size)

    # Test frozen backbone
    print("\n--- Frozen Backbone ---")
    model_frozen = DualInputRegressorWithCorrelation(freeze_backbone=True)
    total_params = count_parameters(model_frozen, trainable_only=False)
    trainable_params = count_parameters(model_frozen, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test unfrozen backbone
    print("\n--- Unfrozen Backbone (Full Fine-tuning) ---")
    model_unfrozen = DualInputRegressorWithCorrelation(freeze_backbone=False)
    total_params = count_parameters(model_unfrozen, trainable_only=False)
    trainable_params = count_parameters(model_unfrozen, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test gradual unfreezing
    print("\n--- Gradual Unfreezing (Last 3 Layers) ---")
    model_gradual = DualInputRegressorWithCorrelation(freeze_backbone=True)
    print(f"Before unfreezing: {count_parameters(model_gradual):,} trainable")

    # Unfreeze last 3 layers
    model_gradual.unfreeze_layers([10, 11, 12])
    print(f"After unfreezing [10,11,12]: {count_parameters(model_gradual):,} trainable")

    # Show layer info
    print("\nBackbone layers:")
    for info in model_gradual.get_layer_info():
        status = "TRAINABLE" if info["trainable"] else "frozen"
        print(f"  Layer {info['index']:2d}: {info['name']:20s} " f"({info['params']:,} params) - {status}")

    # Test parameter groups
    print("\n--- Parameter Groups for Differential LR ---")
    model_unfrozen.unfreeze_backbone()
    param_groups = model_unfrozen.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        print(f"  {group['name']}: {param_count:,} params, LR={group['lr']}")

    # Forward pass test
    print("\n--- Forward Pass ---")
    position, attention_map = model_unfrozen(dummy_piece, dummy_puzzle)
    print(f"Position output shape: {position.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Position values: {position[0].tolist()}")
