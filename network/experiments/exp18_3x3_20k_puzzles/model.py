"""Model for 3x3 grid position prediction with ShuffleNetV2_x0.5 backbone.

Exp18: Same architecture as exp17, using 20K training puzzles instead of 10K.
Uses ShuffleNetV2_x0.5 for fast experimentation.

Key features:
- Region extraction handles 3x3 grid
- predict_cell() method for 9-class cell prediction
"""

from typing import Literal

import timm  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ShuffleNet_V2_X0_5_Weights

BackboneType = Literal["repvgg_a0", "mobileone_s0", "shufflenet_v2_x0_5", "mobilenet_v3_small"]


def get_backbone(backbone_name: BackboneType, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Create a backbone network and return it with its feature dimension.

    Args:
        backbone_name: Name of the backbone to use.
        pretrained: Whether to use pretrained weights.

    Returns:
        Tuple of (backbone_module, feature_dim).
    """
    if backbone_name == "repvgg_a0":
        # RepVGG-A0 from timm - fast re-parameterizable architecture
        model = timm.create_model("repvgg_a0", pretrained=pretrained, features_only=True)
        # Get feature dim from last stage
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = model(dummy)[-1]
            feature_dim = features.shape[1]
        return model, feature_dim

    elif backbone_name == "mobileone_s0":
        # MobileOne-S0 from timm - Apple's fastest backbone
        model = timm.create_model("mobileone_s0", pretrained=pretrained, features_only=True)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = model(dummy)[-1]
            feature_dim = features.shape[1]
        return model, feature_dim

    elif backbone_name == "shufflenet_v2_x0_5":
        # ShuffleNetV2 x0.5 from torchvision - very fast channel shuffle
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = models.shufflenet_v2_x0_5(weights=weights)
        # Extract feature extractor (everything before final FC)
        # ShuffleNetV2 structure: conv1, maxpool, stage2, stage3, stage4, conv5
        backbone = nn.Sequential(
            full_model.conv1,
            full_model.maxpool,
            full_model.stage2,
            full_model.stage3,
            full_model.stage4,
            full_model.conv5,
        )
        feature_dim = 1024  # ShuffleNetV2 x0.5 final conv outputs 1024 channels
        return backbone, feature_dim

    elif backbone_name == "mobilenet_v3_small":
        # MobileNetV3-Small from torchvision - baseline for comparison
        from torchvision.models import MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = models.mobilenet_v3_small(weights=weights)
        backbone = full_model.features
        feature_dim = 576
        return backbone, feature_dim

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


class SpatialCorrelationModule(nn.Module):
    """Computes spatial correlation between piece features and puzzle feature map."""

    def __init__(
        self,
        feature_dim: int = 576,
        correlation_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize the spatial correlation module."""
        super().__init__()

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
        self.temperature = nn.Parameter(torch.ones(1) * (correlation_dim**0.5))

    def forward(self, piece_feat: torch.Tensor, puzzle_feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial correlation between piece and puzzle."""
        batch_size = piece_feat.shape[0]
        _, _, h, w = puzzle_feat_map.shape

        piece_proj = self.piece_proj(piece_feat)
        puzzle_proj = self.puzzle_proj(puzzle_feat_map)

        piece_proj = piece_proj.unsqueeze(-1).unsqueeze(-1)
        correlation = (piece_proj * puzzle_proj).sum(dim=1, keepdim=True)
        correlation = correlation / self.temperature

        correlation_flat = correlation.view(batch_size, -1)
        attention_flat = F.softmax(correlation_flat, dim=-1)
        attention_map = attention_flat.view(batch_size, 1, h, w)

        device = piece_feat.device
        y_coords = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device)
        x_coords = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        attention_squeezed = attention_map.squeeze(1)
        expected_x = (attention_squeezed * xx).sum(dim=[1, 2])
        expected_y = (attention_squeezed * yy).sum(dim=[1, 2])
        expected_position = torch.stack([expected_x, expected_y], dim=1)

        return attention_map, expected_position


class RotationCorrelationModule(nn.Module):
    """Rotation prediction via correlation between piece and puzzle."""

    def __init__(
        self,
        feature_dim: int = 576,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """Initialize the rotation correlation module."""
        super().__init__()
        self.feature_dim = feature_dim

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

        self.comparison_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def _rotate_feature_map(self, feat_map: torch.Tensor, rotation_idx: int) -> torch.Tensor:
        """Rotate a feature map by 0, 90, 180, or 270 degrees."""
        if rotation_idx == 0:
            return feat_map
        elif rotation_idx == 1:
            return torch.rot90(feat_map, k=-1, dims=[2, 3])
        elif rotation_idx == 2:
            return torch.rot90(feat_map, k=2, dims=[2, 3])
        else:
            return torch.rot90(feat_map, k=1, dims=[2, 3])

    def _extract_region(
        self,
        puzzle_feat_map: torch.Tensor,
        position: torch.Tensor,
        target_size: tuple[int, int],
        grid_size: int = 3,
    ) -> torch.Tensor:
        """Extract puzzle region at the predicted position using grid indexing.

        Args:
            puzzle_feat_map: Feature map of the puzzle [B, C, H, W].
            position: Predicted position [B, 2] with (x, y) in [0, 1].
            target_size: Target size (h, w) for the extracted region.
            grid_size: Grid size (3 for 3x3 grid).

        Returns:
            Extracted regions [B, C, h, w].
        """
        batch_size = puzzle_feat_map.shape[0]
        _, c, h_puzzle, w_puzzle = puzzle_feat_map.shape
        h_piece, w_piece = target_size

        # Compute grid cell indices from position
        # Position is in [0, 1], map to grid cell [0, grid_size-1]
        x_idx = torch.clamp((position[:, 0] * grid_size).long(), 0, grid_size - 1)
        y_idx = torch.clamp((position[:, 1] * grid_size).long(), 0, grid_size - 1)

        cell_h = h_puzzle // grid_size
        cell_w = w_puzzle // grid_size

        device = puzzle_feat_map.device
        extracted = torch.zeros(batch_size, c, h_piece, w_piece, device=device)

        for b in range(batch_size):
            h_start = int(y_idx[b].item()) * cell_h
            w_start = int(x_idx[b].item()) * cell_w
            h_end = h_start + cell_h
            w_end = w_start + cell_w

            region = puzzle_feat_map[b, :, h_start:h_end, w_start:w_end]

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
        """Compute rotation scores by comparing piece to puzzle."""
        _, _, h_piece, w_piece = piece_feat_map.shape

        puzzle_region = self._extract_region(puzzle_feat_map, position, (h_piece, w_piece))
        puzzle_proj = self.puzzle_proj(puzzle_region)

        rotation_scores = []
        for rot_idx in range(4):
            rotated_piece = self._rotate_feature_map(piece_feat_map, rot_idx)
            piece_proj = self.piece_proj(rotated_piece)
            combined = torch.cat([piece_proj, puzzle_proj], dim=1)
            score = self.comparison_net(combined)
            rotation_scores.append(score)

        rotation_logits = torch.cat(rotation_scores, dim=1)
        return rotation_logits


class FastBackboneModel(nn.Module):
    """Model with configurable fast backbone for experimentation.

    Supports multiple backbone architectures for speed comparison while
    maintaining the rotation correlation architecture from exp13.
    """

    def __init__(
        self,
        backbone_name: BackboneType = "repvgg_a0",
        pretrained: bool = True,
        correlation_dim: int = 128,
        rotation_hidden_dim: int = 128,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        rotation_dropout: float = 0.2,
    ):
        """Initialize the model with specified backbone.

        Args:
            backbone_name: Which backbone to use.
            pretrained: Whether to use pretrained weights.
            correlation_dim: Dimension for position correlation.
            rotation_hidden_dim: Hidden dimension for rotation comparison.
            freeze_backbone: If True, freeze backbone weights.
            dropout: Dropout rate for position head.
            rotation_dropout: Dropout rate for rotation head.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone_flag = freeze_backbone

        # Create backbone
        self.piece_backbone, self.feature_dim = get_backbone(backbone_name, pretrained)
        self.puzzle_backbone, _ = get_backbone(backbone_name, pretrained)

        # Global pooling
        self.piece_pool = nn.AdaptiveAvgPool2d(1)

        # Freeze if requested
        if freeze_backbone:
            self._freeze_backbones()

        # Spatial correlation for position
        self.spatial_correlation = SpatialCorrelationModule(
            feature_dim=self.feature_dim,
            correlation_dim=correlation_dim,
            dropout=dropout,
        )

        # Position refinement
        self.refinement = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        # Rotation correlation
        self.rotation_correlation = RotationCorrelationModule(
            feature_dim=self.feature_dim,
            hidden_dim=rotation_hidden_dim,
            dropout=rotation_dropout,
        )

        self.use_refinement = True

    def _freeze_backbones(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.piece_backbone.parameters():
            param.requires_grad = False
        for param in self.puzzle_backbone.parameters():
            param.requires_grad = False

    def _extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone, handling different output formats."""
        if self.backbone_name in ["repvgg_a0", "mobileone_s0"]:
            # timm features_only returns list of feature maps
            features = backbone(x)
            return features[-1]  # Last feature map
        else:
            # torchvision sequential returns tensor directly
            return backbone(x)

    def forward(self, piece: torch.Tensor, puzzle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Extract spatial features
        puzzle_feat_map = self._extract_features(self.puzzle_backbone, puzzle)
        piece_feat_map = self._extract_features(self.piece_backbone, piece)

        # Pool piece features for position correlation
        piece_feat = self.piece_pool(piece_feat_map).flatten(1)

        # Spatial correlation for position
        attention_map, expected_pos = self.spatial_correlation(piece_feat, puzzle_feat_map)

        # Position refinement
        if self.use_refinement:
            refinement = self.refinement(expected_pos)
            position = expected_pos + 0.1 * refinement
            position = torch.clamp(position, 0, 1)
        else:
            position = expected_pos

        # Rotation correlation
        rotation_logits = self.rotation_correlation(piece_feat_map, puzzle_feat_map, position)

        return position, rotation_logits, attention_map

    def predict_cell(self, piece: torch.Tensor, puzzle: torch.Tensor, grid_size: int = 3) -> torch.Tensor:
        """Predict which cell the piece belongs to in a grid.

        Args:
            piece: Piece image tensor.
            puzzle: Puzzle image tensor.
            grid_size: Size of the grid (3 for 3x3 grid).

        Returns:
            Cell indices (0 to grid_size*grid_size - 1).
        """
        position, _, _ = self.forward(piece, puzzle)
        cx, cy = position[:, 0], position[:, 1]
        # Map continuous position to grid cell index (row-major)
        col_idx = torch.clamp((cx * grid_size).long(), 0, grid_size - 1)
        row_idx = torch.clamp((cy * grid_size).long(), 0, grid_size - 1)
        cell = row_idx * grid_size + col_idx
        return cell

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """Get parameter groups with differential learning rates."""
        backbone_params = [p for p in self.piece_backbone.parameters() if p.requires_grad]
        backbone_params += [p for p in self.puzzle_backbone.parameters() if p.requires_grad]

        head_params = [p for p in self.spatial_correlation.parameters() if p.requires_grad]
        head_params += [p for p in self.refinement.parameters() if p.requires_grad]
        head_params += [p for p in self.rotation_correlation.parameters() if p.requires_grad]

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


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("Fast Backbone Model Test (Exp18)")
    print("=" * 60)

    batch_size = 4
    piece_size = 128
    puzzle_size = 256

    dummy_piece = torch.randn(batch_size, 3, piece_size, piece_size)
    dummy_puzzle = torch.randn(batch_size, 3, puzzle_size, puzzle_size)

    backbones: list[BackboneType] = [
        "repvgg_a0",
        "mobileone_s0",
        "shufflenet_v2_x0_5",
        "mobilenet_v3_small",
    ]

    for backbone_name in backbones:
        print(f"\n--- {backbone_name} ---")
        try:
            model = FastBackboneModel(backbone_name=backbone_name, pretrained=True)
            total_params = count_parameters(model, trainable_only=False)
            trainable_params = count_parameters(model, trainable_only=True)

            print(f"Feature dimension: {model.feature_dim}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

            # Test forward pass
            position, rotation_logits, attention = model(dummy_piece, dummy_puzzle)
            print(f"Position shape: {position.shape}")
            print(f"Rotation logits shape: {rotation_logits.shape}")
            print("Forward pass: OK")
        except Exception as e:
            print(f"ERROR: {e}")
