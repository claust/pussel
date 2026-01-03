"""Model for exp21: Masked rotation correlation.

Key change from exp20: The rotation correlation module uses masks to compare
only the puzzle content regions, ignoring black background.

Architecture:
- Same dual-backbone (ShuffleNetV2_x0.5) as exp20
- Same spatial correlation for position prediction
- Modified rotation correlation that applies masks during feature comparison
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ShuffleNet_V2_X0_5_Weights

BackboneType = Literal["shufflenet_v2_x0_5", "mobilenet_v3_small"]


def get_backbone(backbone_name: BackboneType, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Create a backbone network and return it with its feature dimension."""
    if backbone_name == "shufflenet_v2_x0_5":
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = models.shufflenet_v2_x0_5(weights=weights)
        backbone = nn.Sequential(
            full_model.conv1,
            full_model.maxpool,
            full_model.stage2,
            full_model.stage3,
            full_model.stage4,
            full_model.conv5,
        )
        feature_dim = 1024
        return backbone, feature_dim

    elif backbone_name == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = models.mobilenet_v3_small(weights=weights)
        backbone = full_model.features
        feature_dim = 576
        return backbone, feature_dim

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


class SpatialCorrelationModule(nn.Module):
    """Spatial correlation for position prediction (unchanged from exp20)."""

    def __init__(
        self,
        feature_dim: int = 1024,
        correlation_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize spatial correlation module."""
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


class MaskedRotationCorrelationModule(nn.Module):
    """Rotation prediction using masked correlation.

    Key change from exp20: When comparing rotated piece features to puzzle
    region features, we apply the mask to ignore black background regions.

    The mask is resized to match feature map spatial dimensions and applied
    during the comparison step.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """Initialize masked rotation correlation module."""
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Project piece and puzzle features to comparison space
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

        # Comparison network - takes masked difference/product features
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

    def _rotate_tensor(self, tensor: torch.Tensor, rotation_idx: int) -> torch.Tensor:
        """Rotate a 4D tensor (B, C, H, W) by 0, 90, 180, or 270 degrees."""
        if rotation_idx == 0:
            return tensor
        elif rotation_idx == 1:  # 90 degrees clockwise
            return torch.rot90(tensor, k=-1, dims=[2, 3])
        elif rotation_idx == 2:  # 180 degrees
            return torch.rot90(tensor, k=2, dims=[2, 3])
        else:  # 270 degrees clockwise (= 90 counter-clockwise)
            return torch.rot90(tensor, k=1, dims=[2, 3])

    def _extract_region(
        self,
        puzzle_feat_map: torch.Tensor,
        position: torch.Tensor,
        target_size: tuple[int, int],
        grid_size: int = 4,
    ) -> torch.Tensor:
        """Extract puzzle region at predicted position using grid indexing."""
        batch_size = puzzle_feat_map.shape[0]
        _, c, h_puzzle, w_puzzle = puzzle_feat_map.shape
        h_piece, w_piece = target_size

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
        piece_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rotation scores using masked correlation.

        Args:
            piece_feat_map: Piece feature map [B, C, H, W].
            puzzle_feat_map: Puzzle feature map [B, C, H', W'].
            position: Predicted position [B, 2].
            piece_mask: Binary mask [B, 1, H_input, W_input] where 1 = content.

        Returns:
            Rotation logits [B, 4].
        """
        batch_size, _, h_piece, w_piece = piece_feat_map.shape

        # Extract puzzle region at predicted position
        puzzle_region = self._extract_region(puzzle_feat_map, position, (h_piece, w_piece))
        puzzle_proj = self.puzzle_proj(puzzle_region)

        # Resize mask to feature map size
        mask_resized = F.interpolate(
            piece_mask,
            size=(h_piece, w_piece),
            mode="bilinear",
            align_corners=True,
        )
        # Binarize after interpolation
        mask_resized = (mask_resized > 0.5).float()

        rotation_scores = []

        for rot_idx in range(4):
            # Rotate piece features and mask
            rotated_piece = self._rotate_tensor(piece_feat_map, rot_idx)
            rotated_mask = self._rotate_tensor(mask_resized, rot_idx)

            # Project piece features
            piece_proj = self.piece_proj(rotated_piece)

            # Apply mask to both piece and puzzle projections
            # This zeros out the background regions in the comparison
            masked_piece = piece_proj * rotated_mask
            masked_puzzle = puzzle_proj * rotated_mask

            # Combine for comparison (concatenate masked features)
            combined = torch.cat([masked_piece, masked_puzzle], dim=1)

            # Compute comparison score
            score = self.comparison_net(combined)
            rotation_scores.append(score)

        rotation_logits = torch.cat(rotation_scores, dim=1)
        return rotation_logits


class MaskedRotationModel(nn.Module):
    """Model with masked rotation correlation for exp21.

    Same architecture as exp20's FastBackboneModel but with:
    - MaskedRotationCorrelationModule instead of RotationCorrelationModule
    - Forward pass accepts mask tensor
    """

    def __init__(
        self,
        backbone_name: BackboneType = "shufflenet_v2_x0_5",
        pretrained: bool = True,
        correlation_dim: int = 128,
        rotation_hidden_dim: int = 128,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        rotation_dropout: float = 0.2,
    ):
        """Initialize masked rotation model."""
        super().__init__()
        self.backbone_name = backbone_name

        # Dual backbones
        self.piece_backbone, self.feature_dim = get_backbone(backbone_name, pretrained)
        self.puzzle_backbone, _ = get_backbone(backbone_name, pretrained)

        # Global pooling for piece
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

        # Masked rotation correlation (key change!)
        self.rotation_correlation = MaskedRotationCorrelationModule(
            feature_dim=self.feature_dim,
            hidden_dim=rotation_hidden_dim,
            dropout=rotation_dropout,
        )

        self.use_refinement = True

    def _freeze_backbones(self) -> None:
        """Freeze backbone parameters."""
        for param in self.piece_backbone.parameters():
            param.requires_grad = False
        for param in self.puzzle_backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        piece: torch.Tensor,
        puzzle: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with mask.

        Args:
            piece: Piece image [B, 3, H, W].
            puzzle: Puzzle image [B, 3, H, W].
            mask: Binary mask [B, 1, H, W] where 1 = puzzle content.

        Returns:
            Tuple of (position, rotation_logits, attention_map).
        """
        # Extract features
        puzzle_feat_map = self.puzzle_backbone(puzzle)
        piece_feat_map = self.piece_backbone(piece)

        # Pool piece features for position
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

        # Masked rotation correlation (key change - pass mask!)
        rotation_logits = self.rotation_correlation(
            piece_feat_map,
            puzzle_feat_map,
            position,
            mask,
        )

        return position, rotation_logits, attention_map

    def predict_cell(
        self, piece: torch.Tensor, puzzle: torch.Tensor, mask: torch.Tensor, grid_size: int = 4
    ) -> torch.Tensor:
        """Predict cell index."""
        position, _, _ = self.forward(piece, puzzle, mask)
        cx, cy = position[:, 0], position[:, 1]
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
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("Masked Rotation Model Test (Exp21)")
    print("=" * 60)

    batch_size = 4
    piece_size = 128
    puzzle_size = 256

    # Create dummy inputs
    dummy_piece = torch.randn(batch_size, 3, piece_size, piece_size)
    dummy_puzzle = torch.randn(batch_size, 3, puzzle_size, puzzle_size)

    # Create dummy mask (simulate ~70% content coverage)
    dummy_mask = torch.ones(batch_size, 1, piece_size, piece_size)
    dummy_mask[:, :, :20, :] = 0  # Top edge
    dummy_mask[:, :, -20:, :] = 0  # Bottom edge
    dummy_mask[:, :, :, :20] = 0  # Left edge
    dummy_mask[:, :, :, -20:] = 0  # Right edge

    print(f"\nMask coverage: {dummy_mask.mean().item():.1%}")

    # Test model
    model = MaskedRotationModel(backbone_name="shufflenet_v2_x0_5", pretrained=True)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"\nFeature dimension: {model.feature_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Forward pass
    position, rotation_logits, attention = model(dummy_piece, dummy_puzzle, dummy_mask)

    print(f"\nPosition shape: {position.shape}")
    print(f"Rotation logits shape: {rotation_logits.shape}")
    print(f"Attention shape: {attention.shape}")

    # Test cell prediction
    cells = model.predict_cell(dummy_piece, dummy_puzzle, dummy_mask, grid_size=4)
    print(f"Predicted cells: {cells.tolist()}")

    print("\nForward pass: OK")
