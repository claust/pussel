"""FastBackboneModel from exp18 for 3x3 grid puzzle piece matching.

This model predicts both position (which of 9 cells) and rotation of a puzzle piece
by comparing the piece features with puzzle features using ShuffleNetV2_x0.5 backbone.

Adapted from network/experiments/exp18_3x3_20k_puzzles/model.py for backend inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ShuffleNet_V2_X0_5_Weights


class SpatialCorrelationModule(nn.Module):
    """Computes spatial correlation between piece features and puzzle feature map."""

    def __init__(
        self,
        feature_dim: int = 1024,
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
        """Compute spatial correlation between piece and puzzle.

        Args:
            piece_feat: Piece feature vector (B, feature_dim).
            puzzle_feat_map: Puzzle spatial features (B, feature_dim, H, W).

        Returns:
            Tuple of:
                - attention_map: Softmax attention over puzzle locations (B, 1, H, W)
                - expected_position: Weighted average position (B, 2)
        """
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
    """Rotation prediction via correlation between piece and puzzle.

    For each rotation r in [0, 90, 180, 270]:
        1. Rotate the piece feature map by r degrees
        2. Extract the puzzle region at the predicted position
        3. Compute similarity between rotated piece and puzzle region
        4. The rotation with highest similarity is the prediction
    """

    def __init__(
        self,
        feature_dim: int = 1024,
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
        """Rotate a feature map by 0, 90, 180, or 270 degrees.

        Args:
            feat_map: Feature map of shape (B, C, H, W).
            rotation_idx: Rotation index (0=0deg, 1=90deg, 2=180deg, 3=270deg).

        Returns:
            Rotated feature map of shape (B, C, H, W).
        """
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

        For 3x3 grid, this extracts the cell at the predicted position.

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
        """Compute rotation scores by comparing piece to puzzle.

        Args:
            piece_feat_map: Piece spatial features (B, C, H_piece, W_piece).
            puzzle_feat_map: Puzzle spatial features (B, C, H_puzzle, W_puzzle).
            position: Predicted position (B, 2) in [0, 1] coords.

        Returns:
            Rotation logits (B, 4) for rotations [0, 90, 180, 270].
        """
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
    """Model with ShuffleNetV2_x0.5 backbone for 3x3 grid puzzle solving.

    This is the production model from exp18 with 82.2% cell accuracy and
    95.1% rotation accuracy on 3x3 grids.
    """

    def __init__(
        self,
        pretrained: bool = True,
        correlation_dim: int = 128,
        rotation_hidden_dim: int = 128,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        rotation_dropout: float = 0.2,
    ):
        """Initialize the model with ShuffleNetV2_x0.5 backbone.

        Args:
            pretrained: Whether to use pretrained weights.
            correlation_dim: Dimension for position correlation.
            rotation_hidden_dim: Hidden dimension for rotation comparison.
            freeze_backbone: If True, freeze backbone weights.
            dropout: Dropout rate for position head.
            rotation_dropout: Dropout rate for rotation head.
        """
        super().__init__()
        self.freeze_backbone_flag = freeze_backbone
        self.feature_dim = 1024  # ShuffleNetV2 x0.5 final conv outputs 1024 channels

        # Create ShuffleNetV2 backbones
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        piece_model = models.shufflenet_v2_x0_5(weights=weights)
        puzzle_model = models.shufflenet_v2_x0_5(weights=weights)

        # Extract feature extractor (everything before final FC)
        self.piece_backbone = nn.Sequential(
            piece_model.conv1,
            piece_model.maxpool,
            piece_model.stage2,
            piece_model.stage3,
            piece_model.stage4,
            piece_model.conv5,
        )
        self.puzzle_backbone = nn.Sequential(
            puzzle_model.conv1,
            puzzle_model.maxpool,
            puzzle_model.stage2,
            puzzle_model.stage3,
            puzzle_model.stage4,
            puzzle_model.conv5,
        )

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

    def forward(self, piece: torch.Tensor, puzzle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            piece: Piece image tensor of shape (batch, 3, 128, 128).
            puzzle: Puzzle image tensor of shape (batch, 3, 256, 256).

        Returns:
            Tuple of:
                - position: Predicted (cx, cy) coordinates (batch, 2) in [0, 1]
                - rotation_logits: Rotation class logits (batch, 4)
                - attention_map: Correlation map over puzzle (batch, 1, H, W)
        """
        # Extract spatial features
        puzzle_feat_map = self.puzzle_backbone(puzzle)
        piece_feat_map = self.piece_backbone(piece)

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
        col_idx = torch.clamp((cx * grid_size).long(), 0, grid_size - 1)
        row_idx = torch.clamp((cy * grid_size).long(), 0, grid_size - 1)
        cell = row_idx * grid_size + col_idx
        return cell
