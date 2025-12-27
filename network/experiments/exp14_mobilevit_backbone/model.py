"""Dual-input regressor with ROTATION CORRELATION using MobileViT-XS backbone.

This is Experiment 14: Testing MobileViT-XS as an alternative backbone.

MobileViT-XS is a hybrid CNN-Transformer architecture that:
- Combines CNNs (great for edges/shapes) with Transformers (great for global context)
- May be better at understanding piece-puzzle relationships due to attention mechanism
- Has 2.3M parameters (vs 2.5M for MobileNetV3-Small)
- Has 384-dim features (vs 576-dim for MobileNetV3-Small)

The hypothesis: A Transformer-based backbone might naturally be better at
understanding the correlation between piece and puzzle regions.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCorrelationModule(nn.Module):
    """Computes spatial correlation between piece features and puzzle feature map.

    This module finds WHERE in the puzzle the piece features best match,
    producing a correlation/attention map over puzzle locations.
    """

    def __init__(
        self,
        feature_dim: int = 384,  # MobileViT-XS uses 384-dim features
        correlation_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize the spatial correlation module.

        Args:
            feature_dim: Feature dimension from backbones (384 for MobileViT-XS).
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

    This is THE KEY FIX for exp10/11: Instead of predicting rotation from
    piece features alone, we COMPARE the piece to the puzzle for each rotation.

    For each rotation r in [0, 90, 180, 270]:
        1. Rotate the piece feature map by r degrees
        2. Extract the puzzle region at the predicted position
        3. Compute similarity between rotated piece and puzzle region
        4. The rotation with highest similarity is the prediction

    This mirrors how a human would solve the puzzle: try different orientations
    and pick the one that matches best.
    """

    def __init__(
        self,
        feature_dim: int = 384,  # MobileViT-XS uses 384-dim features
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """Initialize the rotation correlation module.

        Args:
            feature_dim: Feature dimension from backbones (384 for MobileViT-XS).
            hidden_dim: Hidden dimension for comparison network.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable projection for piece and puzzle features before comparison
        # This allows the model to learn what aspects matter for rotation matching
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

        The puzzle feature map is 8x8 for 256x256 input, and each quadrant is 4x4.
        We determine which quadrant based on the position and extract that region.

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

        # For 2x2 quadrant prediction:
        # - Puzzle feature map is 8x8
        # - Each quadrant is 4x4
        # - Position (0.25, 0.25) -> top-left [0:4, 0:4]
        # - Position (0.75, 0.25) -> top-right [0:4, 4:8]
        # - Position (0.25, 0.75) -> bottom-left [4:8, 0:4]
        # - Position (0.75, 0.75) -> bottom-right [4:8, 4:8]

        # Determine quadrant indices from position
        # x >= 0.5 -> right half, y >= 0.5 -> bottom half
        x_idx = (position[:, 0] >= 0.5).long()  # 0 for left, 1 for right
        y_idx = (position[:, 1] >= 0.5).long()  # 0 for top, 1 for bottom

        # Calculate start indices for slicing
        half_h = h_puzzle // 2
        half_w = w_puzzle // 2

        # For each sample in the batch, we need to extract different regions
        # Create output tensor
        device = puzzle_feat_map.device
        extracted = torch.zeros(batch_size, c, h_piece, w_piece, device=device)

        # Extract regions for each sample
        # We do this in a loop but it's only 4 possible combinations
        for b in range(batch_size):
            h_start = y_idx[b].item() * half_h
            w_start = x_idx[b].item() * half_w
            h_end = h_start + half_h
            w_end = w_start + half_w

            region = puzzle_feat_map[b, :, h_start:h_end, w_start:w_end]

            # Resize to target size if needed (should be same for 2x2 quadrant)
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
        puzzle_region = self._extract_region(puzzle_feat_map, position, (h_piece, w_piece))  # (B, C, H_piece, W_piece)

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
    """Dual-input model with position AND rotation CORRELATION prediction.

    Experiment 14: Uses MobileViT-XS backbone instead of MobileNetV3-Small.

    MobileViT-XS is a hybrid CNN-Transformer architecture that may better
    capture piece-puzzle relationships through its attention mechanism.
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
            freeze_backbone: If True, freeze backbone weights.
            dropout: Dropout rate for position head.
            rotation_dropout: Dropout rate for rotation head.
        """
        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Load pretrained MobileViT-XS from timm
        # Using features_only=True to get spatial feature maps
        self.piece_features = timm.create_model(
            "mobilevit_xs",
            pretrained=True,
            features_only=True,
            out_indices=[4],  # Only get final stage (384 channels, 8x8 for 256x256 input)
        )
        self.puzzle_features = timm.create_model(
            "mobilevit_xs",
            pretrained=True,
            features_only=True,
            out_indices=[4],  # Only get final stage
        )

        # MobileViT-XS feature dimension
        feature_dim = 384

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

        # KEY FIX: ROTATION CORRELATION module
        # Compares piece (at each rotation) to puzzle region
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
        # timm features_only returns a list, we take the first (and only) element
        puzzle_feat_map = self.puzzle_features(puzzle)[0]  # (B, 384, H, W)

        # Extract SPATIAL features from piece (keep spatial dimensions!)
        piece_feat_map = self.piece_features(piece)[0]  # (B, 384, 4, 4)

        # Pool piece features for position correlation only
        piece_feat = self.piece_pool(piece_feat_map).flatten(1)  # (B, 384)

        # Compute spatial correlation for position
        attention_map, expected_pos = self.spatial_correlation(piece_feat, puzzle_feat_map)

        # Optional position refinement
        if self.use_refinement:
            refinement = self.refinement(expected_pos)
            position = expected_pos + 0.1 * refinement  # Small adjustment
            position = torch.clamp(position, 0, 1)
        else:
            position = expected_pos

        # KEY FIX: Rotation prediction by COMPARING piece to puzzle
        # This uses the predicted position to extract the puzzle region,
        # then compares the piece at each rotation to find the best match
        rotation_logits = self.rotation_correlation(piece_feat_map, puzzle_feat_map, position)

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
        # MobileViT-XS has 6 components: stem + stages_0-3 + final_conv
        self._unfrozen_layers = set(range(6))

    def unfreeze_layers(self, layer_indices: list[int]) -> None:
        """Unfreeze specific stages by index for gradual unfreezing.

        MobileViT-XS has 6 components: stem + stages_0-3 + final_conv.

        Args:
            layer_indices: List of component indices to unfreeze (0-5).
        """
        # MobileViT component prefixes (timm uses underscores)
        component_prefixes = ["stem", "stages_0", "stages_1", "stages_2", "stages_3", "final_conv"]

        for idx in layer_indices:
            if idx < len(component_prefixes):
                prefix = component_prefixes[idx]
                for name, param in self.piece_features.named_parameters():
                    if name.startswith(prefix):
                        param.requires_grad = True
                for name, param in self.puzzle_features.named_parameters():
                    if name.startswith(prefix):
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

        # Head parameters (correlation modules, refinement)
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

    def get_num_backbone_layers(self) -> int:
        """Get the number of components in the backbone (MobileViT-XS has 6)."""
        return 6  # MobileViT-XS: stem + stages_0-3 + final_conv

    def get_layer_info(self) -> list[dict]:
        """Get information about each backbone component.

        Returns:
            List of dicts with component info (index, name, param count, trainable).
        """
        # MobileViT component prefixes (timm uses underscores)
        component_prefixes = ["stem", "stages_0", "stages_1", "stages_2", "stages_3", "final_conv"]
        info = []

        for idx, prefix in enumerate(component_prefixes):
            # Count params in this component
            param_count = 0
            trainable = False
            for name, param in self.piece_features.named_parameters():
                if name.startswith(prefix):
                    param_count += param.numel()
                    if param.requires_grad:
                        trainable = True

            info.append(
                {
                    "index": idx,
                    "name": prefix,
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
    print("Model Architecture Test - MobileViT-XS Backbone")
    print("=" * 60)

    batch_size = 4
    piece_size = 128
    puzzle_size = 256

    dummy_piece = torch.randn(batch_size, 3, piece_size, piece_size)
    dummy_puzzle = torch.randn(batch_size, 3, puzzle_size, puzzle_size)

    # Test unfrozen backbone
    print("\n--- Unfrozen Backbone (Full Fine-tuning) ---")
    model = DualInputRegressorWithRotationCorrelation(freeze_backbone=False)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Count rotation correlation parameters specifically
    rotation_params = sum(p.numel() for p in model.rotation_correlation.parameters())
    print(f"Rotation correlation module parameters: {rotation_params:,}")

    # Compare with exp13's MobileNetV3-Small total params
    exp13_total_params = 5_268_902  # From exp13
    print(f"Exp13 (MobileNetV3) total parameters: {exp13_total_params:,}")
    print(f"Parameter difference: {total_params - exp13_total_params:,}")

    # Test parameter groups
    print("\n--- Parameter Groups for Differential LR ---")
    param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        print(f"  {group['name']}: {param_count:,} params, LR={group['lr']}")

    # Show layer info
    print("\n--- Backbone Stage Info ---")
    for layer in model.get_layer_info():
        status = "TRAINABLE" if layer["trainable"] else "frozen"
        print(f"  Stage {layer['index']} ({layer['name']}): {layer['params']:,} params [{status}]")

    # Forward pass test
    print("\n--- Forward Pass ---")
    position, rotation_logits, attention_map = model(dummy_piece, dummy_puzzle)
    print(f"Position output shape: {position.shape}")
    print(f"Rotation logits shape: {rotation_logits.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Position values: {position[0].tolist()}")
    print(f"Rotation logits: {rotation_logits[0].tolist()}")
    print(f"Predicted rotation: {rotation_logits.argmax(dim=1).tolist()}")

    # Test RotationCorrelationModule in isolation
    print("\n--- RotationCorrelationModule Isolation Test ---")
    rot_corr = RotationCorrelationModule(feature_dim=384, hidden_dim=128)
    dummy_piece_feat = torch.randn(batch_size, 384, 4, 4)
    dummy_puzzle_feat = torch.randn(batch_size, 384, 8, 8)
    dummy_position = torch.rand(batch_size, 2)
    rotation_out = rot_corr(dummy_piece_feat, dummy_puzzle_feat, dummy_position)
    print(f"Piece features shape: {dummy_piece_feat.shape}")
    print(f"Puzzle features shape: {dummy_puzzle_feat.shape}")
    print(f"Position shape: {dummy_position.shape}")
    print(f"Rotation output shape: {rotation_out.shape}")
