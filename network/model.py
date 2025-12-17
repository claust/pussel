#!/usr/bin/env python
"""CNN-based model for puzzle piece position and rotation prediction."""

import math
from typing import Dict, Tuple

import pytorch_lightning as pl
import timm  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SpatialCorrelationModule(nn.Module):
    """Computes spatial correlation between piece and puzzle feature maps.

    This module preserves spatial information from the puzzle backbone and computes
    a correlation map showing where the piece features match in the puzzle.
    The correlation map provides explicit spatial position hints.
    """

    def __init__(self, feature_dim: int, correlation_dim: int = 64):
        """Initialize the SpatialCorrelationModule.

        Args:
            feature_dim: Feature dimension from backbone (e.g., 512 for ResNet18)
            correlation_dim: Reduced dimension for correlation computation
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

        # Learnable temperature for scaled dot-product attention.
        # Initialized to sqrt(d) per standard attention: softmax(QK^T / sqrt(d))
        # Dividing by sqrt(d) prevents softmax from becoming too sharp.
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(correlation_dim))

        # Process correlation map to extract position features
        self.correlation_processor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.correlation_feat_dim = 64

    def forward(
        self, piece_feat_map: torch.Tensor, puzzle_feat_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial correlation between piece and puzzle.

        Args:
            piece_feat_map: Piece feature map (B, C, Hp, Wp)
            puzzle_feat_map: Puzzle feature map (B, C, H, W)

        Returns:
            Tuple of:
                - correlation_features: Processed correlation features (B, 64)
                - correlation_map: Raw correlation map (B, 1, H, W) for visualization
        """
        batch_size = piece_feat_map.shape[0]

        # Project to lower dimension
        piece_proj = self.piece_proj(piece_feat_map)  # (B, D, Hp, Wp)
        puzzle_proj = self.puzzle_proj(puzzle_feat_map)  # (B, D, H, W)

        # Global average pool piece features to get a single descriptor
        piece_vec = piece_proj.mean(dim=[2, 3])  # (B, D)

        # Compute correlation: how well does each puzzle location match the piece?
        # Reshape for batch matrix multiplication
        _, D, H, W = puzzle_proj.shape
        puzzle_flat = puzzle_proj.view(batch_size, D, H * W)  # (B, D, H*W)

        # Correlation scores
        correlation = torch.bmm(piece_vec.unsqueeze(1), puzzle_flat)  # (B, 1, H*W)
        correlation = correlation / self.temperature

        # Reshape to spatial map
        correlation_map = correlation.view(batch_size, 1, H, W)  # (B, 1, H, W)

        # Apply softmax to get attention-like weights
        correlation_softmax = F.softmax(
            correlation_map.view(batch_size, -1), dim=-1
        ).view(batch_size, 1, H, W)

        # Process correlation map to extract position-aware features
        correlation_features = self.correlation_processor(correlation_softmax)
        correlation_features = correlation_features.view(batch_size, -1)  # (B, 64)

        return correlation_features, correlation_map


class CrossAttentionFusion(nn.Module):
    """Cross-attention inspired fusion for globally-pooled features.

    Uses piece features to compute attention weights that modulate puzzle features,
    learning which aspects of the puzzle context are relevant for the piece.
    """

    def __init__(self, dim: int, output_dim: int = 512):
        """Initialize the CrossAttentionFusion module.

        Args:
            dim: Feature dimension from backbones
            output_dim: Desired output dimension
        """
        super().__init__()

        # Attention computation: use piece to attend over puzzle
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),  # Combine piece and puzzle context
            nn.Tanh(),
            nn.Linear(dim, dim),  # Generate attention logits
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

    def forward(
        self, piece_feat: torch.Tensor, puzzle_feat: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with attention-based fusion.

        Args:
            piece_feat: Piece features (batch_size, dim)
            puzzle_feat: Puzzle features (batch_size, dim)

        Returns:
            Fused features (batch_size, output_dim)
        """
        # Normalize inputs
        piece_feat = self.norm_piece(piece_feat)
        puzzle_feat = self.norm_puzzle(puzzle_feat)

        # Project features
        piece_proj = self.piece_proj(piece_feat)
        puzzle_proj = self.puzzle_proj(puzzle_feat)

        # Compute attention weights: use piece context to gate puzzle features
        combined = torch.cat([piece_proj, puzzle_proj], dim=1)
        attn_logits = self.attention(combined)  # (batch_size, dim)
        attn_weights = torch.sigmoid(attn_logits)  # Gate values in [0, 1]

        # Apply attention to puzzle features (element-wise gating)
        attended_puzzle = attn_weights * puzzle_proj

        # Concatenate piece with attended puzzle
        fused = torch.cat([piece_proj, attended_puzzle], dim=1)

        # Final projection
        output = self.fusion(fused)

        return output


class PuzzleCNN(pl.LightningModule):
    """LightningModule for puzzle piece position and rotation prediction.

    This model uses a dual-backbone architecture with spatial correlation:
    - Piece backbone: Extracts features from puzzle pieces (with GAP for fusion)
    - Puzzle spatial backbone: Preserves spatial feature maps for correlation
    - Spatial correlation: Computes where piece features match in the puzzle
    - Cross-attention fusion: Combines piece and puzzle context features
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        use_spatial_correlation: bool = True,
    ):
        """Initialize the PuzzleCNN model.

        Args:
            backbone_name: Name of the timm backbone to use
            pretrained: Whether to use pretrained weights for the backbone
            learning_rate: Initial learning rate
            position_weight: Weight for position loss (α)
            rotation_weight: Weight for rotation loss (β)
            use_spatial_correlation: Whether to use spatial correlation module
        """
        super().__init__()
        self.save_hyperparameters()
        self.use_spatial_correlation = use_spatial_correlation

        # Single shared backbone with features_only=True for efficiency
        # Returns spatial feature maps; we apply GAP manually for global features
        # This avoids redundant forward passes (2 instead of 4 when spatial enabled)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],  # Final feature map (works across backbone types)
        )

        # Global average pooling to convert spatial maps to global features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            spatial_features = self.backbone(dummy_input)[-1]
            self.feature_dim = spatial_features.shape[1]

        # Spatial correlation module for position-aware features
        if use_spatial_correlation:
            self.spatial_correlation = SpatialCorrelationModule(
                feature_dim=self.feature_dim,
                correlation_dim=64,
            )
            spatial_feat_dim = self.spatial_correlation.correlation_feat_dim
        else:
            spatial_feat_dim = 0

        # Cross-attention fusion with learned piece-puzzle interactions
        self.fusion_layer = CrossAttentionFusion(
            dim=self.feature_dim,
            output_dim=512,
        )

        # Combined feature dimension: fusion features + spatial correlation features
        self.fused_dim = 512 + spatial_feat_dim

        # Position prediction head (x1, y1, x2, y2)
        # Spatial correlation features are particularly helpful here
        self.position_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # Bound outputs between 0 and 1 for normalized coordinates
        )

        # Rotation prediction head (4 classes: 0°, 90°, 180°, 270°)
        self.rotation_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),  # No activation (will use CrossEntropyLoss)
        )

        # Loss weights
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight

        # Initialize metrics
        self.train_pos_loss = 0.0
        self.train_rot_loss = 0.0
        self.val_pos_loss = 0.0
        self.val_rot_loss = 0.0
        self.val_rot_acc = 0.0

        # Initialize additional metrics (confusion matrix and IoU)
        self.val_confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=4
        )

    @staticmethod
    def calculate_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between predicted and ground truth boxes.

        Calculate the intersection over union between prediction and ground truth boxes.

        Args:
            pred_boxes: Predicted boxes with normalized coordinates (x1, y1, x2, y2)
            gt_boxes: Ground truth boxes with normalized coordinates (x1, y1, x2, y2)

        Returns:
            IoU values for each box pair, shape (batch_size,)
        """
        # Extract coordinates for pred_boxes
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]

        # Extract coordinates for gt_boxes
        gt_x1 = gt_boxes[:, 0]
        gt_y1 = gt_boxes[:, 1]
        gt_x2 = gt_boxes[:, 2]
        gt_y2 = gt_boxes[:, 3]

        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

        # Calculate intersection coordinates
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)

        # Calculate intersection area
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_width * inter_height

        # Calculate union area
        union = pred_area + gt_area - intersection

        # Calculate IoU
        # Add small epsilon to avoid division by zero
        iou = intersection / (union + 1e-6)

        return iou

    def forward(
        self, piece_img: torch.Tensor, puzzle_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            piece_img: Puzzle piece tensor of shape (batch_size, 3, height, width)
            puzzle_img: Full puzzle tensor of shape (batch_size, 3, height, width)

        Returns:
            Tuple of (position_pred, rotation_logits)
        """
        # Extract spatial feature maps from shared backbone (2 forward passes total)
        piece_spatial_maps = self.backbone(piece_img)[-1]
        puzzle_spatial_maps = self.backbone(puzzle_img)[-1]

        # Apply GAP to get global features for cross-attention fusion
        piece_features = self.global_pool(piece_spatial_maps).flatten(1)
        puzzle_features = self.global_pool(puzzle_spatial_maps).flatten(1)

        # Cross-attention fusion: piece queries attend to puzzle context
        fused_features = self.fusion_layer(piece_features, puzzle_features)

        # Add spatial correlation features if enabled
        if self.use_spatial_correlation:
            # Reuse spatial maps from above (no extra forward passes)
            correlation_features, _ = self.spatial_correlation(
                piece_spatial_maps, puzzle_spatial_maps
            )

            # Concatenate spatial correlation features with fusion features
            fused_features = torch.cat([fused_features, correlation_features], dim=1)

        # Get position and rotation predictions
        position_pred = self.position_head(fused_features)
        rotation_logits = self.rotation_head(fused_features)

        return position_pred, rotation_logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch of data
            batch_idx: Index of batch

        Returns:
            Loss tensor
        """
        # Get predictions using both inputs
        position_pred, rotation_logits = self(batch["piece"], batch["puzzle"])

        # Calculate position loss (MSE)
        position_loss = F.mse_loss(position_pred, batch["position"])

        # Calculate rotation loss (CrossEntropy)
        rotation_loss = F.cross_entropy(rotation_logits, batch["rotation"])

        # Calculate total loss
        total_loss = (
            self.position_weight * position_loss + self.rotation_weight * rotation_loss
        )

        # Log metrics
        batch_size = batch["piece"].size(0)
        self.log(
            "train/position_loss", position_loss, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "train/rotation_loss", rotation_loss, prog_bar=True, batch_size=batch_size
        )
        self.log("train/total_loss", total_loss, prog_bar=True, batch_size=batch_size)

        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Batch of data
            batch_idx: Index of batch

        Returns:
            Dictionary of validation metrics
        """
        # Get predictions
        position_pred, rotation_logits = self(batch["piece"], batch["puzzle"])

        # Calculate position loss (MSE)
        position_loss = F.mse_loss(position_pred, batch["position"])

        # Calculate IoU between predicted and ground truth boxes
        iou = self.calculate_iou(position_pred, batch["position"])
        mean_iou = torch.mean(iou)

        # Calculate rotation loss and accuracy
        rotation_loss = F.cross_entropy(rotation_logits, batch["rotation"])
        rotation_preds = torch.argmax(rotation_logits, dim=1)
        rotation_acc = torch.sum(rotation_preds == batch["rotation"]).float() / len(
            batch["rotation"]
        )

        # Update confusion matrix
        self.val_confusion(rotation_preds, batch["rotation"])

        # Calculate total loss
        total_loss = (
            self.position_weight * position_loss + self.rotation_weight * rotation_loss
        )

        # Log metrics
        batch_size = batch["piece"].size(0)
        self.log(
            "val/position_loss", position_loss, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "val/rotation_loss", rotation_loss, prog_bar=True, batch_size=batch_size
        )
        self.log("val/total_loss", total_loss, prog_bar=True, batch_size=batch_size)
        self.log("val/iou", mean_iou, prog_bar=True, batch_size=batch_size)
        self.log("val/rotation_acc", rotation_acc, prog_bar=True, batch_size=batch_size)

        return {
            "val_loss": total_loss,
            "val_pos_loss": position_loss,
            "val_rot_loss": rotation_loss,
            "val_iou": mean_iou,
            "val_rot_acc": rotation_acc,
        }

    def predict_piece(
        self, piece_img: torch.Tensor, puzzle_img: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Make prediction for a single puzzle piece with puzzle context.

        Args:
            piece_img: Preprocessed puzzle piece tensor of shape (1, 3, H, W)
            puzzle_img: Preprocessed full puzzle tensor of shape (1, 3, H, W)

        Returns:
            Tuple of (position, rotation_class, rotation_probabilities)
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with both inputs
            position_pred, rotation_logits = self(piece_img, puzzle_img)

            # Get rotation class and probabilities
            rotation_probs = F.softmax(rotation_logits, dim=1)
            # Ensure rotation_class is an int
            rotation_class = int(torch.argmax(rotation_logits, dim=1).item())

            return position_pred[0], rotation_class, rotation_probs[0]

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary containing optimizer and lr_scheduler
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
