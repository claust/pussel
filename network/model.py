#!/usr/bin/env python
"""CNN-based model for puzzle piece position and rotation prediction."""

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
    """LightningModule for puzzle piece position and rotation prediction."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ):
        """Initialize the PuzzleCNN model.

        Args:
            backbone_name: Name of the timm backbone to use
            pretrained: Whether to use pretrained weights for the backbone
            learning_rate: Initial learning rate
            position_weight: Weight for position loss (α)
            rotation_weight: Weight for rotation loss (β)
        """
        super().__init__()
        self.save_hyperparameters()

        # Create separate backbones for piece and puzzle
        self.piece_backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.puzzle_backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

        # Get feature dimensions from backbones
        # Note: Input size is arbitrary here - only used to probe feature dimensions
        # Actual input sizes are determined by dataset transforms (config.py)
        # Backbones use adaptive pooling, so feature dims are size-agnostic
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            piece_features = self.piece_backbone(dummy_input)
            puzzle_features = self.puzzle_backbone(dummy_input)
            self.piece_feature_dim = piece_features.shape[1]
            self.puzzle_feature_dim = puzzle_features.shape[1]

        # Cross-attention fusion with learned piece-puzzle interactions
        self.fusion_layer = CrossAttentionFusion(
            dim=self.piece_feature_dim,
            output_dim=512,
        )
        self.fused_dim = 512

        # Position prediction head (x1, y1, x2, y2)
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
        # Extract features from backbones
        piece_features = self.piece_backbone(piece_img)
        puzzle_features = self.puzzle_backbone(puzzle_img)

        # Cross-attention fusion: piece queries attend to puzzle context
        fused_features = self.fusion_layer(piece_features, puzzle_features)

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
