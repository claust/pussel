#!/usr/bin/env python
"""CNN-based model for puzzle piece position and rotation prediction."""

from typing import Dict, Tuple

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

        # Create model backbone using timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        # Position prediction head (x1, y1, x2, y2)
        self.position_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # Bound outputs between 0 and 1 for normalized coordinates
        )

        # Rotation prediction head (4 classes: 0°, 90°, 180°, 270°)
        self.rotation_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Tuple of (position_pred, rotation_pred)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Get position and rotation predictions
        position_pred = self.position_head(features)
        rotation_logits = self.rotation_head(features)

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
        # Get predictions
        position_pred, rotation_logits = self(batch["piece"])

        # Calculate position loss (MSE)
        position_loss = F.mse_loss(position_pred, batch["position"])

        # Calculate rotation loss (CrossEntropy)
        rotation_loss = F.cross_entropy(rotation_logits, batch["rotation"])

        # Calculate total loss
        total_loss = (
            self.position_weight * position_loss + self.rotation_weight * rotation_loss
        )

        # Log metrics
        self.log("train/position_loss", position_loss, prog_bar=True)
        self.log("train/rotation_loss", rotation_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Batch of data
            batch_idx: Index of batch

        Returns:
            Dict of metrics
        """
        # Get predictions
        position_pred, rotation_logits = self(batch["piece"])

        # Calculate position loss (MSE)
        position_loss = F.mse_loss(position_pred, batch["position"])

        # Calculate rotation loss and accuracy
        rotation_loss = F.cross_entropy(rotation_logits, batch["rotation"])
        rotation_pred = torch.argmax(rotation_logits, dim=1)
        rotation_acc = (rotation_pred == batch["rotation"]).float().mean()

        # Calculate total loss
        total_loss = (
            self.position_weight * position_loss + self.rotation_weight * rotation_loss
        )

        # Log metrics
        self.log("val/position_loss", position_loss, prog_bar=True)
        self.log("val/rotation_loss", rotation_loss, prog_bar=True)
        self.log("val/rotation_acc", rotation_acc, prog_bar=True)
        self.log("val/total_loss", total_loss, prog_bar=True)

        return {
            "val_loss": total_loss,
            "val_pos_loss": position_loss,
            "val_rot_loss": rotation_loss,
            "val_rot_acc": rotation_acc,
        }

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dict containing optimizer and scheduler config
        """
        optimizer = Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "frequency": 1,
            },
        }

    def predict_piece(
        self, piece_img: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Make prediction for a single puzzle piece.

        Args:
            piece_img: Preprocessed puzzle piece tensor of shape (1, 3, H, W)

        Returns:
            Tuple of (position, rotation_class, rotation_probabilities)
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            position_pred, rotation_logits = self(piece_img)

            # Get rotation class and probabilities
            rotation_probs = F.softmax(rotation_logits, dim=1)
            # Ensure rotation_class is an int
            rotation_class = int(torch.argmax(rotation_logits, dim=1).item())

            return position_pred[0], rotation_class, rotation_probs[0]
