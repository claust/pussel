"""Neural network models for larger backbone cell classification experiment.

This experiment tests Option A from exp3: Increase Backbone Capacity Moderately.

The hypothesis is that the 64-dim feature space from the simple backbone is
insufficient for 950-class classification, even though it worked for regression.
By expanding to 256-dim features, we provide more room for class separation.

Architecture comparison:
- exp3 CellClassifier: 3→16→32→64 (23K params, 64-dim features)
- exp4 LargerBackbone: 3→32→64→128→256 (150K params, 256-dim features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default grid dimensions for puzzle_001
DEFAULT_NUM_CELLS = 950


class CellClassifierLargerBackbone(nn.Module):
    """Cell classifier with larger backbone for more expressive features.

    This tests whether the bottleneck in exp3 was the feature dimensionality.
    We increase the backbone from 64-dim to 256-dim features while keeping
    a direct classification head (no deep FC layers, which failed in exp3).

    Architecture (~150K backbone params + 244K head params = ~394K total):
        Input: 64x64x3
        Conv2D(32, 3x3, stride=2, ReLU, padding=1)  -> 32x32x32
        Conv2D(64, 3x3, stride=2, ReLU, padding=1)  -> 16x16x64
        Conv2D(128, 3x3, stride=2, ReLU, padding=1) -> 8x8x128
        Conv2D(256, 3x3, stride=2, ReLU, padding=1) -> 4x4x256
        GlobalAveragePooling                         -> 256
        Linear(256, num_cells)                       -> logits (950)

    Design choices:
    - Direct 256 → 950 classification (not deep FC, which failed in exp3)
    - 4 conv layers instead of 3 (more depth for richer features)
    - Double the channel width at each layer (32→64→128→256)
    - No batch normalization (keep it simple, like exp3)
    """

    def __init__(self, num_cells: int = DEFAULT_NUM_CELLS):
        """Initialize the network.

        Args:
            num_cells: Number of output classes (grid cells).
        """
        super().__init__()
        self.num_cells = num_cells

        # Larger convolutional backbone (4 layers, 256-dim output)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8 -> 4

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Direct classification head (like exp3 attempt 1, but with 256 features)
        self.fc = nn.Linear(256, num_cells)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: piece image -> cell logits.

        Args:
            x: Piece image tensor of shape (batch, 3, 64, 64).

        Returns:
            Logits tensor of shape (batch, num_cells).
            Use F.softmax(output, dim=1) to get probabilities.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # No activation - cross_entropy expects raw logits

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over cells.

        Args:
            x: Piece image tensor.

        Returns:
            Probability tensor of shape (batch, num_cells).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted cell index.

        Args:
            x: Piece image tensor.

        Returns:
            Predicted cell indices of shape (batch,).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_backbone_parameters(model: CellClassifierLargerBackbone) -> int:
    """Count parameters in the backbone only (conv layers)."""
    backbone_params = 0
    for name, param in model.named_parameters():
        if "conv" in name:
            backbone_params += param.numel()
    return backbone_params


def get_model(model_type: str = "larger_backbone", num_cells: int = DEFAULT_NUM_CELLS) -> nn.Module:
    """Factory function to get model by name.

    Args:
        model_type: One of "larger_backbone".
        num_cells: Number of output classes.

    Returns:
        Instantiated model.
    """
    if model_type == "larger_backbone":
        return CellClassifierLargerBackbone(num_cells=num_cells)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("Model Architecture Test - Larger Backbone")
    print("=" * 60)

    dummy_piece = torch.randn(2, 3, 64, 64)

    # Test CellClassifierLargerBackbone
    print("\nCellClassifierLargerBackbone (950 classes)")
    model = CellClassifierLargerBackbone(num_cells=950)
    total_params = count_parameters(model)
    backbone_params = count_backbone_parameters(model)
    head_params = total_params - backbone_params

    print(f"  Total parameters: {total_params:,}")
    print(f"  Backbone parameters: {backbone_params:,}")
    print(f"  Head parameters: {head_params:,}")
    print("  Architecture: 3 -> 32 -> 64 -> 128 -> 256 (backbone)")
    print("  Classification: 256 -> 950 (direct)")

    logits = model(dummy_piece)
    probs = model.predict_proba(dummy_piece)
    preds = model.predict(dummy_piece)

    print(f"\n  Input shape: {dummy_piece.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probs sum: {probs.sum(dim=1)}")
    print(f"  Predictions: {preds.tolist()}")

    # Compare with exp3 models
    print("\n" + "=" * 60)
    print("Comparison with exp3 models")
    print("=" * 60)
    print("  exp3 CellClassifier:      ~85,334 params (64-dim features)")
    print("  exp3 CellClassifierDeep: ~659,158 params (64-dim, deep FC)")
    print("  exp3 CellClassifierLarge:~380,000 params (256-dim, 256->128->950)")
    print(f"  exp4 LargerBackbone:     {total_params:>8,} params (256-dim, direct)")
    print()
