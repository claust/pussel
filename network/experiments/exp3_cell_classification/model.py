"""Neural network models for cell classification experiment.

The key difference from single_puzzle_overfit is the output layer:
- Before: Linear(64, 2) + Sigmoid -> (cx, cy) coordinates
- Now: Linear(64, num_cells) -> logits for classification

The backbone architecture remains the same to isolate the effect of
the output formulation change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default grid dimensions for puzzle_001
DEFAULT_NUM_CELLS = 950


class CellClassifier(nn.Module):
    """Cell classification network for puzzle pieces.

    Same backbone as PieceLocNet from single_puzzle_overfit,
    but with a classification head instead of regression.

    Architecture:
        Input: 64x64x3
        Conv2D(16, 3x3, stride=2, ReLU, padding=1)  -> 32x32x16
        Conv2D(32, 3x3, stride=2, ReLU, padding=1)  -> 16x16x32
        Conv2D(64, 3x3, stride=2, ReLU, padding=1)  -> 8x8x64
        GlobalAveragePooling                         -> 64
        Linear(64, num_cells)                        -> logits
    """

    def __init__(self, num_cells: int = DEFAULT_NUM_CELLS):
        """Initialize the network.

        Args:
            num_cells: Number of output classes (grid cells).
        """
        super().__init__()
        self.num_cells = num_cells

        # Convolutional backbone (same as PieceLocNet)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16 -> 8

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.fc = nn.Linear(64, num_cells)

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


class CellClassifierDeep(nn.Module):
    """Cell classifier with deeper FC head.

    Same backbone as CellClassifier, but with additional fully connected
    layers before the classification head. This tests whether the bottleneck
    in attempt 1 was the direct 64 -> 950 mapping.

    The key insight: 64 features trying to distinguish 950 classes is too
    abrupt. By adding intermediate layers, we give the network room to
    progressively transform and organize the features.

    Architecture (~660K params):
        Input: 64x64x3
        Conv2D(16, 3x3, stride=2, ReLU, padding=1)  -> 32x32x16
        Conv2D(32, 3x3, stride=2, ReLU, padding=1)  -> 16x16x32
        Conv2D(64, 3x3, stride=2, ReLU, padding=1)  -> 8x8x64
        GlobalAveragePooling                         -> 64
        Linear(64, 256) + ReLU                       -> 256
        Linear(256, 512) + ReLU                      -> 512
        Linear(512, num_cells)                       -> logits

    Comparison:
        CellClassifier: 64 -> 950 (direct, 85K params)
        CellClassifierDeep: 64 -> 256 -> 512 -> 950 (progressive, 660K params)
    """

    def __init__(self, num_cells: int = DEFAULT_NUM_CELLS):
        """Initialize the network."""
        super().__init__()
        self.num_cells = num_cells

        # Same convolutional backbone as CellClassifier
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16 -> 8

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Deep classification head with progressive expansion
        # 64 -> 256 -> 512 -> 950
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_cells)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: piece image -> cell logits."""
        # Backbone (same as CellClassifier)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Deep classification head
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over cells."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted cell index."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class CellClassifierLarge(nn.Module):
    """Larger cell classifier with more capacity.

    If CellClassifier fails to achieve high accuracy, this provides
    more capacity to test whether the problem is model size.

    Architecture (~90K params):
        Input: 64x64x3
        Conv2D(32, 3x3, stride=2, ReLU, padding=1)  -> 32x32x32
        Conv2D(64, 3x3, stride=2, ReLU, padding=1)  -> 16x16x64
        Conv2D(128, 3x3, stride=2, ReLU, padding=1) -> 8x8x128
        Conv2D(256, 3x3, stride=2, ReLU, padding=1) -> 4x4x256
        GlobalAveragePooling                         -> 256
        Linear(256, 128)                             -> 128
        Linear(128, num_cells)                       -> logits
    """

    def __init__(self, num_cells: int = DEFAULT_NUM_CELLS):
        """Initialize the network."""
        super().__init__()
        self.num_cells = num_cells

        # Larger convolutional backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head with hidden layer
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_cells)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: piece image -> cell logits."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over cells."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted cell index."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_type: str = "cell_classifier", num_cells: int = DEFAULT_NUM_CELLS) -> nn.Module:
    """Factory function to get model by name.

    Args:
        model_type: One of "cell_classifier", "cell_classifier_deep", "cell_classifier_large".
        num_cells: Number of output classes.

    Returns:
        Instantiated model.
    """
    if model_type == "cell_classifier":
        return CellClassifier(num_cells=num_cells)
    elif model_type == "cell_classifier_deep":
        return CellClassifierDeep(num_cells=num_cells)
    elif model_type == "cell_classifier_large":
        return CellClassifierLarge(num_cells=num_cells)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("Model Architecture Tests")
    print("=" * 60)

    dummy_piece = torch.randn(2, 3, 64, 64)

    # Test CellClassifier
    print("\n1. CellClassifier (950 classes) - Attempt 1 baseline")
    model = CellClassifier(num_cells=950)
    print(f"   Parameters: {count_parameters(model):,}")
    print("   Architecture: 64 -> 950 (direct)")

    logits = model(dummy_piece)
    probs = model.predict_proba(dummy_piece)
    preds = model.predict(dummy_piece)

    print(f"   Input shape: {dummy_piece.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probs sum: {probs.sum(dim=1)}")

    # Test CellClassifierDeep
    print("\n2. CellClassifierDeep (950 classes) - Attempt 2")
    model_deep = CellClassifierDeep(num_cells=950)
    print(f"   Parameters: {count_parameters(model_deep):,}")
    print("   Architecture: 64 -> 256 -> 512 -> 950 (progressive)")

    logits_deep = model_deep(dummy_piece)
    print(f"   Output shape: {logits_deep.shape}")

    # Test CellClassifierLarge
    print("\n3. CellClassifierLarge (950 classes)")
    model_large = CellClassifierLarge(num_cells=950)
    print(f"   Parameters: {count_parameters(model_large):,}")
    print("   Architecture: Larger backbone (256 channels) + 256 -> 128 -> 950")

    logits_large = model_large(dummy_piece)
    print(f"   Output shape: {logits_large.shape}")

    # Compare parameter counts
    print("\n" + "=" * 60)
    print("Parameter comparison")
    print("=" * 60)
    print(f"  CellClassifier:      {count_parameters(model):>10,} params (attempt 1)")
    print(f"  CellClassifierDeep:  {count_parameters(model_deep):>10,} params (attempt 2)")
    print(f"  CellClassifierLarge: {count_parameters(model_large):>10,} params")
    print()
    print("  Note: PieceLocNet (regression) had ~23K params")
