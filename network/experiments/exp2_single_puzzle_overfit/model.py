"""Neural network models for single puzzle overfit experiment.

This module provides two model options:
1. PieceLocNet: Piece-only encoder (same architecture as baseline TinyLocNet)
2. DualEncoderNet: Dual encoder with piece and puzzle branches

We start with PieceLocNet to test pure memorization capability.
"""

import torch
import torch.nn as nn


class PieceLocNet(nn.Module):
    """Piece-only encoder for position prediction.

    Same architecture as baseline TinyLocNet - the piece image goes in,
    and the network must learn to output where that piece belongs.

    This tests whether the network can memorize the mapping:
    piece_texture -> (cx, cy) position

    Architecture:
        Input: 64x64x3
        Conv2D(16, 3x3, stride=2, ReLU, padding=1)  -> 32x32x16
        Conv2D(32, 3x3, stride=2, ReLU, padding=1)  -> 16x16x32
        Conv2D(64, 3x3, stride=2, ReLU, padding=1)  -> 8x8x64
        GlobalAveragePooling                         -> 64
        Linear(64, 2)                                -> (cx, cy)
        Sigmoid                                      -> output in [0, 1]
    """

    def __init__(self):
        """Initialize the network layers."""
        super().__init__()

        # Convolutional layers (same as baseline)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16 -> 8

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Final linear layer
        self.fc = nn.Linear(64, 2)

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: piece image -> (cx, cy) coordinates.

        Args:
            x: Piece image tensor of shape (batch, 3, 64, 64).

        Returns:
            Position tensor of shape (batch, 2) with values in [0, 1].
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class PieceLocNetLarge(nn.Module):
    """Larger piece encoder with more capacity.

    If PieceLocNet fails to memorize, this provides more capacity
    to test whether the problem is model size vs. task difficulty.

    Architecture (~60K params):
        Input: 64x64x3
        Conv2D(32, 3x3, stride=2, ReLU, padding=1)  -> 32x32x32
        Conv2D(64, 3x3, stride=2, ReLU, padding=1)  -> 16x16x64
        Conv2D(128, 3x3, stride=2, ReLU, padding=1) -> 8x8x128
        Conv2D(256, 3x3, stride=2, ReLU, padding=1) -> 4x4x256
        GlobalAveragePooling                         -> 256
        Linear(256, 64)                              -> 64
        Linear(64, 2)                                -> (cx, cy)
        Sigmoid                                      -> output in [0, 1]
    """

    def __init__(self):
        """Initialize the network layers."""
        super().__init__()

        # Convolutional layers (larger)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MLP head
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: piece image -> (cx, cy) coordinates."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class DualEncoderNet(nn.Module):
    """Dual encoder with piece and puzzle branches.

    This is closer to the main model architecture where both
    the piece and puzzle images are processed.

    Architecture:
        Piece branch (64x64):
            Conv layers -> feature vector (64-dim)
        Puzzle branch (256x256):
            Conv layers -> feature vector (64-dim)
        Fusion:
            Concatenate -> 128-dim
            Linear(128, 64) -> ReLU
            Linear(64, 2) -> Sigmoid -> (cx, cy)
    """

    def __init__(self, piece_size: int = 64, puzzle_size: int = 256):
        """Initialize the dual encoder.

        Args:
            piece_size: Expected piece image size (default 64).
            puzzle_size: Expected puzzle image size (default 256).
        """
        super().__init__()
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size

        # Piece encoder (64x64 input)
        self.piece_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.piece_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.piece_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Puzzle encoder (256x256 input)
        self.puzzle_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 128
        self.puzzle_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 64
        self.puzzle_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32
        self.puzzle_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 16
        self.puzzle_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 8

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fusion layers
        self.fusion_fc1 = nn.Linear(128, 64)  # 64 (piece) + 64 (puzzle)
        self.fusion_fc2 = nn.Linear(64, 2)

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode_piece(self, x: torch.Tensor) -> torch.Tensor:
        """Encode piece image to feature vector."""
        x = self.relu(self.piece_conv1(x))
        x = self.relu(self.piece_conv2(x))
        x = self.relu(self.piece_conv3(x))
        x = self.gap(x)
        return x.view(x.size(0), -1)

    def encode_puzzle(self, x: torch.Tensor) -> torch.Tensor:
        """Encode puzzle image to feature vector."""
        x = self.relu(self.puzzle_conv1(x))
        x = self.relu(self.puzzle_conv2(x))
        x = self.relu(self.puzzle_conv3(x))
        x = self.relu(self.puzzle_conv4(x))
        x = self.relu(self.puzzle_conv5(x))
        x = self.gap(x)
        return x.view(x.size(0), -1)

    def forward(self, piece: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
        """Forward pass: (piece, puzzle) -> (cx, cy) coordinates.

        Args:
            piece: Piece image tensor of shape (batch, 3, 64, 64).
            puzzle: Puzzle image tensor of shape (batch, 3, 256, 256).

        Returns:
            Position tensor of shape (batch, 2) with values in [0, 1].
        """
        piece_features = self.encode_piece(piece)
        puzzle_features = self.encode_puzzle(puzzle)

        # Concatenate features
        combined = torch.cat([piece_features, puzzle_features], dim=1)

        # Fusion MLP
        x = self.relu(self.fusion_fc1(combined))
        x = self.fusion_fc2(x)
        x = self.sigmoid(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_type: str = "piece_loc") -> nn.Module:
    """Factory function to get model by name.

    Args:
        model_type: One of "piece_loc", "piece_loc_large", "dual_encoder".

    Returns:
        Instantiated model.
    """
    models = {
        "piece_loc": PieceLocNet,
        "piece_loc_large": PieceLocNetLarge,
        "dual_encoder": DualEncoderNet,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type]()


if __name__ == "__main__":
    print("=" * 60)
    print("Model Architecture Tests")
    print("=" * 60)

    # Test PieceLocNet
    print("\n1. PieceLocNet (baseline equivalent)")
    model = PieceLocNet()
    print(f"   Parameters: {count_parameters(model):,}")
    dummy_piece = torch.randn(2, 3, 64, 64)
    output = model(dummy_piece)
    print(f"   Input shape: {dummy_piece.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Test PieceLocNetLarge
    print("\n2. PieceLocNetLarge (more capacity)")
    model_large = PieceLocNetLarge()
    print(f"   Parameters: {count_parameters(model_large):,}")
    output_large = model_large(dummy_piece)
    print(f"   Input shape: {dummy_piece.shape}")
    print(f"   Output shape: {output_large.shape}")

    # Test DualEncoderNet
    print("\n3. DualEncoderNet (piece + puzzle)")
    model_dual = DualEncoderNet()
    print(f"   Parameters: {count_parameters(model_dual):,}")
    dummy_puzzle = torch.randn(2, 3, 256, 256)
    output_dual = model_dual(dummy_piece, dummy_puzzle)
    print(f"   Piece input shape: {dummy_piece.shape}")
    print(f"   Puzzle input shape: {dummy_puzzle.shape}")
    print(f"   Output shape: {output_dual.shape}")
    print(f"   Output range: [{output_dual.min().item():.3f}, {output_dual.max().item():.3f}]")
