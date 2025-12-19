"""
Simplest possible CNN for bounding box regression (~15K params).

Architecture:
    Input: 64x64x3
    Conv2D(16, 3x3, stride=2, ReLU, padding=1)  → 32x32x16
    Conv2D(32, 3x3, stride=2, ReLU, padding=1)  → 16x16x32
    Conv2D(64, 3x3, stride=2, ReLU, padding=1)  → 8x8x64
    GlobalAveragePooling                         → 64
    Linear(64, 2)                                → (x, y)
    Sigmoid                                      → output in [0, 1]

No BatchNorm. No Dropout. No skip connections. Just the bare minimum.
"""

import torch
import torch.nn as nn


class TinyLocNet(nn.Module):
    """Minimal CNN for square localization."""

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16 -> 8

        # Global average pooling (done in forward)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Final linear layer
        self.fc = nn.Linear(64, 2)

        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv layers with ReLU
        x = self.relu(self.conv1(x))  # 64x64x3 -> 32x32x16
        x = self.relu(self.conv2(x))  # 32x32x16 -> 16x16x32
        x = self.relu(self.conv3(x))  # 16x16x32 -> 8x8x64

        # Global average pooling
        x = self.gap(x)  # 8x8x64 -> 1x1x64
        x = x.view(x.size(0), -1)  # Flatten: (batch, 64)

        # Linear + Sigmoid for [0, 1] output
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = TinyLocNet()
    print(f"Model architecture:\n{model}\n")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
