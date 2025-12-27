# Multi-Scale Architecture Plan

## Problem Summary

The current architecture has a catastrophic scale mismatch:
- Puzzle images (3000×2000 px) are compressed to 256×256
- Piece images (~79×80 px) are expanded to 224×224
- A piece that represents 0.1% of the puzzle appears as 76% of the puzzle's size after resizing
- The spatial correlation module's feature map resolution (~7×7) is coarser than the target location (~2.6% × 4%)

## Proposed Solution: Multi-Scale Puzzle Processing

### Core Idea

1. **Keep the puzzle at higher resolution** to preserve spatial detail
2. **Use a Feature Pyramid Network (FPN)** to extract multi-scale features
3. **Correlate piece features at multiple scales** to find the best match
4. **Return position from the scale with highest correlation**

### Architecture Overview

```
                    ┌─────────────────┐
                    │   Piece Image   │
                    │   (224 × 224)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Piece Backbone │
                    │   (ResNet18)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  P3 feat  │  │  P4 feat  │  │  P5 feat  │
        │  (28×28)  │  │  (14×14)  │  │  (7×7)    │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              │         CORRELATION         │
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │ Puzzle P3 │  │ Puzzle P4 │  │ Puzzle P5 │
        │  (64×64)  │  │  (32×32)  │  │  (16×16)  │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
                    ┌────────▼────────┐
                    │ Puzzle Backbone │
                    │  (ResNet18 FPN) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Puzzle Image   │
                    │   (512 × 512)   │
                    └─────────────────┘
```

## Implementation Details

### 1. Configuration Changes (`config.py`)

```python
"data": {
    "piece_size": (224, 224),      # Keep piece size (captures detail)
    "puzzle_size": (512, 512),     # Increase from 256 to 512
}

"model": {
    "backbone_name": "resnet18",   # Lighter backbone for FPN
    "use_fpn": True,               # Enable Feature Pyramid Network
    "fpn_channels": 256,           # FPN channel dimension
    "correlation_scales": [3, 4, 5],  # P3, P4, P5 pyramid levels
}
```

### 2. Feature Pyramid Network Module

Add a new FPN module that extracts multi-scale features:

```python
class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""

    def __init__(self, backbone_name: str, fpn_channels: int = 256):
        super().__init__()

        # Create backbone with multiple output stages
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=[2, 3, 4],  # C3, C4, C5 stages
        )

        # Get channel dimensions for each stage
        dummy = torch.zeros(1, 3, 512, 512)
        features = self.backbone(dummy)
        self.in_channels = [f.shape[1] for f in features]

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_channels, 1)
            for in_ch in self.in_channels
        ])

        # Output convolutions (3x3 conv to smooth features)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
            for _ in self.in_channels
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            List of feature maps [P3, P4, P5] from fine to coarse
        """
        # Get backbone features
        features = self.backbone(x)  # [C3, C4, C5]

        # Build FPN top-down pathway
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down fusion (coarse to fine)
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode='nearest'
            )
            laterals[i] = laterals[i] + upsampled

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return outputs  # [P3, P4, P5]
```

### 3. Multi-Scale Correlation Module

Replace the current `SpatialCorrelationModule` with a multi-scale version:

```python
class MultiScaleCorrelation(nn.Module):
    """Correlate piece features against puzzle at multiple scales."""

    def __init__(self, fpn_channels: int = 256):
        super().__init__()

        self.fpn_channels = fpn_channels

        # Learnable temperature per scale
        self.temperatures = nn.ParameterList([
            nn.Parameter(torch.ones(1) * math.sqrt(fpn_channels))
            for _ in range(3)
        ])

        # Process correlation maps at each scale
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            for _ in range(3)
        ])

        # Fuse multi-scale correlation features
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Position regression from correlation features
        self.position_regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        piece_features: List[torch.Tensor],  # [P3, P4, P5] from piece
        puzzle_features: List[torch.Tensor], # [P3, P4, P5] from puzzle
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute multi-scale correlation and predict position.

        Args:
            piece_features: List of piece feature maps at different scales
            puzzle_features: List of puzzle feature maps at different scales

        Returns:
            position_pred: Predicted (x1, y1, x2, y2) normalized coordinates
            correlation_maps: List of correlation maps for visualization
        """
        batch_size = piece_features[0].shape[0]
        correlation_maps = []
        processed_correlations = []

        for i, (piece_feat, puzzle_feat, temp) in enumerate(
            zip(piece_features, puzzle_features, self.temperatures)
        ):
            # Global average pool piece to get descriptor
            # Shape: (B, C, H, W) -> (B, C)
            piece_desc = piece_feat.mean(dim=[2, 3])

            # Compute correlation with puzzle spatial locations
            B, C, H, W = puzzle_feat.shape
            puzzle_flat = puzzle_feat.view(B, C, H * W)  # (B, C, H*W)

            # Normalized dot product correlation
            piece_norm = F.normalize(piece_desc, dim=1)  # (B, C)
            puzzle_norm = F.normalize(puzzle_flat, dim=1)  # (B, C, H*W)

            correlation = torch.bmm(
                piece_norm.unsqueeze(1),  # (B, 1, C)
                puzzle_norm               # (B, C, H*W)
            )  # (B, 1, H*W)

            correlation = correlation / temp
            correlation_map = correlation.view(B, 1, H, W)
            correlation_maps.append(correlation_map)

            # Apply softmax to get attention weights
            correlation_softmax = F.softmax(
                correlation_map.view(B, -1), dim=-1
            ).view(B, 1, H, W)

            # Process correlation map
            processed = self.scale_processors[i](correlation_softmax)

            # Upsample to common resolution (largest scale)
            if i > 0:
                target_size = puzzle_features[0].shape[-2:]
                processed = F.interpolate(
                    processed, size=target_size, mode='bilinear', align_corners=False
                )

            processed_correlations.append(processed)

        # Concatenate multi-scale correlation features
        fused = torch.cat(processed_correlations, dim=1)  # (B, 64*3, H, W)

        # Fuse and regress position
        fused = self.fusion(fused)  # (B, 128, 1, 1)
        fused = fused.view(batch_size, -1)  # (B, 128)

        position_pred = self.position_regressor(fused)

        return position_pred, correlation_maps
```

### 4. Updated PuzzleCNN Model

```python
class PuzzleCNN(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        fpn_channels: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Shared FPN backbone for both piece and puzzle
        self.fpn = FeaturePyramidNetwork(backbone_name, fpn_channels)

        # Multi-scale correlation for position prediction
        self.correlation = MultiScaleCorrelation(fpn_channels)

        # Rotation head (uses coarsest scale features)
        self.rotation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_channels * 2, 256),  # Concat piece + puzzle
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
        )

    def forward(self, piece_img, puzzle_img):
        # Extract multi-scale features
        piece_features = self.fpn(piece_img)    # [P3, P4, P5]
        puzzle_features = self.fpn(puzzle_img)  # [P3, P4, P5]

        # Position prediction via multi-scale correlation
        position_pred, correlation_maps = self.correlation(
            piece_features, puzzle_features
        )

        # Rotation prediction (use coarsest features)
        piece_global = piece_features[-1].mean(dim=[2, 3])
        puzzle_global = puzzle_features[-1].mean(dim=[2, 3])
        rotation_input = torch.cat([piece_global, puzzle_global], dim=1)
        rotation_logits = self.rotation_head(
            rotation_input.view(rotation_input.size(0), -1, 1, 1)
        )

        return position_pred, rotation_logits
```

### 5. Dataset Changes (`dataset.py`)

Update the data module to use larger puzzle images:

```python
class PuzzleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        piece_size: Tuple[int, int] = (224, 224),
        puzzle_size: Tuple[int, int] = (512, 512),  # Increased from 256
    ):
        # ... rest unchanged
```

## Feature Map Resolution Analysis

With the proposed changes:

| Level | Puzzle Input | Feature Map | Each Cell Covers |
|-------|-------------|-------------|------------------|
| P3    | 512×512     | 64×64       | 1.56% × 1.56%    |
| P4    | 512×512     | 32×32       | 3.12% × 3.12%    |
| P5    | 512×512     | 16×16       | 6.25% × 6.25%    |

Target piece size: ~2.6% × 4%

**P3 (64×64) can now resolve the piece location!** The piece spans approximately 1-2 cells in P3, which is reasonable for localization.

## Training Considerations

### Loss Function

Keep the hybrid CIoU + SmoothL1 loss, but the multi-scale correlation should provide much better gradients:

```python
def training_step(self, batch, batch_idx):
    position_pred, rotation_logits = self(batch["piece"], batch["puzzle"])

    # Position loss
    ciou_loss = complete_box_iou_loss(position_pred, batch["position"])
    l1_loss = F.smooth_l1_loss(position_pred, batch["position"])
    position_loss = 0.5 * ciou_loss + 0.5 * l1_loss

    # Rotation loss
    rotation_loss = F.cross_entropy(rotation_logits, batch["rotation"])

    # Log correlation maps for debugging (optional)
    if batch_idx % 100 == 0:
        self._log_correlation_maps(correlation_maps, batch_idx)

    return position_loss + rotation_loss
```

### Learning Rate

Start with 1e-4, use LR finder to tune. The FPN should train more stably than the previous architecture.

### Batch Size

May need to reduce batch size due to larger puzzle images (512×512 vs 256×256). Try:
- Batch size 16 if GPU memory is limited
- Gradient accumulation if needed

## Memory Considerations

| Component | Old | New | Change |
|-----------|-----|-----|--------|
| Puzzle tensor | 256×256×3 | 512×512×3 | 4× larger |
| Feature maps | ~7×7 | 16×16 to 64×64 | ~8-80× larger |
| Estimated batch | 32 | 16 | 2× smaller |

For a GPU with 8GB VRAM, batch size 16 should be feasible.

## Implementation Steps

1. **Add FPN module** to `model.py`
2. **Add MultiScaleCorrelation module** to `model.py`
3. **Update PuzzleCNN** to use new modules
4. **Update config.py** with new default sizes
5. **Update dataset.py** puzzle_size default
6. **Add correlation map logging** for debugging
7. **Run training** with reduced batch size
8. **Monitor correlation maps** in TensorBoard

## Expected Improvements

1. **No more prediction collapse** - correlation provides explicit spatial signal
2. **Better gradient flow** - multi-scale features provide gradients at appropriate resolution
3. **Interpretable** - correlation maps show where the model is looking
4. **Scale-aware** - different pyramid levels handle different piece sizes

## Fallback Options

If this approach still struggles:

1. **Increase puzzle resolution further** (768×768 or 1024×1024)
2. **Use deformable convolutions** in the correlation module
3. **Add explicit position encoding** (sinusoidal embeddings)
4. **Use a detection head** (like FCOS or CenterNet) instead of direct regression
