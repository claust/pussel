# Model Architecture Review: PuzzleCNN

## Architecture Overview

The model is a dual-backbone CNN that takes both a puzzle piece and the complete
puzzle image as inputs to predict:

- **Position**: Bounding box coordinates (x1, y1, x2, y2) normalized to [0, 1]
- **Rotation**: Classification into 4 classes (0°, 90°, 180°, 270°)

```
┌─────────────────┐     ┌─────────────────┐
│  Piece Image    │     │  Puzzle Image   │
│   (128×128×3)   │     │   (128×128×3)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ ResNet18        │     │ ResNet18        │
│ Backbone        │     │ Backbone        │
│ (pretrained)    │     │ (pretrained)    │
└────────┬────────┘     └────────┬────────┘
         │ 512-dim               │ 512-dim
         └───────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │   Concatenate   │
           │   (1024-dim)    │
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │  Fusion Layer   │
           │ 1024→512 (ReLU) │
           │  Dropout(0.3)   │
           │ 512→512 (ReLU)  │
           │  Dropout(0.2)   │
           └────────┬────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Position Head  │     │  Rotation Head  │
│ 512→256 (ReLU)  │     │ 512→256 (ReLU)  │
│  Dropout(0.1)   │     │  Dropout(0.1)   │
│ 256→4 (Sigmoid) │     │     256→4       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
   (x1,y1,x2,y2)          4-class logits
```

## Current Configuration (config.py)

| Parameter           | Value    | Notes               |
| ------------------- | -------- | ------------------- |
| Backbone            | ResNet18 | Lightweight, fast   |
| Pretrained          | True     | ImageNet weights    |
| Learning Rate       | 3e-3     | Relatively high     |
| Piece Size          | 128×128  | Reduced from 224    |
| Puzzle Size         | 128×128  | Reduced from 224    |
| Batch Size          | 32       | Standard            |
| Max Epochs          | 100      | With early stopping |
| Early Stop Patience | 10       | epochs              |

## Identified Issues & Causes for Poor Performance

### 1. Information Loss from Aggressive Downscaling (Critical)

| Issue               | Details                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Puzzle size 128×128 | Full puzzle images are resized to 128×128, losing spatial detail                                 |
| Impact              | The model cannot distinguish fine-grained positions because the puzzle context is too compressed |

**Why it matters:** If a puzzle piece originally occupies a 50×50 region in a
1000×1000 puzzle, after resizing to 128×128, that region becomes ~6×6
pixels—barely distinguishable.

### 2. Same Resolution for Piece and Puzzle (High)

Both piece and puzzle are resized to 128×128. This creates a scale mismatch:

- A puzzle piece that was 5% of the original image becomes 100% of the input
- The puzzle that was 100% also becomes 100%
- The model loses relative scale information

### 3. Simple Concatenation Fusion (Medium)

The fusion mechanism just concatenates features:

```python
combined = torch.cat([piece_features, puzzle_features], dim=1)
```

This doesn't explicitly model the spatial relationship between the piece and
puzzle. The model has to learn this relationship purely from the FC layers.

### 4. No Spatial Attention Mechanism (Medium)

The model uses global average pooling in the ResNet backbone, discarding spatial
information. For position prediction, knowing where features are located is
crucial.

### 5. Position Regression with MSE Loss (Medium)

MSE loss treats all coordinate errors equally. A prediction of (0.1, 0.1, 0.2,
0.2) vs ground truth (0.5, 0.5, 0.6, 0.6) has the same loss structure as (0.1,
0.1, 0.2, 0.2) vs (0.15, 0.15, 0.25, 0.25), but the first is a completely wrong
location while the second is nearly correct.

### 6. Limited Data Augmentation (Medium)

Current augmentations (dataset.py:225-238):

- `RandomBrightnessContrast(p=0.5)`
- `HueSaturationValue(p=0.3)`

Missing augmentations that could help:

- Cutout/RandomErasing
- Gaussian noise
- JPEG compression artifacts
- More aggressive color jittering

### 7. Small Dataset (High - Based on Your Feedback)

You mentioned training on only a few images. For a dual-backbone CNN with ~22M
parameters (ResNet18 × 2), you typically need:

- **Minimum:** 1,000-5,000 training samples
- **Recommended:** 10,000+ samples
- **Ideal:** 50,000+ samples with diverse puzzles

## Recommendations for Improvement

### Priority 1: Increase Training Data

| Action                      | Expected Impact |
| --------------------------- | --------------- |
| Generate more puzzle pieces | High            |
| Use multiple source puzzles | High            |
| Apply heavy augmentation    | Medium-High     |

### Priority 2: Fix Resolution Asymmetry

```python
# config.py - suggested change
"data": {
    "piece_size": (128, 128),   # Keep piece small
    "puzzle_size": (384, 384),  # Increase puzzle resolution
}
```

### Priority 3: Add Spatial Attention

Replace simple concatenation with a cross-attention mechanism:

```python
# Suggested architecture change
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, piece_feat, puzzle_feat):
        q = self.query(piece_feat)
        k = self.key(puzzle_feat)
        v = self.value(puzzle_feat)
        attn = F.softmax(q @ k.T / math.sqrt(dim), dim=-1)
        return attn @ v
```

### Priority 4: Use IoU Loss for Position

Replace MSE with GIoU or DIoU loss for better bounding box regression:

```python
# In model.py training_step
from torchvision.ops import generalized_box_iou_loss

position_loss = generalized_box_iou_loss(position_pred, batch["position"])
```

### Priority 5: Consider Feature Pyramid for Multi-Scale

Use FPN-style features to capture both fine and coarse details:

```python
# Replace backbone with feature pyramid
self.backbone = timm.create_model(
    backbone_name,
    pretrained=pretrained,
    features_only=True,      # Returns multi-scale features
    out_indices=[2, 3, 4]
)
```

### Priority 6: Add More Augmentations

```python
# In dataset.py
self.train_piece_transform = A.Compose([
    A.Resize(height=self.piece_size[0], width=self.piece_size[1]),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.2),                            # NEW
    A.CoarseDropout(max_holes=8, p=0.3),            # NEW
    A.ImageCompression(quality_lower=70, p=0.2),   # NEW
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Summary of Issues

| Issue                               | Severity | Quick Fix Available   |
| ----------------------------------- | -------- | --------------------- |
| Insufficient training data          | Critical | Generate more pieces  |
| Puzzle resolution too low (128×128) | High     | Change config         |
| No spatial attention                | Medium   | Requires code changes |
| MSE loss for positions              | Medium   | Replace with GIoU     |
| Simple concatenation fusion         | Medium   | Add attention         |
| Limited augmentation                | Medium   | Extend transforms     |
