# CNN-Based Puzzle Piece Prediction: POC Implementation Plan

## Project Structure
```
pussel/
└── network/
    ├── puzzle_generator.py  # Existing preprocessing script
    ├── specification.md     # Existing specification
    ├── model.py             # LightningModule with timm backbone ✓
    ├── dataset.py           # Dataset handling with Albumentations ✓
    ├── train.py             # Training script with Lightning Trainer ✓
    └── config.py            # Configuration parameters ✓
```

## Implementation Tasks

### 1. Setup Project & Dependencies
- [x] Virtual environment is ready
- [x] Install required packages:
  ```
  torch torchvision
  pytorch-lightning
  timm
  albumentations
  pillow
  pandas
  matplotlib
  ```

### 2. Dataset Implementation (network/dataset.py)
- [x] Create PyTorch Dataset class that reads from metadata.csv
- [x] Implement basic transforms using Albumentations
- [x] Create DataModule for Lightning with train/val loading

### 3. Model Implementation (network/model.py)
- [x] Create PuzzleCNN LightningModule with timm ResNet50 backbone
- [x] Implement position prediction head (bounding box coordinates)
- [x] Implement rotation prediction head (4-class classification)
- [x] Define loss functions (MSE, CrossEntropy)
- [x] Add training_step, validation_step with metrics

### 4. Training Setup (network/train.py)
- [x] Configure Lightning Trainer
- [x] Add callbacks (model checkpoint, early stopping)
- [x] Implement simple logging of metrics
- [x] Add CLI arguments for key parameters

### 5. Configuration (network/config.py)
- [x] Model configuration (backbone, head dims)
- [x] Training parameters
- [x] Data paths and parameters

### 6. Initial Testing & Visualization
- [x] Add simple inference function in model.py
- [ ] Implement basic visualization to show predictions on puzzle

## Timeline
- Dataset implementation: ✓ Done
- Model implementation: ✓ Done
- Training setup: ✓ Done
- Configuration: ✓ Done
- Initial testing & visualization: Partially done (1/2)

## Future Improvements (Post-POC)
- Experiment with different backbones
- Implement more sophisticated data augmentation
- Add detailed evaluation metrics
- Create standalone prediction script
